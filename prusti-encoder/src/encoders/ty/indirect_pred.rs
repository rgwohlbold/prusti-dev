use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, FunctionIdn, PredicateIdn};

use crate::encoders::TyUseImpureEnc;

use super::{
    RustTy, RustTyDecomposition,
    data::TySpecifics,
    generics::{
        GArgsTyEnc, GenericParamsEnc, TyExprEnc,
        arity::{ArityEnc, add},
        clamp::ClampEnc,
        lifetime_position::{LifetimePosition, LifetimePositionEnc, Occurrence},
    },
    pure::TyPureEnc,
};

pub struct IndirectPredEnc;

#[derive(Debug, Clone, Copy)]
pub struct IndirectPredRef<'vir> {
    pub predicate: PredicateIdn<
        'vir,
        (
            vir::Snap,
            vir::ManyTyVal,
            vir::ManyCSnap,
            vir::Int,
            vir::Int,
        ),
    >,
    /// snapshot extension function `ext_X(snapshot, T_0, ..., pred, lo, hi)` which extends
    /// `snapshot` by the lifetime positions `[lo, hi)`, whose permissions are held
    /// by `p_X_ind(pred, T_0, ..., lo, hi)`.
    pub ext: FunctionIdn<
        'vir,
        (
            vir::Snap,
            vir::ManyTyVal,
            vir::ManyCSnap,
            vir::Snap,
            vir::Int,
            vir::Int,
        ),
        vir::Snap,
    >,
}

impl<'vir> task_encoder::OutputRefAny for IndirectPredRef<'vir> {}

// Lifetime query that corresponds to a lifetime position [lo, hi), either lifetime or type parameter
#[derive(Clone, Copy)]
struct Subquery<'vir> {
    param: LifetimePosition<'vir>,
    guard: vir::ExprBool<'vir>,
    start: vir::ExprInt<'vir>,
    end: vir::ExprInt<'vir>,
}

/// A field of a struct or of an enum variant: its type and its reader.
type Field<'vir> = (
    RustTyDecomposition<'vir>,
    vir::AdtDestructor<'vir, vir::CSnap, vir::Snap>,
);

impl IndirectPredEnc {
    /// The subqueries of the query `[lo, hi)` for lifetime and type parameters of `ty`.
    fn subqueries<'vir>(
        ty: RustTy<'vir>,
        lo: vir::ExprInt<'vir>,
        hi: vir::ExprInt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<Subquery<'vir>>, EncodeFullError<'vir, Self>> {
        let arity = deps.require_ref::<ArityEnc>(ty)?.arity;
        let clamp = deps.require_dep::<ClampEnc>(())?;
        let own_positions =
            deps.require_dep::<LifetimePositionEnc>(RustTyDecomposition::identity(ty))?;
        vir::with_vcx(|vcx| {
            let sub = |a: vir::ExprInt<'vir>, b: vir::ExprInt<'vir>| -> vir::ExprInt<'vir> {
                vcx.mk_bin_op_expr(vir::BinOpKind::Sub, a, b).downcast_ty()
            };
            own_positions
                .into_iter()
                .map(
                    |Occurrence {
                         param, start: x, ..
                     }| {
                        Ok(match param {
                            // An own lifetime has the single position `x`, which the query either contains or not.
                            LifetimePosition::Lifetime(_) => Subquery {
                                param,
                                guard: vir::expr! { vcx; ((lo) <= (x)) && ((x) < (hi)) },
                                start: vcx.mk_uint::<0>(),
                                end: vcx.mk_uint::<1>(),
                            },
                            // The subquery of an own type parameter at the positions `[x, x + w)` is `[clamp(lo - x, 0, w), clamp(hi - x, 0, w))`.
                            LifetimePosition::Param(ty) => {
                                let w = arity.call()(deps.require_dep::<TyExprEnc>(ty)?);
                                let zero = vcx.mk_uint::<0>();
                                let start = clamp.call()(sub(lo, x), zero, w);
                                let end = clamp.call()(sub(hi, x), zero, w);
                                Subquery {
                                    param,
                                    guard: vir::expr! { vcx; (start) < (end) },
                                    start,
                                    end,
                                }
                            }
                        })
                    },
                )
                .collect()
        })
    }

    /// The snapshot on which the field predicate of the occurrence `index`
    /// in the field type `field_ty` is defined: `field_snap`, extended by the
    /// enclosing occurrences whose parameters the query contains, outermost
    /// first. The caller has already extended `field_snap` by the enclosing
    /// occurrences of the other parameters.
    fn occurrence_snap<'vir>(
        subqueries: &[Subquery<'vir>],
        field_ty: RustTyDecomposition<'vir>,
        occurrences: &[Occurrence<'vir>],
        index: usize,
        field_snap: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        let mut snap = field_snap;
        for &enclosing in &occurrences[index].enclosing {
            let occurrence = &occurrences[enclosing];
            let Some(&subquery) = subqueries.iter().find(|s| s.param == occurrence.param) else {
                continue;
            };
            let key = Self::occurrence_snap(
                subqueries,
                field_ty,
                occurrences,
                enclosing,
                field_snap,
                deps,
            )?;
            let ext = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.ext;
            let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
            let guard = subquery.guard;
            let start = add(occurrence.start, subquery.start);
            let end = add(occurrence.start, subquery.end);
            let extended = ext.call()(snap, args.get_ty(), args.get_const(), key, start, end);
            // An enclosing occurrence of the same parameter is in the query
            // whenever the occurrence `index` is.
            snap = if occurrence.param == occurrences[index].param {
                extended
            } else {
                vir::expr! { (guard) ? (extended) : (snap) }
            };
        }
        Ok(snap)
    }

    /// The predicates of the field of type `field_ty` with the snapshot
    /// `field_snap` for `subquery`: one for each occurrence of the subquery's
    /// parameter in the field type.
    fn field_preds<'vir>(
        subqueries: &[Subquery<'vir>],
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        field_snap: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprBool<'vir>>, EncodeFullError<'vir, Self>> {
        let occurrences = deps.require_dep::<LifetimePositionEnc>(field_ty)?;
        let mut preds = Vec::new();
        for (index, occurrence) in occurrences.iter().enumerate() {
            if occurrence.param != subquery.param {
                continue;
            }
            let pred = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.predicate;
            let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
            let snap =
                Self::occurrence_snap(subqueries, field_ty, &occurrences, index, field_snap, deps)?;
            let start = add(occurrence.start, subquery.start);
            let end = add(occurrence.start, subquery.end);
            preds.push(vir::expr! {
                acc([pred](snap, [..[args.get_ty()]], [..[args.get_const()]], start, end))
            });
        }
        Ok(preds)
    }

    /// The snapshot `field_snap` of the field of type `field_ty`, extended by
    /// `subquery` at each occurrence of the subquery's parameter in the field
    /// type, if the subquery is non-empty. The field predicates holding the
    /// permissions are defined on `field_pred`, extended as in
    /// `occurrence_snap`.
    fn field_ext<'vir>(
        subqueries: &[Subquery<'vir>],
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        field_snap: vir::ExprSnap<'vir>,
        field_pred: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        let occurrences = deps.require_dep::<LifetimePositionEnc>(field_ty)?;
        if occurrences
            .iter()
            .all(|occurrence| occurrence.param != subquery.param)
        {
            return Ok(field_snap);
        }
        let mut snap = field_snap;
        for (index, occurrence) in occurrences.iter().enumerate() {
            if occurrence.param != subquery.param {
                continue;
            }
            let ext = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.ext;
            let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
            let key =
                Self::occurrence_snap(subqueries, field_ty, &occurrences, index, field_pred, deps)?;
            let start = add(occurrence.start, subquery.start);
            let end = add(occurrence.start, subquery.end);
            snap = ext.call()(snap, args.get_ty(), args.get_const(), key, start, end);
        }
        let guard = subquery.guard;
        Ok(vir::expr! { (guard) ? (snap) : (field_snap) })
    }

    /// The field predicates of the fields `fields`, given by their types and
    /// readers, of the snapshot `k` for `subquery`.
    fn fields_preds<'vir>(
        subqueries: &[Subquery<'vir>],
        subquery: Subquery<'vir>,
        fields: &[Field<'vir>],
        k: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprBool<'vir>>, EncodeFullError<'vir, Self>> {
        let mut preds = Vec::new();
        for &(field_ty, read) in fields {
            let field_snap = read.call()(k.downcast_ty());
            preds.extend(Self::field_preds(
                subqueries, subquery, field_ty, field_snap, deps,
            )?);
        }
        Ok(preds)
    }

    /// The fields `fields`, given by their types and readers, of `snapshot`,
    /// each extended by all `subqueries`. The field predicates holding the
    /// permissions are on the fields of `pred`.
    fn fields_ext<'vir>(
        subqueries: &[Subquery<'vir>],
        fields: &[Field<'vir>],
        snapshot: vir::ExprSnap<'vir>,
        pred: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprSnap<'vir>>, EncodeFullError<'vir, Self>> {
        let mut field_snaps = Vec::new();
        for &(field_ty, read) in fields {
            let mut field_snap = read.call()(snapshot.downcast_ty());
            let field_pred = read.call()(pred.downcast_ty());
            for &subquery in subqueries {
                field_snap =
                    Self::field_ext(subqueries, subquery, field_ty, field_snap, field_pred, deps)?;
            }
            field_snaps.push(field_snap);
        }
        Ok(field_snaps)
    }
}

impl TaskEncoder for IndirectPredEnc {
    task_encoder::encoder_cache!(IndirectPredEnc);
    const ENCODER_NAME: &'static str = "indirect predicate encoder";

    type TaskDescription<'vir> = RustTy<'vir>;

    type OutputRef<'vir> = IndirectPredRef<'vir>;
    type OutputFullLocal<'vir> = (vir::Predicate<'vir>, vir::Function<'vir>);

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        let pure = deps.require_dep::<TyPureEnc>(*task_key)?;
        let params = deps.require_dep::<GenericParamsEnc>(task_key.params)?;
        let ty = task_key.zip(pure);
        vir::with_vcx(|vcx| {
            let snapshot = pure.data.snapshot;
            let predicate = PredicateIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "p_{}_ind", task_key.name())),
                (
                    snapshot,
                    params.ty_args(),
                    params.const_args(),
                    vir::TYPE_INT,
                    vir::TYPE_INT,
                ),
            );
            let ext = FunctionIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "ext_{}", task_key.name())),
                (
                    snapshot,
                    params.ty_args(),
                    params.const_args(),
                    snapshot,
                    vir::TYPE_INT,
                    vir::TYPE_INT,
                ),
                snapshot,
            );
            deps.emit_output_ref(*task_key, IndirectPredRef { predicate, ext })?;

            let k_decl = vcx.mk_local_decl("k", snapshot);
            let lo_decl = vcx.mk_local_decl("lo", vir::TYPE_INT);
            let hi_decl = vcx.mk_local_decl("hi", vir::TYPE_INT);
            let k: vir::ExprSnap<'vir> = vcx.mk_local_ex(k_decl);
            let lo = vcx.mk_local_ex(lo_decl);
            let hi = vcx.mk_local_ex(hi_decl);
            // The arguments of the snapshot extension function.
            let snapshot_decl = vcx.mk_local_decl("snapshot", snapshot);
            let pred_decl = vcx.mk_local_decl("pred", snapshot);
            let ext_snapshot: vir::ExprSnap<'vir> = vcx.mk_local_ex(snapshot_decl);
            let ext_pred: vir::ExprSnap<'vir> = vcx.mk_local_ex(pred_decl);

            let (body, ext_body) = match &ty.specifics {
                TySpecifics::StructLike(data) => {
                    let subqueries = Self::subqueries(task_key, lo, hi, deps)?;
                    let fields = data
                        .fields
                        .iter()
                        .map(|(field, pure_field)| {
                            (field.decompose(task_key.params), pure_field.read)
                        })
                        .collect::<Vec<_>>();
                    let mut conjuncts = Vec::new();
                    for &subquery in &subqueries {
                        let preds = Self::fields_preds(&subqueries, subquery, &fields, k, deps)?;
                        if !preds.is_empty() {
                            let guard = subquery.guard;
                            let preds = vcx.mk_conj(&preds);
                            conjuncts.push(vir::expr! { vcx; (guard) ==> (preds) });
                        }
                    }
                    // Each field is extended at the occurrences of the
                    // parameters in its type.
                    let field_snaps =
                        Self::fields_ext(&subqueries, &fields, ext_snapshot, ext_pred, deps)?;
                    let ext_body = data.data.1.field_snaps_to_snap.call()(&field_snaps).upcast_ty();
                    (Some(vcx.mk_conj(&conjuncts)), Some(ext_body))
                }
                TySpecifics::EnumLike(data) => {
                    let subqueries = Self::subqueries(task_key, lo, hi, deps)?;
                    let discr = data.data.1.snap_to_discr_snap;
                    // The variants whose fields have lifetime positions, with
                    // their discriminants, fields and constructors.
                    let mut variants = Vec::new();
                    for variant in &data.variants {
                        let fields = variant
                            .inner
                            .fields
                            .iter()
                            .map(|(field, pure_field)| {
                                (field.decompose(task_key.params), pure_field.read)
                            })
                            .collect::<Vec<_>>();
                        let mut has_positions = false;
                        for &(field_ty, _) in &fields {
                            has_positions |= !deps
                                .require_dep::<LifetimePositionEnc>(field_ty)?
                                .is_empty();
                        }
                        if has_positions {
                            let cons = variant.inner.data.1.field_snaps_to_snap;
                            variants.push((variant.data.1.discr, fields, cons));
                        }
                    }
                    // The field predicates of each variant are guarded by its
                    // discriminant.
                    let mut conjuncts = Vec::new();
                    for &subquery in &subqueries {
                        let mut preds = Vec::new();
                        for (variant_discr, fields, _) in &variants {
                            let variant_preds =
                                Self::fields_preds(&subqueries, subquery, fields, k, deps)?;
                            if !variant_preds.is_empty() {
                                let in_variant =
                                    vcx.mk_eq_expr(discr.call()(k.downcast_ty()), *variant_discr);
                                let variant_preds = vcx.mk_conj(&variant_preds);
                                preds.push(vir::expr! { vcx; (in_variant) ==> (variant_preds) });
                            }
                        }
                        if !preds.is_empty() {
                            let guard = subquery.guard;
                            let preds = vcx.mk_conj(&preds);
                            conjuncts.push(vir::expr! { vcx; (guard) ==> (preds) });
                        }
                    }
                    // The fields of a variant are extended if both snapshots
                    // are of that variant: the field predicates are guarded by
                    // the discriminant of `pred`, and the variant of
                    // `snapshot` must not change. Otherwise, `snapshot` stays
                    // as it is.
                    let mut ext_body = ext_snapshot;
                    for (variant_discr, fields, cons) in variants.iter().rev() {
                        let field_snaps =
                            Self::fields_ext(&subqueries, fields, ext_snapshot, ext_pred, deps)?;
                        let extended: vir::ExprSnap<'vir> = cons.call()(&field_snaps).upcast_ty();
                        let snapshot_in_variant = vcx
                            .mk_eq_expr(discr.call()(ext_snapshot.downcast_ty()), *variant_discr);
                        let pred_in_variant =
                            vcx.mk_eq_expr(discr.call()(ext_pred.downcast_ty()), *variant_discr);
                        ext_body = vir::expr! { vcx;
                            ((snapshot_in_variant) && (pred_in_variant)) ? (extended) : (ext_body)
                        };
                    }
                    (Some(vcx.mk_conj(&conjuncts)), Some(ext_body))
                }
                TySpecifics::MutRef((data, pure_data)) => {
                    let referent_ty = data.referent.decompose(task_key.params);
                    let referent = deps.require_dep::<TyUseImpureEnc>(referent_ty)?;
                    let addr = pure_data.deref_access.call()(k.downcast_ty());
                    let referent_pred = referent.ref_to_pred(vcx, addr, None);
                    let subqueries = Self::subqueries(task_key, lo, hi, deps)?;
                    // If the query contains lifetime 0, we want to read the snapshot from the heap,
                    // otherwise, the snapshot of the referent is stored in our snapshot
                    let in_query = subqueries
                        .iter()
                        .find(|subquery| matches!(subquery.param, LifetimePosition::Lifetime(_)))
                        .unwrap()
                        .guard;
                    let current_snap = referent.ref_to_snap(addr);
                    let stored_snap: vir::ExprSnap<'vir> =
                        pure_data.value_access.call()(k.downcast_ty()).upcast_ty();
                    let referent_snap = vir::expr! { vcx;
                        (in_query) ? (current_snap) : (stored_snap)
                    };
                    let mut conjuncts = Vec::new();
                    for &subquery in &subqueries {
                        let mut preds = Vec::new();
                        // The own lifetime of the reference gives access to the referent.
                        if matches!(subquery.param, LifetimePosition::Lifetime(_)) {
                            preds.push(referent_pred);
                        }
                        // The positions of the referent's type are passed on to the referent like those of a field.
                        preds.extend(Self::field_preds(
                            &subqueries,
                            subquery,
                            referent_ty,
                            referent_snap,
                            deps,
                        )?);
                        if !preds.is_empty() {
                            let guard = subquery.guard;
                            let preds = vcx.mk_conj(&preds);
                            conjuncts.push(vir::expr! { vcx; (guard) ==> (preds) });
                        }
                    }
                    // The value is filled in with the referent's current
                    // snapshot if the query contains the own lifetime, and
                    // the stored value is kept otherwise. Either is then
                    // extended at the positions of the referent's type.
                    let pred_addr = pure_data.deref_access.call()(ext_pred.downcast_ty());
                    let current_snap = referent.ref_to_snap(pred_addr);
                    let stored_snap: vir::ExprSnap<'vir> =
                        pure_data.value_access.call()(ext_snapshot.downcast_ty()).upcast_ty();
                    let stored_pred: vir::ExprSnap<'vir> =
                        pure_data.value_access.call()(ext_pred.downcast_ty()).upcast_ty();
                    let mut extended_current = current_snap;
                    let mut extended_stored = stored_snap;
                    for &subquery in &subqueries {
                        extended_current = Self::field_ext(
                            &subqueries,
                            subquery,
                            referent_ty,
                            extended_current,
                            current_snap,
                            deps,
                        )?;
                        extended_stored = Self::field_ext(
                            &subqueries,
                            subquery,
                            referent_ty,
                            extended_stored,
                            stored_pred,
                            deps,
                        )?;
                    }
                    let value = vir::expr! { vcx;
                        (in_query) ? (extended_current) : (extended_stored)
                    };
                    let ext_body = pure_data.prim_to_snap.call()(
                        pure_data.deref_access.call()(ext_snapshot.downcast_ty()),
                        pure_data.metadata_access.call()(ext_snapshot.downcast_ty()),
                        value.downcast_ty(),
                    );
                    (Some(vcx.mk_conj(&conjuncts)), Some(ext_body.upcast_ty()))
                }
                // A shared reference has no lifetime positions.
                TySpecifics::ImmRef(_) => (Some(vcx.mk_bool::<true>()), Some(ext_snapshot)),
                // TODO: the remaining types.
                _ => (None, None),
            };

            let predicate_decl = vcx.mk_predicate(
                predicate,
                (
                    k_decl,
                    params.ty_decls(),
                    params.const_decls(),
                    lo_decl,
                    hi_decl,
                ),
                body,
            );

            let pre = vir::expr! { vcx;
                acc([predicate](
                    ext_pred,
                    [..[params.ty_exprs()]],
                    [..[params.const_exprs()]],
                    lo,
                    hi
                ))
            };
            let ext_body = ext_body.map(|ext_body| {
                vir::expr! { vcx;
                    unfolding ([predicate](
                        ext_pred,
                        [..[params.ty_exprs()]],
                        [..[params.const_exprs()]],
                        lo,
                        hi
                    )) in (ext_body)
                }
            });
            let ext_decl = vcx.mk_function(
                ext,
                (
                    snapshot_decl,
                    params.ty_decls(),
                    params.const_decls(),
                    pred_decl,
                    lo_decl,
                    hi_decl,
                ),
                vcx.alloc_slice(&[pre]),
                &[],
                None,
                ext_body,
            );

            Ok(((predicate_decl, ext_decl), ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for (predicate, ext) in Self::all_outputs_local_no_errors(program) {
            program.add_predicate(predicate);
            program.add_function(ext);
        }
    }
}

/// Tests that verify Viper code using the indirect predicates with Silicon.
#[cfg(test)]
mod verification_tests {
    use prusti_rustc_interface::span::DUMMY_SP;
    use task_encoder::TaskEncoder;

    use super::IndirectPredEnc;
    use crate::encoders::ty::{
        RustTyDecomposition, generics::IndirectCastersEnc, test_util::verify,
    };

    /// Verifies the Viper code `viper` together with the indirect predicate
    /// of the type alias `X` defined in `src`.
    fn verify_with_pred(src: &str, viper: &str) -> Result<(), String> {
        verify(
            src,
            |ty| {
                IndirectPredEnc::encode(ty.ty, true, DUMMY_SP).unwrap();
            },
            viper,
        )
    }

    /// Verifies the Viper code `viper` together with the indirect predicate
    /// of the type alias `X` defined in `src` and the casts of the
    /// predicates of its type arguments.
    fn verify_with_casts(src: &str, viper: &str) -> Result<(), String> {
        verify(
            src,
            |ty| {
                IndirectPredEnc::encode(ty.ty, true, DUMMY_SP).unwrap();
                for arg in ty.args.args() {
                    let arg = RustTyDecomposition::from_ty(arg.expect_ty(), ty.args.context());
                    IndirectCastersEnc::encode(
                        (RustTyDecomposition::param(), arg.ty),
                        true,
                        DUMMY_SP,
                    )
                    .unwrap();
                }
            },
            viper,
        )
    }

    fn assert_verifies(src: &str, viper: &str) {
        if let Err(output) = verify_with_pred(src, viper) {
            panic!("verification failed:\n{output}");
        }
    }

    /// Asserts that the verification fails with an error containing `error`.
    fn assert_fails(src: &str, viper: &str, error: &str) {
        match verify_with_pred(src, viper) {
            Ok(()) => panic!("verification succeeded"),
            Err(output) => assert!(
                output.contains(error),
                "verification failed differently:\n{output}"
            ),
        }
    }

    /// `&'a mut U`: position 0 is `'a`, the referent's positions are
    /// `[1, 1 + arity(U))`.
    #[test]
    fn mutable_reference() {
        let src = "type X<'a, T> = &'a mut T;";
        assert_verifies(
            src,
            r#"
            // 'a holds the referent.
            method lifetime(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, 0, 1)
              ensures p_Param(k.s_Ref_mutable_0, U)
            {
              unfold p_Ref_mutable_ind(k, U, 0, 1)
            }

            // The referent's positions hold those of the referent, on the
            // snapshot stored in `k`.
            method referent_positions(k: s_Ref_mutable, U: Type)
              requires 0 < arity(U)
              requires p_Ref_mutable_ind(k, U, 1, 1 + arity(U))
              ensures p_Param_ind(k.s_Ref_mutable_2, U, 0, arity(U))
            {
              unfold p_Ref_mutable_ind(k, U, 1, 1 + arity(U))
            }

            // With 'a, they are on the referent's current snapshot.
            method all_positions(k: s_Ref_mutable, U: Type)
              requires 0 < arity(U)
              requires p_Ref_mutable_ind(k, U, 0, 1 + arity(U))
              ensures p_Param(k.s_Ref_mutable_0, U)
              ensures p_Param_ind(p_Param_snap(k.s_Ref_mutable_0, U), U, 0, arity(U))
            {
              unfold p_Ref_mutable_ind(k, U, 0, 1 + arity(U))
            }

            // Extending by 'a reads the referent's snapshot and keeps the address.
            method extend(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, 0, 1)
              ensures p_Ref_mutable_ind(k, U, 0, 1)
              ensures ext_Ref_mutable(k, U, k, 0, 1).s_Ref_mutable_0 == k.s_Ref_mutable_0
              ensures ext_Ref_mutable(k, U, k, 0, 1).s_Ref_mutable_2 ==
                (unfolding p_Ref_mutable_ind(k, U, 0, 1) in p_Param_snap(k.s_Ref_mutable_0, U))
            {
            }

            // Split off 'a, the referent's positions are on the snapshot
            // extended by 'a.
            method split(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, 0, 1 + arity(U))
              ensures p_Ref_mutable_ind(k, U, 0, 1)
              ensures p_Ref_mutable_ind(ext_Ref_mutable(k, U, k, 0, 1), U, 1, 1 + arity(U))
            {
              unfold p_Ref_mutable_ind(k, U, 0, 1 + arity(U))
              fold p_Ref_mutable_ind(k, U, 0, 1)
              fold p_Ref_mutable_ind(ext_Ref_mutable(k, U, k, 0, 1), U, 1, 1 + arity(U))
            }
            "#,
        );
        // The referent's positions do not hold the referent.
        assert_fails(
            src,
            r#"
            method test(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, 1, 1 + arity(U))
              ensures p_Param(k.s_Ref_mutable_0, U)
            {
              unfold p_Ref_mutable_ind(k, U, 1, 1 + arity(U))
            }
            "#,
            "insufficient permission to access p_Param",
        );
        // 'a does not hold the referent's positions.
        assert_fails(
            src,
            r#"
            method test(k: s_Ref_mutable, U: Type)
              requires 0 < arity(U)
              requires p_Ref_mutable_ind(k, U, 0, 1)
              ensures p_Param(k.s_Ref_mutable_0, U)
              ensures p_Param_ind(p_Param_snap(k.s_Ref_mutable_0, U), U, 0, arity(U))
            {
              unfold p_Ref_mutable_ind(k, U, 0, 1)
            }
            "#,
            "insufficient permission to access p_Param_ind",
        );
    }

    /// `&'a U` has no positions: its predicate holds nothing.
    #[test]
    fn shared_reference() {
        let src = "type X<'a, T> = &'a T;";
        assert_verifies(
            src,
            r#"
            method test(k: s_Ref_immutable, U: Type)
              ensures p_Ref_immutable_ind(k, U, 0, 1)
              ensures ext_Ref_immutable(k, U, k, 0, 1) == k
            {
              fold p_Ref_immutable_ind(k, U, 0, 1)
            }
            "#,
        );
        assert_fails(
            src,
            r#"
            method test(k: s_Ref_immutable, U: Type)
              requires p_Ref_immutable_ind(k, U, 0, 1)
            {
              unfold p_Ref_immutable_ind(k, U, 0, 1)
              assert false
            }
            "#,
            "Assert might fail",
        );
    }

    /// `'static` is not a position of a struct: no position holds the
    /// referent of a `&'static mut`.
    #[test]
    fn static_mutable_reference() {
        let src = "struct S<'a> { s: &'static mut i32, t: &'a mut i32 }
                   type X<'a> = S<'a>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_S)
              requires p_S_ind(k, 0, 1)
              ensures p_Ref_mutable_ind(k.s_S_1, s_Int_i32_type(), 0, 1)
            {
              unfold p_S_ind(k, 0, 1)
            }
            "#,
        );
        assert_fails(
            src,
            r#"
            method test(k: s_S)
              requires p_S_ind(k, 0, 1)
              ensures p_Ref_mutable_ind(k.s_S_0, s_Int_i32_type(), 0, 1)
            {
              unfold p_S_ind(k, 0, 1)
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
    }

    /// A struct passes its positions on to its fields: `'a @ 0` and
    /// `T @ [1, 1 + arity(T))`.
    #[test]
    fn struct_fields() {
        let src = "struct S<'a, T> { x: &'a mut i32, y: T }
                   type X<'a, T> = S<'a, T>;";
        assert_verifies(
            src,
            r#"
            method lifetime(k: s_S, T: Type)
              requires p_S_ind(k, T, 0, 1)
              ensures p_Ref_mutable_ind(k.s_S_0, s_Int_i32_type(), 0, 1)
            {
              unfold p_S_ind(k, T, 0, 1)
            }

            method type_param(k: s_S, T: Type)
              requires 0 < arity(T)
              requires p_S_ind(k, T, 1, 1 + arity(T))
              ensures p_Param_ind(k.s_S_1, T, 0, arity(T))
            {
              unfold p_S_ind(k, T, 1, 1 + arity(T))
            }

            method split(k: s_S, T: Type)
              requires p_S_ind(k, T, 0, 1 + arity(T))
              ensures p_S_ind(k, T, 0, 1) && p_S_ind(k, T, 1, 1 + arity(T))
            {
              unfold p_S_ind(k, T, 0, 1 + arity(T))
              fold p_S_ind(k, T, 0, 1)
              fold p_S_ind(k, T, 1, 1 + arity(T))
            }

            method merge(k: s_S, T: Type)
              requires p_S_ind(k, T, 0, 1) && p_S_ind(k, T, 1, 1 + arity(T))
              ensures p_S_ind(k, T, 0, 1 + arity(T))
            {
              unfold p_S_ind(k, T, 0, 1)
              unfold p_S_ind(k, T, 1, 1 + arity(T))
              fold p_S_ind(k, T, 0, 1 + arity(T))
            }
            "#,
        );
        // The positions of T do not hold `x`.
        assert_fails(
            src,
            r#"
            method test(k: s_S, T: Type)
              requires p_S_ind(k, T, 1, 1 + arity(T))
              ensures p_Ref_mutable_ind(k.s_S_0, s_Int_i32_type(), 0, 1)
            {
              unfold p_S_ind(k, T, 1, 1 + arity(T))
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
        // 'a does not hold `y`.
        assert_fails(
            src,
            r#"
            method test(k: s_S, T: Type)
              requires 0 < arity(T)
              requires p_S_ind(k, T, 0, 1)
              ensures p_Param_ind(k.s_S_1, T, 0, arity(T))
            {
              unfold p_S_ind(k, T, 0, 1)
            }
            "#,
            "insufficient permission to access p_Param_ind",
        );
    }

    /// The positions of a tuple are those of its elements in turn.
    #[test]
    fn tuple() {
        let src = "type X<T, U> = (T, U);";
        assert_verifies(
            src,
            r#"
            method elements(k: s_2_Tuple, T: Type, U: Type)
              requires 0 < arity(T) && 0 < arity(U)
              requires p_2_Tuple_ind(k, T, U, 0, arity(T) + arity(U))
              ensures p_Param_ind(k.s_2_Tuple_0, T, 0, arity(T))
              ensures p_Param_ind(k.s_2_Tuple_1, U, 0, arity(U))
            {
              unfold p_2_Tuple_ind(k, T, U, 0, arity(T) + arity(U))
            }

            method split(k: s_2_Tuple, T: Type, U: Type)
              requires p_2_Tuple_ind(k, T, U, 0, arity(T) + arity(U))
              ensures p_2_Tuple_ind(k, T, U, 0, arity(T))
              ensures p_2_Tuple_ind(k, T, U, arity(T), arity(T) + arity(U))
            {
              unfold p_2_Tuple_ind(k, T, U, 0, arity(T) + arity(U))
              fold p_2_Tuple_ind(k, T, U, 0, arity(T))
              fold p_2_Tuple_ind(k, T, U, arity(T), arity(T) + arity(U))
            }
            "#,
        );
        // The positions of T do not hold those of U.
        assert_fails(
            src,
            r#"
            method test(k: s_2_Tuple, T: Type, U: Type)
              requires 0 < arity(U)
              requires p_2_Tuple_ind(k, T, U, 0, arity(T))
              ensures p_Param_ind(k.s_2_Tuple_1, U, 0, arity(U))
            {
              unfold p_2_Tuple_ind(k, T, U, 0, arity(T))
            }
            "#,
            "insufficient permission to access p_Param_ind",
        );
    }

    /// In the field type `Three<'a, 'b, 'a>`, `'a` occurs at positions 0
    /// and 2, with `'b` in between.
    #[test]
    fn duplicate_lifetimes() {
        let src = "struct Three<'x, 'y, 'z>(&'x mut i32, &'y mut i32, &'z mut i32);
                   struct S<'a, 'b> { f: Three<'a, 'b, 'a> }
                   type X<'a, 'b> = S<'a, 'b>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_S)
              requires p_S_ind(k, 0, 1)
              ensures p_Three_ind(k.s_S_0, 0, 1) && p_Three_ind(k.s_S_0, 2, 3)
            {
              unfold p_S_ind(k, 0, 1)
            }
            "#,
        );
        // 'a does not hold the position of 'b.
        assert_fails(
            src,
            r#"
            method test(k: s_S)
              requires p_S_ind(k, 0, 1)
              ensures p_Three_ind(k.s_S_0, 1, 2)
            {
              unfold p_S_ind(k, 0, 1)
            }
            "#,
            "insufficient permission to access p_Three_ind",
        );
    }

    /// In the field type `(T, U, T)`, `T` occurs twice, with `U` in between.
    #[test]
    fn duplicate_type_params() {
        let src = "struct S<T, U> { f: (T, U, T) }
                   type X<T, U> = S<T, U>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_S, T: Type, U: Type)
              requires 0 < arity(T)
              requires p_S_ind(k, T, U, 0, arity(T))
              ensures p_Param_ind(k.s_S_0.s_3_Tuple_0, T, 0, arity(T))
              ensures p_Param_ind(k.s_S_0.s_3_Tuple_2, T, 0, arity(T))
            {
              unfold p_S_ind(k, T, U, 0, arity(T))
              unfold p_3_Tuple_ind(k.s_S_0, T, U, T, 0, arity(T))
              unfold p_3_Tuple_ind(k.s_S_0, T, U, T, arity(T) + arity(U), 2 * arity(T) + arity(U))
            }
            "#,
        );
        // The positions of T do not hold those of U.
        assert_fails(
            src,
            r#"
            method test(k: s_S, T: Type, U: Type)
              requires 0 < arity(T) && 0 < arity(U)
              requires p_S_ind(k, T, U, 0, arity(T))
              ensures p_Param_ind(k.s_S_0.s_3_Tuple_1, U, 0, arity(U))
            {
              unfold p_S_ind(k, T, U, 0, arity(T))
              unfold p_3_Tuple_ind(k.s_S_0, T, U, T, 0, arity(T))
              unfold p_3_Tuple_ind(k.s_S_0, T, U, T, arity(T) + arity(U), 2 * arity(T) + arity(U))
            }
            "#,
            "insufficient permission to access p_Param_ind",
        );
    }

    /// In `&'a mut &'b mut i32`, the position of `'b` lies behind `'a`. Split
    /// off `'a`, it is on the snapshot extended by `'a`.
    #[test]
    fn nested_mutable_references() {
        let src = "struct S<'a, 'b> { r: &'a mut &'b mut i32 }
                   type X<'a, 'b> = S<'a, 'b>;";
        assert_verifies(
            src,
            r#"
            method split(k: s_S)
              requires p_S_ind(k, 0, 2)
              ensures p_S_ind(k, 0, 1) && p_S_ind(ext_S(k, k, 0, 1), 1, 2)
            {
              unfold p_S_ind(k, 0, 2)
              fold p_S_ind(k, 0, 1)
              fold p_S_ind(ext_S(k, k, 0, 1), 1, 2)
            }
            "#,
        );
        // Not on the snapshot that is not extended.
        assert_fails(
            src,
            r#"
            method split(k: s_S)
              requires p_S_ind(k, 0, 2)
              ensures p_S_ind(k, 0, 1) && p_S_ind(k, 1, 2)
            {
              unfold p_S_ind(k, 0, 2)
              fold p_S_ind(k, 0, 1)
              fold p_S_ind(k, 1, 2)
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
    }

    /// In `&'a mut &'a mut i32`, both positions belong to `'a`, so the
    /// predicate itself puts the inner one on the field snapshot extended by
    /// the outer one.
    #[test]
    fn nested_mutable_references_same_lifetime() {
        let src = "struct E<'a>(&'a mut &'a mut i32);
                   type X<'a> = E<'a>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_E)
              requires p_E_ind(k, 0, 1)
              ensures p_Ref_mutable_ind(k.s_E_0, s_Ref_mutable_type(s_Int_i32_type()), 0, 1)
              ensures p_Ref_mutable_ind(ext_Ref_mutable(k.s_E_0, s_Ref_mutable_type(s_Int_i32_type()), k.s_E_0, 0, 1), s_Ref_mutable_type(s_Int_i32_type()), 1, 2)
            {
              unfold p_E_ind(k, 0, 1)
            }
            "#,
        );
        // Not on the field snapshot that is not extended.
        assert_fails(
            src,
            r#"
            method test(k: s_E)
              requires p_E_ind(k, 0, 1)
              ensures p_Ref_mutable_ind(k.s_E_0, s_Ref_mutable_type(s_Int_i32_type()), 1, 2)
            {
              unfold p_E_ind(k, 0, 1)
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
    }

    /// An enum passes its positions on to the fields of the variant it is
    /// in: `'a @ 0` to variant `A`, `T @ [1, 1 + arity(T))` to variant `B`.
    #[test]
    fn enum_variants() {
        let src = "enum E<'a, T> { A(&'a mut i32), B(T), C }
                   type X<'a, T> = E<'a, T>;";
        assert_verifies(
            src,
            r#"
            method variant_a(k: s_E, T: Type)
              requires p_E_ind(k, T, 0, 1) && k.iss_E_0_cons
              ensures p_Ref_mutable_ind(k.s_E_0_0, s_Int_i32_type(), 0, 1)
            {
              unfold p_E_ind(k, T, 0, 1)
            }

            method variant_b(k: s_E, T: Type)
              requires 0 < arity(T)
              requires p_E_ind(k, T, 1, 1 + arity(T)) && k.iss_E_1_cons
              ensures p_Param_ind(k.s_E_1_0, T, 0, arity(T))
            {
              unfold p_E_ind(k, T, 1, 1 + arity(T))
            }

            // Variant `C` holds nothing.
            method variant_c(k: s_E, T: Type)
              requires k.iss_E_2_cons
              ensures p_E_ind(k, T, 0, 1 + arity(T))
            {
              fold p_E_ind(k, T, 0, 1 + arity(T))
            }

            // Extension keeps the variant.
            method extend(k: s_E, T: Type)
              requires p_E_ind(k, T, 0, 1)
              ensures p_E_ind(k, T, 0, 1)
              ensures s_E_discr(ext_E(k, T, k, 0, 1)) == s_E_discr(k)
            {
            }

            method split(k: s_E, T: Type)
              requires p_E_ind(k, T, 0, 1 + arity(T))
              ensures p_E_ind(k, T, 0, 1) && p_E_ind(k, T, 1, 1 + arity(T))
            {
              unfold p_E_ind(k, T, 0, 1 + arity(T))
              fold p_E_ind(k, T, 0, 1)
              fold p_E_ind(k, T, 1, 1 + arity(T))
            }
            "#,
        );
        // Without knowing the variant, the field is not held.
        assert_fails(
            src,
            r#"
            method test(k: s_E, T: Type)
              requires p_E_ind(k, T, 0, 1)
              ensures p_Ref_mutable_ind(k.s_E_0_0, s_Int_i32_type(), 0, 1)
            {
              unfold p_E_ind(k, T, 0, 1)
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
        // In variant `B`, 'a holds nothing.
        assert_fails(
            src,
            r#"
            method test(k: s_E, T: Type)
              requires p_E_ind(k, T, 0, 1) && k.iss_E_1_cons
              ensures p_Ref_mutable_ind(k.s_E_0_0, s_Int_i32_type(), 0, 1)
            {
              unfold p_E_ind(k, T, 0, 1)
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
    }

    /// The casts exchange the predicate of `&'a mut U` and that of a type
    /// parameter instantiated with it, which lets `Option<&'a mut U>`'s
    /// predicate be unfolded down to the referent.
    #[test]
    fn casts() {
        let src = "type X<'a, T> = Option<&'a mut T>;";
        let result = verify_with_casts(
            src,
            r#"
            // The snapshot of the field of `Some` is that of a reference
            // once its type is known.
            method unfold_option(k: s_Option, U: Type)
              requires p_Option_ind(k, s_Ref_mutable_type(U), 0, 1) && k.iss_Option_1_cons
              requires s_Param_typ(k.s_Option_1_0) == s_Ref_mutable_type(U)
              ensures p_Param(k.s_Option_1_0.make_concrete_s_Ref_mutable.s_Ref_mutable_0, U)
            {
              unfold p_Option_ind(k, s_Ref_mutable_type(U), 0, 1)
              make_concrete_ind_Ref_mutable(k.s_Option_1_0.make_concrete_s_Ref_mutable, U, 0, 1)
              unfold p_Ref_mutable_ind(k.s_Option_1_0.make_concrete_s_Ref_mutable, U, 0, 1)
            }

            // Casting back and forth keeps the extension.
            method round_trip(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, 0, 1)
              ensures p_Ref_mutable_ind(k, U, 0, 1)
              ensures ext_Ref_mutable(k, U, k, 0, 1) == old(ext_Ref_mutable(k, U, k, 0, 1))
            {
              make_generic_ind_Ref_mutable(k, U, 0, 1)
              make_concrete_ind_Ref_mutable(k, U, 0, 1)
            }

            // The extension of the option is that of the reference.
            method extend_option(k: s_Option, U: Type)
              requires p_Option_ind(k, s_Ref_mutable_type(U), 0, 1) && k.iss_Option_1_cons
              requires s_Param_typ(k.s_Option_1_0) == s_Ref_mutable_type(U)
              ensures p_Ref_mutable_ind(k.s_Option_1_0.make_concrete_s_Ref_mutable, U, 0, 1)
              ensures old(ext_Option(k, s_Ref_mutable_type(U), k, 0, 1)).s_Option_1_0 ==
                make_generic_s_Ref_mutable(ext_Ref_mutable(
                  k.s_Option_1_0.make_concrete_s_Ref_mutable, U,
                  k.s_Option_1_0.make_concrete_s_Ref_mutable, 0, 1), U)
            {
              unfold p_Option_ind(k, s_Ref_mutable_type(U), 0, 1)
              make_concrete_ind_Ref_mutable(k.s_Option_1_0.make_concrete_s_Ref_mutable, U, 0, 1)
            }
            "#,
        );
        if let Err(output) = result {
            panic!("verification failed:\n{output}");
        }
        // The casts do not make the state inconsistent.
        let result = verify_with_casts(
            src,
            r#"
            method test(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, 0, 1)
            {
              make_generic_ind_Ref_mutable(k, U, 0, 1)
              make_concrete_ind_Ref_mutable(k, U, 0, 1)
              assert false
            }
            "#,
        );
        match result {
            Ok(()) => panic!("verification succeeded"),
            Err(output) => assert!(
                output.contains("Assert might fail"),
                "verification failed differently:\n{output}"
            ),
        }
    }
}
