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
        lifetime_position::{LifetimePosition, LifetimePositionEnc},
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
                .map(|(param, x)| {
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
                })
                .collect()
        })
    }

    /// The first positions of the occurrences of the subquery's parameter in
    /// the field type `field_ty`.
    fn occurrences<'vir>(
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprInt<'vir>>, EncodeFullError<'vir, Self>> {
        Ok(deps
            .require_dep::<LifetimePositionEnc>(field_ty)?
            .into_iter()
            .filter(|&(param, _)| param == subquery.param)
            .map(|(_, y)| y)
            .collect())
    }

    /// The predicates of the field of type `field_ty` with the snapshot `field_snap` for `subquery`:
    /// one for each occurrence of the subquery's parameter in the field type
    fn field_preds<'vir>(
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        field_snap: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprBool<'vir>>, EncodeFullError<'vir, Self>> {
        let occurrences = Self::occurrences(subquery, field_ty, deps)?;
        if occurrences.is_empty() {
            return Ok(Vec::new());
        }
        let pred = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.predicate;
        let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
        Ok(occurrences
            .into_iter()
            .map(|y| {
                let start = add(y, subquery.start);
                let end = add(y, subquery.end);
                vir::expr! {
                    acc([pred](
                        field_snap,
                        [..[args.get_ty()]],
                        [..[args.get_const()]],
                        start,
                        end
                    ))
                }
            })
            .collect())
    }

    /// The snapshot `field_snap` of the field of type `field_ty`, extended by
    /// `subquery` at each occurrence of the subquery's parameter in the field
    /// type. The field predicates holding the permissions are defined on
    /// `field_pred`.
    fn field_ext<'vir>(
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        field_snap: vir::ExprSnap<'vir>,
        field_pred: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        let occurrences = Self::occurrences(subquery, field_ty, deps)?;
        if occurrences.is_empty() {
            return Ok(field_snap);
        }
        let ext = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.ext;
        let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
        Ok(vir::with_vcx(|vcx| {
            occurrences.into_iter().fold(field_snap, |snap, y| {
                let guard = subquery.guard;
                let start = add(y, subquery.start);
                let end = add(y, subquery.end);
                let extended = ext.call()(
                    snap,
                    args.get_ty(),
                    args.get_const(),
                    field_pred,
                    start,
                    end,
                );
                vir::expr! { vcx; (guard) ? (extended) : (snap) }
            })
        }))
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
                    let mut conjuncts = Vec::new();
                    for &subquery in &subqueries {
                        let mut preds = Vec::new();
                        for (field, pure_field) in &data.fields {
                            let field_ty = field.decompose(task_key.params);
                            let field_snap = pure_field.read.call()(k.downcast_ty());
                            preds.extend(Self::field_preds(subquery, field_ty, field_snap, deps)?);
                        }
                        if !preds.is_empty() {
                            let guard = subquery.guard;
                            let preds = vcx.mk_conj(&preds);
                            conjuncts.push(vir::expr! { vcx; (guard) ==> (preds) });
                        }
                    }
                    // Each field is extended at the occurrences of the
                    // parameters in its type.
                    let mut field_snaps = Vec::new();
                    for (field, pure_field) in &data.fields {
                        let field_ty = field.decompose(task_key.params);
                        let mut field_snap = pure_field.read.call()(ext_snapshot.downcast_ty());
                        let field_pred = pure_field.read.call()(ext_pred.downcast_ty());
                        for &subquery in &subqueries {
                            field_snap =
                                Self::field_ext(subquery, field_ty, field_snap, field_pred, deps)?;
                        }
                        field_snaps.push(field_snap);
                    }
                    let ext_body = data.data.1.field_snaps_to_snap.call()(&field_snaps).upcast_ty();
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
                            subquery,
                            referent_ty,
                            extended_current,
                            current_snap,
                            deps,
                        )?;
                        extended_stored = Self::field_ext(
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
                // TODO: enums and the remaining types.
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

#[cfg(test)]
mod tests {
    use prusti_rustc_interface::span::DUMMY_SP;
    use task_encoder::TaskEncoder;
    use vir::CastType;

    use super::IndirectPredEnc;
    use crate::encoders::ty::test_util::{dedent, simplified, with_type_alias};

    fn indirect_pred(src: &str) -> String {
        fn conjuncts<'vir>(e: vir::ExprDyn<'vir>, out: &mut Vec<vir::ExprDyn<'vir>>) {
            match e.kind {
                vir::ExprKindGenData::BinOp(b) if b.kind == vir::BinOpKind::And => {
                    conjuncts(b.lhs, out);
                    conjuncts(b.rhs, out);
                }
                _ => out.push(e),
            }
        }

        fn one_line(e: vir::ExprDyn<'_>) -> String {
            format!("{e:?}").replace('\n', " ")
        }

        fn show(e: vir::ExprDyn<'_>) -> String {
            match e.kind {
                vir::ExprKindGenData::BinOp(b) if b.kind == vir::BinOpKind::Implies => {
                    let mut instances = Vec::new();
                    conjuncts(b.rhs, &mut instances);
                    let instances = instances
                        .iter()
                        .map(|instance| format!("    ({})", one_line(instance)))
                        .collect::<Vec<_>>();
                    format!("  (({}) ==>\n{})", one_line(b.lhs), instances.join(" &&\n"))
                }
                _ => format!("  ({})", one_line(e)),
            }
        }

        fn show_predicate(predicate: vir::Predicate<'_>) -> String {
            let debug = format!("{predicate:?}");
            let signature = debug.lines().next().unwrap().trim_end_matches('{');
            let Some(body) = predicate.expr else {
                return signature.to_string();
            };
            let mut body_conjuncts = Vec::new();
            conjuncts(body.as_dyn(), &mut body_conjuncts);
            let body_conjuncts = body_conjuncts.into_iter().map(show).collect::<Vec<_>>();
            format!("{signature} {{\n{}\n}}", body_conjuncts.join(" &&\n"))
        }

        fn show_function(function: vir::Function<'_>) -> String {
            let args = function
                .args
                .iter()
                .map(|arg| format!("{arg:?}"))
                .collect::<Vec<_>>();
            let mut lines = vec![format!(
                "function {}({}): {:?}",
                function.name,
                args.join(", "),
                function.ret
            )];
            lines.extend(
                function
                    .pres
                    .iter()
                    .map(|pre| format!("  requires {}", one_line(pre.as_dyn()))),
            );
            if let Some(body) = function.expr {
                lines.push("{".to_string());
                lines.extend(format!("{body:?}").lines().map(|line| format!("  {line}")));
                lines.push("}".to_string());
            }
            lines.join("\n")
        }

        with_type_alias(src, |ty| {
            let (_, (predicate, ext), _) = IndirectPredEnc::encode(ty.ty, true, DUMMY_SP)
                .unwrap()
                .unwrap();
            format!(
                "{}\n\n{}",
                show_predicate(simplified(predicate)),
                show_function(simplified(ext))
            )
        })
    }

    #[test]
    fn lifetimes() {
        assert_eq!(
            indirect_pred(
                "struct Pair<'a, 'b>(&'a mut i32, &'b mut i32);
                 type X<'a, 'b> = Pair<'a, 'b>;"
            ),
            dedent(
                r#"
                predicate p_Pair_ind(k: s_Pair, lo: Int, hi: Int) {
                  ((((lo) <= (0)) && ((0) < (hi))) ==>
                    (p_Ref_mutable_ind(k.s_Pair_0, s_Int_i32_type(), 0, 1))) &&
                  ((((lo) <= (1)) && ((1) < (hi))) ==>
                    (p_Ref_mutable_ind(k.s_Pair_1, s_Int_i32_type(), 0, 1)))
                }

                function ext_Pair(snapshot: s_Pair, pred: s_Pair, lo: Int, hi: Int): s_Pair
                  requires p_Pair_ind(pred, lo, hi)
                {
                  unfolding p_Pair_ind(pred, lo, hi) in (s_Pair_cons(((lo) <= (0)) && ((0) < (hi))
                  ? ext_Ref_mutable(snapshot.s_Pair_0, s_Int_i32_type(), pred.s_Pair_0, 0, 1)
                  : snapshot.s_Pair_0, ((lo) <= (1)) && ((1) < (hi))
                  ? ext_Ref_mutable(snapshot.s_Pair_1, s_Int_i32_type(), pred.s_Pair_1, 0, 1)
                  : snapshot.s_Pair_1))
                }
                "#
            ),
        );
    }

    #[test]
    fn nested() {
        assert_eq!(
            indirect_pred(
                "struct Pair<'a, 'b>(&'a mut i32, &'b mut i32);
                 struct S<'c, 'd> {
                     p: Pair<'d, 'c>,
                     q: &'c mut i32,
                     r: &'d i32,
                     s: &'static mut i32,
                 }
                 type X<'c, 'd> = S<'c, 'd>;"
            ),
            dedent(
                r#"
                predicate p_S_ind(k: s_S, lo: Int, hi: Int) {
                  ((((lo) <= (0)) && ((0) < (hi))) ==>
                    (p_Pair_ind(k.s_S_0, 1, 2)) &&
                    (p_Ref_mutable_ind(k.s_S_1, s_Int_i32_type(), 0, 1))) &&
                  ((((lo) <= (1)) && ((1) < (hi))) ==>
                    (p_Pair_ind(k.s_S_0, 0, 1)))
                }

                function ext_S(snapshot: s_S, pred: s_S, lo: Int, hi: Int): s_S
                  requires p_S_ind(pred, lo, hi)
                {
                  unfolding p_S_ind(pred, lo, hi) in (s_S_cons(((lo) <= (1)) && ((1) < (hi))
                  ? ext_Pair(((lo) <= (0)) && ((0) < (hi))
                    ? ext_Pair(snapshot.s_S_0, pred.s_S_0, 1, 2)
                    : snapshot.s_S_0, pred.s_S_0, 0, 1)
                  : ((lo) <= (0)) && ((0) < (hi))
                    ? ext_Pair(snapshot.s_S_0, pred.s_S_0, 1, 2)
                    : snapshot.s_S_0, ((lo) <= (0)) && ((0) < (hi))
                  ? ext_Ref_mutable(snapshot.s_S_1, s_Int_i32_type(), pred.s_S_1, 0, 1)
                  : snapshot.s_S_1, snapshot.s_S_2, snapshot.s_S_3))
                }
                "#
            ),
        );
    }

    #[test]
    fn after_type_param() {
        assert_eq!(
            indirect_pred(
                "struct S<'a, T> { f: (T, &'a mut i32) }
                 type X<'a, T> = S<'a, T>;"
            ),
            dedent(
                r#"
                predicate p_S_ind(k: s_S, T$1: Type, lo: Int, hi: Int) {
                  ((((lo) <= (0)) && ((0) < (hi))) ==>
                    (p_2_Tuple_ind(k.s_S_0, T$1, s_Ref_mutable_type(s_Int_i32_type()), arity(T$1), (arity(T$1)) + (1)))) &&
                  (((clamp((lo) - (1), 0, arity(T$1))) < (clamp((hi) - (1), 0, arity(T$1)))) ==>
                    (p_2_Tuple_ind(k.s_S_0, T$1, s_Ref_mutable_type(s_Int_i32_type()), clamp((lo) - (1), 0, arity(T$1)), clamp((hi) - (1), 0, arity(T$1)))))
                }

                function ext_S(snapshot: s_S, T$1: Type, pred: s_S, lo: Int, hi: Int): s_S
                  requires p_S_ind(pred, T$1, lo, hi)
                {
                  unfolding p_S_ind(pred, T$1, lo, hi) in (s_S_cons((clamp((lo) - (1), 0, arity(T$1))) < (clamp((hi) - (1), 0, arity(T$1)))
                  ? ext_2_Tuple(((lo) <= (0)) && ((0) < (hi))
                    ? ext_2_Tuple(snapshot.s_S_0, T$1, s_Ref_mutable_type(s_Int_i32_type()), pred.s_S_0, arity(T$1), (arity(T$1)) + (1))
                    : snapshot.s_S_0, T$1, s_Ref_mutable_type(s_Int_i32_type()), pred.s_S_0, clamp((lo) - (1), 0, arity(T$1)), clamp((hi) - (1), 0, arity(T$1)))
                  : ((lo) <= (0)) && ((0) < (hi))
                    ? ext_2_Tuple(snapshot.s_S_0, T$1, s_Ref_mutable_type(s_Int_i32_type()), pred.s_S_0, arity(T$1), (arity(T$1)) + (1))
                    : snapshot.s_S_0))
                }
                "#
            ),
        );
    }

    #[test]
    fn tuple() {
        assert_eq!(
            indirect_pred("type X<T, U> = (T, U);"),
            dedent(
                r#"
                predicate p_2_Tuple_ind(k: s_2_Tuple, T$0: Type, U$1: Type, lo: Int, hi: Int) {
                  (((clamp(lo, 0, arity(T$0))) < (clamp(hi, 0, arity(T$0)))) ==>
                    (p_Param_ind(k.s_2_Tuple_0, T$0, clamp(lo, 0, arity(T$0)), clamp(hi, 0, arity(T$0))))) &&
                  (((clamp((lo) - (arity(T$0)), 0, arity(U$1))) < (clamp((hi) - (arity(T$0)), 0, arity(U$1)))) ==>
                    (p_Param_ind(k.s_2_Tuple_1, U$1, clamp((lo) - (arity(T$0)), 0, arity(U$1)), clamp((hi) - (arity(T$0)), 0, arity(U$1)))))
                }

                function ext_2_Tuple(snapshot: s_2_Tuple, T$0: Type, U$1: Type, pred: s_2_Tuple, lo: Int, hi: Int): s_2_Tuple
                  requires p_2_Tuple_ind(pred, T$0, U$1, lo, hi)
                {
                  unfolding p_2_Tuple_ind(pred, T$0, U$1, lo, hi) in (s_2_Tuple_cons((clamp(lo, 0, arity(T$0))) < (clamp(hi, 0, arity(T$0)))
                  ? ext_Param(snapshot.s_2_Tuple_0, T$0, pred.s_2_Tuple_0, clamp(lo, 0, arity(T$0)), clamp(hi, 0, arity(T$0)))
                  : snapshot.s_2_Tuple_0, (clamp((lo) - (arity(T$0)), 0, arity(U$1))) < (clamp((hi) - (arity(T$0)), 0, arity(U$1)))
                  ? ext_Param(snapshot.s_2_Tuple_1, U$1, pred.s_2_Tuple_1, clamp((lo) - (arity(T$0)), 0, arity(U$1)), clamp((hi) - (arity(T$0)), 0, arity(U$1)))
                  : snapshot.s_2_Tuple_1))
                }
                "#
            ),
        );
    }

    /// A shared reference has no lifetime positions, so its predicate holds
    /// nothing.
    #[test]
    fn immutable_reference() {
        assert_eq!(
            indirect_pred("type X<'a, T> = &'a T;"),
            dedent(
                r#"
                predicate p_Ref_immutable_ind(k: s_Ref_immutable, U$1: Type, lo: Int, hi: Int) {
                  (true)
                }

                function ext_Ref_immutable(snapshot: s_Ref_immutable, U$1: Type, pred: s_Ref_immutable, lo: Int, hi: Int): s_Ref_immutable
                  requires p_Ref_immutable_ind(pred, U$1, lo, hi)
                {
                  unfolding p_Ref_immutable_ind(pred, U$1, lo, hi) in (snapshot)
                }
                "#
            ),
        );
    }

    #[test]
    fn mutable_reference() {
        assert_eq!(
            indirect_pred("type X<'a, T> = &'a mut T;"),
            dedent(
                r#"
                predicate p_Ref_mutable_ind(k: s_Ref_mutable, U$1: Type, lo: Int, hi: Int) {
                  ((((lo) <= (0)) && ((0) < (hi))) ==>
                    (inhabited(U$1)) &&
                    (p_Param(k.s_Ref_mutable_0, U$1))) &&
                  (((clamp((lo) - (1), 0, arity(U$1))) < (clamp((hi) - (1), 0, arity(U$1)))) ==>
                    (p_Param_ind(((lo) <= (0)) && ((0) < (hi)) ? p_Param_snap(k.s_Ref_mutable_0, U$1) : k.s_Ref_mutable_2, U$1, clamp((lo) - (1), 0, arity(U$1)), clamp((hi) - (1), 0, arity(U$1)))))
                }

                function ext_Ref_mutable(snapshot: s_Ref_mutable, U$1: Type, pred: s_Ref_mutable, lo: Int, hi: Int): s_Ref_mutable
                  requires p_Ref_mutable_ind(pred, U$1, lo, hi)
                {
                  unfolding p_Ref_mutable_ind(pred, U$1, lo, hi) in (s_Ref_mutable_cons(snapshot.s_Ref_mutable_0, snapshot.s_Ref_mutable_1, ((lo) <= (0)) && ((0) < (hi))
                  ? (clamp((lo) - (1), 0, arity(U$1))) < (clamp((hi) - (1), 0, arity(U$1)))
                    ? ext_Param(p_Param_snap(pred.s_Ref_mutable_0, U$1), U$1, p_Param_snap(pred.s_Ref_mutable_0, U$1), clamp((lo) - (1), 0, arity(U$1)), clamp((hi) - (1), 0, arity(U$1)))
                    : p_Param_snap(pred.s_Ref_mutable_0, U$1)
                  : (clamp((lo) - (1), 0, arity(U$1))) < (clamp((hi) - (1), 0, arity(U$1)))
                    ? ext_Param(snapshot.s_Ref_mutable_2, U$1, pred.s_Ref_mutable_2, clamp((lo) - (1), 0, arity(U$1)), clamp((hi) - (1), 0, arity(U$1)))
                    : snapshot.s_Ref_mutable_2))
                }
                "#
            ),
        );
    }

    /// The predicate of a type parameter is abstract.
    #[test]
    fn type_param() {
        assert_eq!(
            indirect_pred("type X<T> = T;"),
            dedent(
                r#"
                predicate p_Param_ind(k: s_Param, T$0: Type, lo: Int, hi: Int)

                function ext_Param(snapshot: s_Param, T$0: Type, pred: s_Param, lo: Int, hi: Int): s_Param
                  requires p_Param_ind(pred, T$0, lo, hi)
                "#
            ),
        );
    }

    /// `'a` occurs at positions 0 and 1 of the field, and 1 is inside the
    /// referent of 0. Both field predicates are part of the one conjunct of
    /// `'a`, so no caller can extend the snapshot in between: the second
    /// should be on `ext_Ref_mutable(k.s_S_0, .., k.s_S_0, 0, 1)`, which is
    /// not implemented yet.
    #[test]
    fn enclosed_occurrence() {
        assert_eq!(
            indirect_pred(
                "struct S<'a>(&'a mut &'a mut i32);
                 type X<'a> = S<'a>;"
            ),
            dedent(
                r#"
                predicate p_S_ind(k: s_S, lo: Int, hi: Int) {
                  ((((lo) <= (0)) && ((0) < (hi))) ==>
                    (p_Ref_mutable_ind(k.s_S_0, s_Ref_mutable_type(s_Int_i32_type()), 0, 1)) &&
                    (p_Ref_mutable_ind(k.s_S_0, s_Ref_mutable_type(s_Int_i32_type()), 1, 2)))
                }

                function ext_S(snapshot: s_S, pred: s_S, lo: Int, hi: Int): s_S
                  requires p_S_ind(pred, lo, hi)
                {
                  unfolding p_S_ind(pred, lo, hi) in (s_S_cons(((lo) <= (0)) && ((0) < (hi))
                  ? ext_Ref_mutable(((lo) <= (0)) && ((0) < (hi))
                    ? ext_Ref_mutable(snapshot.s_S_0, s_Ref_mutable_type(s_Int_i32_type()), pred.s_S_0, 0, 1)
                    : snapshot.s_S_0, s_Ref_mutable_type(s_Int_i32_type()), pred.s_S_0, 1, 2)
                  : ((lo) <= (0)) && ((0) < (hi))
                    ? ext_Ref_mutable(snapshot.s_S_0, s_Ref_mutable_type(s_Int_i32_type()), pred.s_S_0, 0, 1)
                    : snapshot.s_S_0))
                }
                "#
            ),
        );
    }
}
