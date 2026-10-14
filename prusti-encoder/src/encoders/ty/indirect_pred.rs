use prusti_rustc_interface::middle::ty;
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, FunctionIdn, PredicateIdn, Reify};

use crate::encoders::TyUseImpureEnc;

use super::{
    RustTy, RustTyDecomposition, TySpecifics,
    data::StructData,
    generics::{GArgsTyEnc, GParams, GenericParamsEnc},
    lifetime_query::{LifetimeQueryEnc, LifetimeQueryFns},
    occurrences::{End, Occurrence, occurrence_query, occurrences, param_position},
    pure::{PureTyDatas, TyPureEnc},
    rust_ty::RustTyDatas,
};

/// Encodes for a type constructor `X`:
/// - the indirect predicate `p_X_ind(key: s_X, T.., q: LifetimeQuery)`: what
///   `X` reaches through the references of the one parameter occurrence the
///   query `q` selects, i.e. all occurrences of that parameter in the fields
///   (one conjunct each, see `occurrences`). The key is the shallow snapshot
///   of the value.
/// - the key extension `ext_X(K: s_X, T.., Kp: s_X, pq: LifetimeQuery): s_X`:
///   the key `K` with the cells owned by the piece `p_X_ind(Kp, .., pq)` read
///   in. `K` supplies the values, `Kp` only the permission; both must agree on
///   the discriminants that `ext_X` branches on (a precondition).
pub struct TyIndirectEnc;

pub type TyIndirectPred<'vir> =
    PredicateIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::LifetimeQuery)>;

pub type TyIndirectExt<'vir> = FunctionIdn<
    'vir,
    (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::Snap, vir::LifetimeQuery),
    vir::Snap,
>;

#[derive(Debug, Clone, Copy)]
pub struct TyIndirectRef<'vir> {
    pub pred: TyIndirectPred<'vir>,
    pub ext: TyIndirectExt<'vir>,
}

impl<'vir> task_encoder::OutputRefAny for TyIndirectRef<'vir> {}

/// The discriminant agreement of two keys `(K, Kp)` that `ext_X(K, .., Kp, ..)`
/// requires.
pub(crate) type Agree<'vir> =
    vir::ExprGenBool<'vir, (vir::ExprSnap<'vir>, vir::ExprSnap<'vir>), vir::ExprKind<'vir>>;

#[derive(Clone)]
pub struct TyIndirectDep<'vir> {
    /// `None` if `ext_X` requires no agreement.
    pub(crate) agree: Option<Agree<'vir>>,
}

#[derive(Clone)]
pub struct TyIndirectLocal<'vir> {
    predicate: vir::Predicate<'vir>,
    ext: vir::Function<'vir>,
}

type EncResult<'vir, T> = Result<T, EncodeFullError<'vir, TyIndirectEnc>>;

impl TaskEncoder for TyIndirectEnc {
    task_encoder::encoder_cache!(TyIndirectEnc);
    const ENCODER_NAME: &'static str = "indirect predicate encoder";

    type TaskDescription<'vir> = RustTy<'vir>;
    type OutputRef<'vir> = TyIndirectRef<'vir>;
    type OutputFullLocal<'vir> = TyIndirectLocal<'vir>;
    type OutputFullDependency<'vir> = TyIndirectDep<'vir>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        let ty = *task_key;
        let snap = deps.require_dep::<TyPureEnc>(ty)?;
        let params = deps.require_dep::<GenericParamsEnc>(ty.params)?;
        let query = deps.require_dep::<LifetimeQueryEnc>(())?;
        vir::with_vcx(|vcx| {
            let pred = PredicateIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "p_{}_ind", ty.name())),
                (
                    snap.snapshot,
                    params.ty_args(),
                    params.const_args(),
                    vir::TYPE_LIFETIME_QUERY,
                ),
            );
            let ext = FunctionIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "ext_{}", ty.name())),
                (
                    snap.snapshot,
                    params.ty_args(),
                    params.const_args(),
                    snap.snapshot,
                    vir::TYPE_LIFETIME_QUERY,
                ),
                snap.snapshot,
            );
            deps.emit_output_ref(ty, TyIndirectRef { pred, ext })?;

            let key_decl = vcx.mk_local_decl("key", snap.snapshot);
            let q_decl = vcx.mk_local_decl("q", vir::TYPE_LIFETIME_QUERY);
            let mut builder = IndirectBuilder {
                vcx,
                deps,
                params: ty.params,
                query,
                q: vcx.mk_local_ex(q_decl),
            };
            let key = vcx.mk_local_ex(key_decl);
            let combined = ty.zip(snap);
            let body = match &combined.specifics {
                TySpecifics::MutRef((rust, pure)) => {
                    let key = key.downcast_ty::<vir::CSnap>();
                    let referent = rust.referent.decompose(ty.params);
                    let referent_impure = builder.deps.require_dep::<TyUseImpureEnc>(referent)?;
                    let addr = pure.deref_access.call()(key);
                    // `q_here(0)`: the reference itself, which owns the referent;
                    // `q_in(0, rest)`: inside the referent, keyed by the (filled)
                    // value slot.
                    let slot_key: vir::ExprSnap<'vir> = pure.value_access.call()(key).upcast_ty();
                    let inner = builder.apply(referent, slot_key, builder.query.rest(builder.q))?;
                    Some(vcx.mk_conj(&[
                        mk_implies(
                            vcx,
                            builder.query.is_here(builder.q, 0),
                            referent_impure.ref_to_pred(vcx, addr, None),
                        ),
                        mk_implies(vcx, builder.query.is_in(builder.q, 0), inner),
                    ]))
                }
                TySpecifics::StructLike(data) => {
                    let fields = builder.fields(key.downcast_ty::<vir::CSnap>(), data)?;
                    Some(vcx.mk_conj(&fields))
                }
                TySpecifics::EnumLike(data) => {
                    let key = key.downcast_ty::<vir::CSnap>();
                    let discr = data.data.1.snap_to_discr_snap.call()(key);
                    let mut variants = Vec::new();
                    for variant in &data.variants {
                        let fields = builder.fields(key, &variant.inner)?;
                        if !fields.is_empty() {
                            variants.push(mk_implies(
                                vcx,
                                vcx.mk_eq_expr(discr, variant.data.1.discr),
                                vcx.mk_conj(&fields),
                            ));
                        }
                    }
                    Some(vcx.mk_conj(&variants))
                }
                // Nothing behind these. TODO: shared references and the
                // elements of arrays/slices (see `occurrences`).
                TySpecifics::Primitive(_)
                | TySpecifics::Raw(_)
                | TySpecifics::Builtin(_)
                | TySpecifics::ImmRef(_)
                | TySpecifics::ArrayLike(_) => Some(vcx.mk_bool::<true>()),
                // Abstract: the instantiation is unknown.
                TySpecifics::Param(_) | TySpecifics::Opaque(_) => None,
            };
            let predicate = vcx.mk_predicate(
                pred,
                (key_decl, params.ty_decls(), params.const_decls(), q_decl),
                body,
            );

            // ext_X
            let k_decl = vcx.mk_local_decl("K", snap.snapshot);
            let kp_decl = vcx.mk_local_decl("Kp", snap.snapshot);
            let pq_decl = vcx.mk_local_decl("pq", vir::TYPE_LIFETIME_QUERY);
            let k = vcx.mk_local_ex(k_decl);
            let kp = vcx.mk_local_ex(kp_decl);
            builder.q = vcx.mk_local_ex(pq_decl);
            // `acc(p_X_ind(Kp, .., pq), wildcard)`
            let parent = || {
                pred(kp, params.ty_exprs(), params.const_exprs(), builder.q)(Some(
                    vcx.mk_wildcard(),
                ))
            };
            let unfolding = |expr: vir::ExprSnap<'vir>| vcx.mk_unfolding_expr(parent(), expr);
            let mut pres = vec![vcx.mk_predicate_app_expr(parent())];
            let (body, agree) = match &combined.specifics {
                TySpecifics::MutRef((rust, pure)) => {
                    let (k, kp) = (k.downcast_ty::<vir::CSnap>(), kp.downcast_ty::<vir::CSnap>());
                    let referent = rust.referent.decompose(ty.params);
                    let referent_impure = builder.deps.require_dep::<TyUseImpureEnc>(referent)?;
                    // The reference itself: the value slot becomes the snapshot
                    // of the referent the parent owns. Inside the referent:
                    // extend the slot.
                    let owned = referent_impure.ref_to_snap(pure.deref_access.call()(kp));
                    let slot = pure.value_access.call()(k).upcast_ty();
                    let slot_ext = builder.ext(
                        referent,
                        slot,
                        pure.value_access.call()(kp).upcast_ty(),
                        builder.query.rest(builder.q),
                    )?;
                    let value = vcx.mk_ternary_expr(
                        builder.query.is_here(builder.q, 0),
                        owned,
                        vcx.mk_ternary_expr(builder.query.is_in(builder.q, 0), slot_ext, slot),
                    );
                    let cons = (pure.prim_to_snap)(
                        pure.deref_access.call()(k),
                        pure.metadata_access.call()(k),
                        value.downcast_ty::<vir::PSnap>(),
                    );
                    (Some(unfolding(cons.upcast_ty())), None)
                }
                TySpecifics::StructLike(data) => {
                    let (k, kp) = (k.downcast_ty::<vir::CSnap>(), kp.downcast_ty::<vir::CSnap>());
                    let fields = builder.ext_fields(k, kp, data)?;
                    let body = match fields {
                        Some(fields) => {
                            let cons = (data.data.1.field_snaps_to_snap)(fields.as_slice());
                            unfolding(cons.upcast_ty())
                        }
                        None => k.upcast_ty(),
                    };
                    let fields = builder.agree_fields(data)?;
                    let agree = (!fields.is_empty()).then(|| {
                        vcx.mk_lazy_expr(
                            "agree_struct",
                            vir::TYPE_BOOL,
                            Box::new(move |vcx, (k, kp): (vir::ExprSnap<'vir>, vir::ExprSnap<'vir>)| {
                                agree_fields(vcx, &fields, k, kp).kind
                            }),
                        )
                    });
                    (Some(body), agree)
                }
                TySpecifics::EnumLike(data) => {
                    let (k, kp) = (k.downcast_ty::<vir::CSnap>(), kp.downcast_ty::<vir::CSnap>());
                    let snap_to_discr = data.data.1.snap_to_discr_snap;
                    let discr = snap_to_discr.call()(k);
                    let mut body = k.upcast_ty();
                    let mut variants = Vec::new();
                    for variant in data.variants.iter().rev() {
                        let Some(fields) = builder.ext_fields(k, kp, &variant.inner)? else {
                            continue;
                        };
                        let cons = (variant.inner.data.1.field_snaps_to_snap)(fields.as_slice());
                        body = vcx.mk_ternary_expr(
                            vcx.mk_eq_expr(discr, variant.data.1.discr),
                            cons.upcast_ty(),
                            body,
                        );
                        variants.push((
                            variant.data.1.discr,
                            builder.agree_fields(&variant.inner)?,
                        ));
                    }
                    // Agreement is needed iff `ext_X` branches.
                    let branches = !variants.is_empty();
                    let agree = branches.then(|| {
                        vcx.mk_lazy_expr(
                            "agree_enum",
                            vir::TYPE_BOOL,
                            Box::new(move |vcx, (k, kp): (vir::ExprSnap<'vir>, vir::ExprSnap<'vir>)| {
                                let k_discr = snap_to_discr.call()(k.downcast_ty());
                                let kp_discr = snap_to_discr.call()(kp.downcast_ty());
                                let mut conjuncts = vec![vcx.mk_eq_expr(k_discr, kp_discr)];
                                for (discr, fields) in &variants {
                                    if !fields.is_empty() {
                                        conjuncts.push(mk_implies(
                                            vcx,
                                            vcx.mk_eq_expr(k_discr, *discr),
                                            agree_fields(vcx, fields, k, kp),
                                        ));
                                    }
                                }
                                vcx.mk_conj(&conjuncts).kind
                            }),
                        )
                    });
                    let body = if branches { unfolding(body) } else { body };
                    (Some(body), agree)
                }
                TySpecifics::Primitive(_)
                | TySpecifics::Raw(_)
                | TySpecifics::Builtin(_)
                | TySpecifics::ImmRef(_)
                | TySpecifics::ArrayLike(_) => (Some(k), None),
                TySpecifics::Param(_) | TySpecifics::Opaque(_) => (None, None),
            };
            if let Some(agree) = agree {
                pres.push(agree.reify(vcx, (k, kp)));
            }
            let ext = vcx.mk_function(
                ext,
                (k_decl, params.ty_decls(), params.const_decls(), kp_decl, pq_decl),
                vcx.alloc_slice(&pres),
                &[],
                None,
                body,
            );
            Ok((TyIndirectLocal { predicate, ext }, TyIndirectDep { agree }))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for output in Self::all_outputs_local_no_errors(program) {
            program.add_predicate(output.predicate);
            program.add_function(output.ext);
        }
        LifetimeQueryEnc::emit_outputs(program);
    }
}


/// A generic parameter of the constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Param {
    Lifetime(usize),
    Ty(usize),
}

/// The occurrences of the constructor's parameters in the type of a field.
struct FieldOccurrences<'vir> {
    ty: RustTyDecomposition<'vir>,
    occurrences: Vec<Occurrence<'vir>>,
    /// Per occurrence: the parameter it is, `None` for others (e.g. `'static`).
    params: Vec<Option<Param>>,
}

impl FieldOccurrences<'_> {
    /// The parameters occurring, in order.
    fn params(&self) -> Vec<Param> {
        let mut params = self.params.iter().flatten().copied().collect::<Vec<_>>();
        params.sort();
        params.dedup();
        params
    }
}

struct IndirectBuilder<'a, 'vir> {
    vcx: &'vir vir::VirCtxt<'vir>,
    deps: &'a mut TaskEncoderDependencies<'vir, TyIndirectEnc>,
    /// The constructor's generic parameters.
    params: GParams<'vir>,
    query: LifetimeQueryFns<'vir>,
    /// The query of the constructor.
    q: vir::ExprLifetimeQuery<'vir>,
}

impl<'vir> IndirectBuilder<'_, 'vir> {
    /// Per parameter selected by `q`: `guard ==> pieces`, the pieces of its
    /// occurrences in the fields.
    fn fields(
        &mut self,
        key: vir::ExprCSnap<'vir>,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
    ) -> EncResult<'vir, Vec<vir::ExprBool<'vir>>> {
        let mut by_param: Vec<(Param, Vec<vir::ExprBool<'vir>>)> = Vec::new();
        for (rust, pure) in &data.fields {
            let field = self.field_occurrences(rust.ty().decompose(self.params));
            let field_key = pure.read.call()(key);
            for param in field.params() {
                let pieces = self.pieces(&field, param, field_key)?;
                match by_param.iter_mut().find(|(p, _)| *p == param) {
                    Some((_, all)) => all.extend(pieces),
                    None => by_param.push((param, pieces)),
                }
            }
        }
        by_param.sort_by_key(|(p, _)| *p);
        Ok(by_param
            .into_iter()
            .map(|(param, pieces)| mk_implies(self.vcx, self.guard(param), self.vcx.mk_conj(&pieces)))
            .collect())
    }

    /// The pieces of the occurrences of `param` in a field with key `key`.
    /// The key of an occurrence is extended by the pieces of the occurrences
    /// of the same parameter enclosing it (the others are in `key` already).
    fn pieces(
        &mut self,
        field: &FieldOccurrences<'vir>,
        param: Param,
        key: vir::ExprSnap<'vir>,
    ) -> EncResult<'vir, Vec<vir::ExprBool<'vir>>> {
        let keys = self.occurrence_keys(field, param, key)?;
        let mut pieces = Vec::new();
        for (idx, key) in keys {
            let query = self.occurrence_query(field, idx);
            pieces.push(self.apply(field.ty, key, query)?);
        }
        Ok(pieces)
    }

    /// The occurrences of `param` in the field and their keys, enclosing ones
    /// first.
    fn occurrence_keys(
        &mut self,
        field: &FieldOccurrences<'vir>,
        param: Param,
        key: vir::ExprSnap<'vir>,
    ) -> EncResult<'vir, Vec<(usize, vir::ExprSnap<'vir>)>> {
        let mut keys: Vec<(usize, vir::ExprSnap<'vir>)> = Vec::new();
        for (idx, occurrence) in field.occurrences.iter().enumerate() {
            if field.params[idx] != Some(param) {
                continue;
            }
            let mut occurrence_key = key;
            for (parent, parent_key) in keys.clone() {
                if occurrence.parents.contains(&parent) {
                    let query = self.occurrence_query(field, parent);
                    occurrence_key = self.ext(field.ty, occurrence_key, parent_key, query)?;
                }
            }
            keys.push((idx, occurrence_key));
        }
        Ok(keys)
    }

    /// Per field: the field of `K` extended by the occurrences of the
    /// parameter `pq` selects, or just `K.f` if none occurs. `None` if no
    /// field is extended.
    fn ext_fields(
        &mut self,
        k: vir::ExprCSnap<'vir>,
        kp: vir::ExprCSnap<'vir>,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
    ) -> EncResult<'vir, Option<Vec<vir::ExprSnap<'vir>>>> {
        let mut extended = false;
        let mut values = Vec::new();
        for (rust, pure) in &data.fields {
            let field = self.field_occurrences(rust.ty().decompose(self.params));
            let field_k = pure.read.call()(k);
            let field_kp = pure.read.call()(kp);
            let mut value = field_k;
            for param in field.params().into_iter().rev() {
                extended = true;
                let mut ext = field_k;
                for (idx, parent_key) in self.occurrence_keys(&field, param, field_kp)? {
                    let query = self.occurrence_query(&field, idx);
                    ext = self.ext(field.ty, ext, parent_key, query)?;
                }
                value = self.vcx.mk_ternary_expr(self.guard(param), ext, value);
            }
            values.push(value);
        }
        Ok(extended.then_some(values))
    }

    /// The agreement `ext_F` requires, for every field `F` that `ext_X` may
    /// extend.
    fn agree_fields(
        &mut self,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
    ) -> EncResult<'vir, Vec<(FieldRead<'vir>, Agree<'vir>)>> {
        let mut result = Vec::new();
        for (rust, pure) in &data.fields {
            let field_ty = rust.ty().decompose(self.params);
            if self.field_occurrences(field_ty).params().is_empty() {
                continue;
            }
            if let Some(agree) = self.deps.require_dep::<TyIndirectEnc>(field_ty.ty)?.agree {
                result.push((pure.read, agree));
            }
        }
        Ok(result)
    }

    /// `ext_F(k, .., kp, query)` for the (decomposed) type `F`.
    fn ext(
        &mut self,
        ty: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        kp: vir::ExprSnap<'vir>,
        query: vir::ExprLifetimeQuery<'vir>,
    ) -> EncResult<'vir, vir::ExprSnap<'vir>> {
        let ext = self.deps.require_ref::<TyIndirectEnc>(ty.ty)?.ext;
        let args = self.deps.require_dep::<GArgsTyEnc>(ty.args)?;
        Ok(ext(k, args.get_ty(), args.get_const(), kp, query))
    }

    /// `p_F_ind(key, .., query)` for the (decomposed) type `F`.
    fn apply(
        &mut self,
        ty: RustTyDecomposition<'vir>,
        key: vir::ExprSnap<'vir>,
        query: vir::ExprLifetimeQuery<'vir>,
    ) -> EncResult<'vir, vir::ExprBool<'vir>> {
        let pred = self.deps.require_ref::<TyIndirectEnc>(ty.ty)?.pred;
        let args = self.deps.require_dep::<GArgsTyEnc>(ty.args)?;
        let app = pred(key, args.get_ty(), args.get_const(), query)(None);
        Ok(self.vcx.mk_predicate_app_expr(app))
    }

    /// Does `q` select `param`?
    fn guard(&self, param: Param) -> vir::ExprBool<'vir> {
        match param {
            Param::Lifetime(i) => self.query.is_here(self.q, i),
            Param::Ty(j) => self.query.is_in(self.q, j),
        }
    }

    /// The query of occurrence `idx` in the field's type; inside a type
    /// parameter, it continues with the rest of `q`.
    fn occurrence_query(&self, field: &FieldOccurrences<'vir>, idx: usize) -> vir::ExprLifetimeQuery<'vir> {
        occurrence_query(
            &self.query,
            &field.occurrences[idx].path,
            Some(self.query.rest(self.q)),
        )
    }

    fn field_occurrences(&self, ty: RustTyDecomposition<'vir>) -> FieldOccurrences<'vir> {
        // TODO: lifetimes enclosing each other are not supported (the nodes
        // of such a type cannot be encoded either).
        let occurrences = occurrences(ty).unwrap_or_default();
        let params = occurrences
            .iter()
            .map(|o| match o.end {
                End::Region(region) => match region.kind() {
                    ty::RegionKind::ReEarlyParam(param) => Some(Param::Lifetime(param_position(
                        self.params,
                        param.index,
                        |arg| arg.as_region().is_some(),
                    ))),
                    // Not a lifetime of the constructor (e.g. `'static`).
                    _ => None,
                },
                End::Param(param) => Some(Param::Ty(param_position(self.params, param.index, |arg| {
                    arg.as_type().is_some()
                }))),
            })
            .collect();
        FieldOccurrences { ty, occurrences, params }
    }
}

type FieldRead<'vir> = vir::AdtDestructor<'vir, vir::CSnap, vir::Snap>;

/// The conjunction of the fields' agreements on `(K.f, Kp.f)`.
fn agree_fields<'vir>(
    vcx: &'vir vir::VirCtxt<'_>,
    fields: &[(FieldRead<'vir>, Agree<'vir>)],
    k: vir::ExprSnap<'vir>,
    kp: vir::ExprSnap<'vir>,
) -> vir::ExprBool<'vir> {
    let conjuncts = fields
        .iter()
        .map(|(read, agree)| {
            agree.reify(
                vcx,
                (
                    read.call()(k.downcast_ty()),
                    read.call()(kp.downcast_ty()),
                ),
            )
        })
        .collect::<Vec<_>>();
    vcx.mk_conj(&conjuncts)
}

fn mk_implies<'vir>(
    vcx: &'vir vir::VirCtxt<'_>,
    lhs: vir::ExprBool<'vir>,
    rhs: vir::ExprBool<'vir>,
) -> vir::ExprBool<'vir> {
    vcx.mk_bin_op_expr(vir::BinOpKind::Implies, lhs, rhs)
        .downcast_ty()
}
