use std::collections::BTreeSet;

use prusti_rustc_interface::middle::ty;
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, FunctionIdn, PredicateIdn, Reify};

use crate::encoders::TyUseImpureEnc;

use super::{
    RustParamData, RustTy, RustTyDecomposition, TySpecifics,
    data::StructData,
    generics::{GArgsTyEnc, GParams, GenericParamsEnc},
    lifetime_query::{LifetimeQueryEnc, LifetimeQueryFns, LifetimeQueryShapeEnc},
    pure::{PureTyDatas, TyPureEnc},
    rust_ty::RustTyDatas,
};

/// Encodes for a type constructor `X`:
/// - the indirect predicate `p_X_ind(key: s_X, T.., q: LifetimeQuery)`: what
///   `X` reaches through references whose lifetimes the query `q` selects. The
///   key is the shallow snapshot of the value.
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

/// What `ext_X(.., pq)` requires of the query `pq`.
pub(crate) type QueryPre<'vir> =
    vir::ExprGenBool<'vir, vir::ExprLifetimeQuery<'vir>, vir::ExprKind<'vir>>;

#[derive(Clone)]
pub struct TyIndirectDep<'vir> {
    /// `None` if `ext_X` requires no agreement.
    pub(crate) agree: Option<Agree<'vir>>,
    /// `None` if `ext_X` requires nothing of its query.
    pub(crate) query_pre: Option<QueryPre<'vir>>,
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
                    // the own lifetime (0) and what is selected inside the referent (type arg 0)
                    let own = builder.query.selects_lifetime(builder.q, 0);
                    let inner = builder.query.selects_in_arg(builder.q, 0);
                    let inner_query = builder.query.selection_in_arg(builder.q, 0);
                    // If the own lifetime is selected, this piece owns the referent
                    // and reads the key of the inner selection from it; otherwise
                    // the inner selection is keyed by the (filled) value slot.
                    let owned_key = referent_impure.ref_to_snap(addr);
                    let slot_key: vir::ExprSnap<'vir> = pure.value_access.call()(key).upcast_ty();
                    let owned_inner = builder.apply(referent, owned_key, inner_query)?;
                    let slot_inner = builder.apply(referent, slot_key, inner_query)?;
                    Some(vcx.mk_conj(&[
                        mk_implies(vcx, own, referent_impure.ref_to_pred(vcx, addr, None)),
                        mk_implies(vcx, vcx.mk_conj(&[own, inner]), owned_inner),
                        mk_implies(vcx, vcx.mk_conj(&[mk_not(vcx, own), inner]), slot_inner),
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
                // Nothing behind these.
                TySpecifics::Primitive(_) | TySpecifics::Raw(_) | TySpecifics::Builtin(_) => {
                    Some(vcx.mk_bool::<true>())
                }
                // Abstract: the instantiation is unknown.
                TySpecifics::Param(_) | TySpecifics::Opaque(_) => None,
                // TODO: shared references and the elements of arrays/slices.
                TySpecifics::ImmRef(_) | TySpecifics::ArrayLike(_) => None,
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
            let mut query_pre = None;
            let (body, agree) = match &combined.specifics {
                TySpecifics::MutRef((rust, pure)) => {
                    let (k, kp) = (k.downcast_ty::<vir::CSnap>(), kp.downcast_ty::<vir::CSnap>());
                    let referent = rust.referent.decompose(ty.params);
                    let referent_impure = builder.deps.require_dep::<TyUseImpureEnc>(referent)?;
                    let own = builder.query.selects_lifetime(builder.q, 0);
                    let inner = builder.query.selects_in_arg(builder.q, 0);
                    let inner_query = builder.query.selection_in_arg(builder.q, 0);
                    let fns = builder.query;
                    query_pre = Some(vcx.mk_lazy_expr(
                        "ext_query_pre",
                        vir::TYPE_BOOL,
                        Box::new(move |vcx, pq: vir::ExprLifetimeQuery<'vir>| {
                            vcx.mk_disj(&[fns.selects_lifetime(pq, 0), fns.selects_in_arg(pq, 0)])
                                .kind
                        }),
                    ));
                    // If the own lifetime is in the parent, the parent owns the
                    // referent: the value slot becomes its snapshot (extended by
                    // the parent's inner selection). Otherwise extend inside it.
                    let owned = referent_impure.ref_to_snap(pure.deref_access.call()(kp));
                    let owned_ext = builder.ext(referent, owned, owned, inner_query)?;
                    let slot_ext = builder.ext(
                        referent,
                        pure.value_access.call()(k).upcast_ty(),
                        pure.value_access.call()(kp).upcast_ty(),
                        inner_query,
                    )?;
                    let value = vcx.mk_ternary_expr(
                        own,
                        vcx.mk_ternary_expr(inner, owned_ext, owned),
                        slot_ext,
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
                TySpecifics::Primitive(_) | TySpecifics::Raw(_) | TySpecifics::Builtin(_) => {
                    (Some(k), None)
                }
                TySpecifics::Param(_)
                | TySpecifics::Opaque(_)
                | TySpecifics::ImmRef(_)
                | TySpecifics::ArrayLike(_) => (None, None),
            };
            if let Some(query_pre) = query_pre {
                pres.push(query_pre.reify(vcx, builder.q));
            }
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
            Ok((TyIndirectLocal { predicate, ext }, TyIndirectDep { agree, query_pre }))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for output in Self::all_outputs_local_no_errors(program) {
            program.add_predicate(output.predicate);
            program.add_function(output.ext);
        }
        LifetimeQueryEnc::emit_outputs(program);
        LifetimeQueryShapeEnc::emit_outputs(program);
    }
}

/// A `selects_*` guard of a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Guard {
    /// `selects_lifetime(q, i)`
    Lifetime(usize),
    /// `selects_in_arg(q, j)`
    Arg(usize),
}

/// The query of a field (or part of one), in terms of the query `q` of the
/// constructor. Empty if it is statically `q_none()`.
struct SubQuery<'vir> {
    expr: vir::ExprLifetimeQuery<'vir>,
    /// The query is `q_none()` unless one of these holds.
    guards: BTreeSet<Guard>,
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
    /// `guards ==> p_F_ind(key(field), .., query(field))` for every field `F`
    /// with a non-empty query.
    fn fields(
        &mut self,
        key: vir::ExprCSnap<'vir>,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
    ) -> EncResult<'vir, Vec<vir::ExprBool<'vir>>> {
        let mut conjuncts = Vec::new();
        for (rust, pure) in &data.fields {
            let field_ty = rust.ty().decompose(self.params);
            let Some(query) = self.ty_query(field_ty)? else {
                continue;
            };
            let field_key = pure.read.call()(key);
            let pred = self.apply(field_ty, field_key, query.expr)?;
            conjuncts.push(mk_implies(self.vcx, self.guards(&query.guards), pred));
        }
        Ok(conjuncts)
    }

    /// Per field: `guards ? ext_F(K.f, .., Kp.f, query(field)) : K.f`, or just
    /// `K.f` if its query is empty. `None` if no field is extended.
    fn ext_fields(
        &mut self,
        k: vir::ExprCSnap<'vir>,
        kp: vir::ExprCSnap<'vir>,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
    ) -> EncResult<'vir, Option<Vec<vir::ExprSnap<'vir>>>> {
        let mut extended = false;
        let mut values = Vec::new();
        for (rust, pure) in &data.fields {
            let field_ty = rust.ty().decompose(self.params);
            let field_k = pure.read.call()(k);
            values.push(match self.ty_query(field_ty)? {
                Some(query) => {
                    extended = true;
                    let field_kp = pure.read.call()(kp);
                    let ext = self.ext(field_ty, field_k, field_kp, query.expr)?;
                    self.vcx
                        .mk_ternary_expr(self.guards(&query.guards), ext, field_k)
                }
                None => field_k,
            });
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
            if self.ty_query(field_ty)?.is_none() {
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

    fn guards(&self, guards: &BTreeSet<Guard>) -> vir::ExprBool<'vir> {
        let guards = guards
            .iter()
            .map(|guard| match *guard {
                Guard::Lifetime(i) => self.query.selects_lifetime(self.q, i),
                Guard::Arg(j) => self.query.selects_in_arg(self.q, j),
            })
            .collect::<Vec<_>>();
        self.vcx.mk_disj(&guards)
    }

    /// The query of a value of type `ty` inside the constructor. A type
    /// parameter passes on what `q` selects inside it; any other type gets a
    /// query of its own shape, built from its generic arguments.
    fn ty_query(&mut self, ty: RustTyDecomposition<'vir>) -> EncResult<'vir, Option<SubQuery<'vir>>> {
        if let TySpecifics::Param(RustParamData::Generic) = &ty.ty.specifics {
            let arg = ty.args.args()[0].expect_ty();
            return Ok(match arg.kind() {
                ty::TyKind::Param(param) => {
                    let j = self.position(param.index, |arg| arg.as_type().is_some());
                    Some(SubQuery {
                        expr: self.query.selection_in_arg(self.q, j),
                        guards: BTreeSet::from([Guard::Arg(j)]),
                    })
                }
                // TODO: lifetimes inside associated types are not covered.
                _ => None,
            });
        }
        let mut guards = BTreeSet::new();
        let mut lifetimes = Vec::new();
        let mut args = Vec::new();
        for arg in ty.args.args() {
            if let Some(region) = arg.as_region() {
                lifetimes.push(match region.kind() {
                    ty::RegionKind::ReEarlyParam(param) => {
                        let i = self.position(param.index, |arg| arg.as_region().is_some());
                        guards.insert(Guard::Lifetime(i));
                        self.query.selects_lifetime(self.q, i)
                    }
                    // Not a lifetime of the constructor (e.g. `'static`).
                    _ => self.vcx.mk_bool::<false>(),
                });
            } else if let Some(arg_ty) = arg.as_type() {
                let arg_ty = RustTyDecomposition::from_ty(arg_ty, self.params);
                args.push(match self.ty_query(arg_ty)? {
                    Some(query) => {
                        guards.extend(query.guards);
                        query.expr
                    }
                    None => self.query.q_none(),
                });
            }
        }
        if guards.is_empty() {
            return Ok(None);
        }
        let shape = self
            .deps
            .require_dep::<LifetimeQueryShapeEnc>((lifetimes.len(), args.len()))?;
        Ok(Some(SubQuery {
            expr: shape(lifetimes.as_slice(), args.as_slice()),
            guards,
        }))
    }

    /// The position of the constructor's generic parameter `index` among
    /// those of its kind.
    fn position(&self, index: u32, of_kind: impl Fn(ty::GenericArg<'vir>) -> bool) -> usize {
        self.params.rust_params()[..index as usize]
            .iter()
            .filter(|arg| of_kind(**arg))
            .count()
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

fn mk_not<'vir>(vcx: &'vir vir::VirCtxt<'_>, expr: vir::ExprBool<'vir>) -> vir::ExprBool<'vir> {
    vcx.mk_unary_op_expr(vir::UnOpKind::Not, expr.upcast_ty())
        .downcast_ty()
}
