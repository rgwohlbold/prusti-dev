use std::collections::BTreeSet;

use prusti_rustc_interface::middle::ty;
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, PredicateIdn};

use crate::encoders::TyUseImpureEnc;

use super::{
    RustParamData, RustTy, RustTyDecomposition, TySpecifics,
    data::StructData,
    generics::{GArgsTyEnc, GParams, GenericParamsEnc},
    lifetime_query::{LifetimeQueryEnc, LifetimeQueryFns, LifetimeQueryShapeEnc},
    pure::{PureTyDatas, TyPureEnc},
    rust_ty::RustTyDatas,
};

/// Encodes the indirect predicate `p_X_ind(key: s_X, T.., q: LifetimeQuery)` of
/// a type constructor `X`: what `X` reaches through references whose lifetimes
/// the query `q` selects. The key is the shallow snapshot of the value.
pub struct TyIndirectEnc;

pub type TyIndirectPred<'vir> =
    PredicateIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::LifetimeQuery)>;

#[derive(Debug, Clone, Copy)]
pub struct TyIndirectRef<'vir> {
    pub pred: TyIndirectPred<'vir>,
}

impl<'vir> task_encoder::OutputRefAny for TyIndirectRef<'vir> {}

type EncResult<'vir, T> = Result<T, EncodeFullError<'vir, TyIndirectEnc>>;

impl TaskEncoder for TyIndirectEnc {
    task_encoder::encoder_cache!(TyIndirectEnc);
    const ENCODER_NAME: &'static str = "indirect predicate encoder";

    type TaskDescription<'vir> = RustTy<'vir>;
    type OutputRef<'vir> = TyIndirectRef<'vir>;
    type OutputFullLocal<'vir> = vir::Predicate<'vir>;

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
            deps.emit_output_ref(ty, TyIndirectRef { pred })?;

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
            Ok((predicate, ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for predicate in Self::all_outputs_local_no_errors(program) {
            program.add_predicate(predicate);
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
