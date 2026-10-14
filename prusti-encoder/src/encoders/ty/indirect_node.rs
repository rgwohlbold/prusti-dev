use std::collections::HashSet;

use pcg::borrow_pcg::region_projection::{
    ExtractRegionsCtxt, LifetimeProjectionIdx, PcgRegion, Region,
};
use prusti_rustc_interface::{index::IndexVec, middle::ty, span::def_id::DefId};
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{MethodIdn, Reify};

use crate::encoders::{MirLocalDefEnc, MirLocalDefEncTask};

use super::{
    RustTyDecomposition, TySpecifics,
    generics::{GArgsTyEnc, GParams, GenericParamsEnc},
    indirect::PrustiPcgCtxt,
    indirect_pred::TyIndirectEnc,
    lifetime_query::{LifetimeQueryEnc, LifetimeQueryShapeEnc},
};

type RegionIdx = LifetimeProjectionIdx<Region>;

/// The predicate of a PCG lifetime projection node `p↓r`, over the snapshot
/// `S` of its base `p`.
pub type NodePred<'vir> = vir::ExprGenBool<'vir, vir::ExprSnap<'vir>, vir::ExprKind<'vir>>;

/// Encodes, for a type `T` (the type of the base of PCG lifetime projection
/// nodes), the indirect predicate of each node `p↓r`:
/// `p_T_ind(key(r), .., Q_r)`, where `Q_r` selects the region `r` in `T`. The
/// key is `S` if no other region's reference encloses `r` (the piece reads its
/// key itself); otherwise it is `S` extended by the pieces of the enclosing
/// regions, outer ones first: `ext_T(.. ext_T(S, .., key(p_1), Q_p1) .., key(p_k), Q_pk)`.
pub struct IndirectNodesEnc;

#[derive(Clone)]
pub struct IndirectNodes<'vir> {
    /// Per region of the type (in PCG region index order): its predicate,
    /// `None` if the node holds nothing.
    preds: IndexVec<RegionIdx, Option<NodePred<'vir>>>,
    /// The region indices, enclosing regions first.
    order: Vec<RegionIdx>,
}

impl<'vir> IndirectNodes<'vir> {
    #[allow(dead_code)]
    pub fn get(&self, idx: RegionIdx) -> Option<NodePred<'vir>> {
        self.preds[idx]
    }

    /// The predicates of all nodes, enclosing regions first (so that the keys
    /// of later ones are well-defined given the earlier ones).
    pub fn ordered(&self) -> impl Iterator<Item = NodePred<'vir>> + '_ {
        self.order.iter().filter_map(|idx| self.preds[*idx])
    }
}

impl TaskEncoder for IndirectNodesEnc {
    task_encoder::encoder_cache!(IndirectNodesEnc);
    const ENCODER_NAME: &'static str = "indirect nodes encoder";

    type TaskDescription<'vir> = RustTyDecomposition<'vir>;
    type OutputFullDependency<'vir> = IndirectNodes<'vir>;
    type EncodingError = String;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;
        let decomp = *task_key;
        let error = |msg: String| EncodeFullError::EncodingError(msg, None);
        let regions = PrustiPcgCtxt.extract_regions(decomp);

        // which region's reference encloses which region
        let mut enclosure = Enclosure::default();
        enclosure.walk(decomp);
        for (idx, r) in regions.iter_enumerated() {
            if enclosure.shared.contains(r) {
                return Err(error(format!(
                    "indirect predicates through shared references are not supported \
                     (region {idx:?} of {decomp:?})"
                )));
            }
        }
        let parents: IndexVec<RegionIdx, Vec<RegionIdx>> = regions
            .iter()
            .map(|r| {
                regions
                    .iter_enumerated()
                    .filter(|(_, p)| *p != r && enclosure.pairs.contains(&(**p, *r)))
                    .map(|(p_idx, _)| p_idx)
                    .collect()
            })
            .collect();

        // enclosing regions first
        let mut order = Vec::new();
        while order.len() < regions.len() {
            let next = regions.indices().find(|idx| {
                !order.contains(idx) && parents[*idx].iter().all(|p| order.contains(p))
            });
            let Some(next) = next else {
                return Err(error(format!(
                    "regions enclosing each other are not supported ({decomp:?})"
                )));
            };
            order.push(next);
        }
        let parents: IndexVec<RegionIdx, Vec<RegionIdx>> = parents
            .into_iter()
            .map(|ps| order.iter().copied().filter(|idx| ps.contains(idx)).collect())
            .collect();

        let queries = regions
            .iter()
            .map(|r| region_query(deps, decomp, *r))
            .collect::<Result<IndexVec<RegionIdx, _>, _>>()?;
        let ind = deps.require_ref::<TyIndirectEnc>(decomp.ty)?;
        let args = deps.require_dep::<GArgsTyEnc>(decomp.args)?;
        let (tys, consts) = (args.get_ty(), args.get_const());

        vir::with_vcx(|vcx| {
            let preds = regions
                .indices()
                .map(|idx| {
                    let query = queries[idx]?;
                    let parents = parents.clone();
                    let queries = queries.clone();
                    Some(vcx.mk_lazy_expr(
                        "indirect_node",
                        vir::TYPE_BOOL,
                        Box::new(move |vcx, s: vir::ExprSnap<'vir>| {
                            let key = key(idx, s, &parents, &queries, &|k, kp, q| {
                                (ind.ext)(k, tys, consts, kp, q)
                            });
                            vcx.mk_predicate_app_expr((ind.pred)(key, tys, consts, query)(None))
                                .kind
                        }),
                    ))
                })
                .collect();
            Ok(((), IndirectNodes { preds, order }))
        })
    }
}

/// The key of region `idx`: `s`, extended by the pieces of its enclosing
/// regions (`parents`, outer ones first).
fn key<'vir>(
    idx: RegionIdx,
    s: vir::ExprSnap<'vir>,
    parents: &IndexVec<RegionIdx, Vec<RegionIdx>>,
    queries: &IndexVec<RegionIdx, Option<vir::ExprLifetimeQuery<'vir>>>,
    ext: &dyn Fn(
        vir::ExprSnap<'vir>,
        vir::ExprSnap<'vir>,
        vir::ExprLifetimeQuery<'vir>,
    ) -> vir::ExprSnap<'vir>,
) -> vir::ExprSnap<'vir> {
    parents[idx].iter().fold(s, |k, parent| {
        let parent_key = key(*parent, s, parents, queries, ext);
        // An enclosing region occurs in the type's arguments, so its query
        // selects something.
        ext(k, parent_key, queries[*parent].unwrap())
    })
}

/// The query selecting region `r` in the type `decomp`, `None` if it is empty.
fn region_query<'vir>(
    deps: &mut TaskEncoderDependencies<'vir, IndirectNodesEnc>,
    decomp: RustTyDecomposition<'vir>,
    r: PcgRegion<'vir>,
) -> Result<Option<vir::ExprLifetimeQuery<'vir>>, EncodeFullError<'vir, IndirectNodesEnc>> {
    let query = deps.require_dep::<LifetimeQueryEnc>(())?;
    let mut selected = false;
    let mut lifetimes = Vec::new();
    let mut args = Vec::new();
    for arg in decomp.args.args() {
        if let Some(region) = arg.as_region() {
            let is_r = PcgRegion::from(region) == r;
            selected |= is_r;
            lifetimes.push(vir::with_vcx(|vcx| {
                if is_r { vcx.mk_bool::<true>() } else { vcx.mk_bool::<false>() }
            }));
        } else if let Some(arg_ty) = arg.as_type() {
            let arg_query = match arg_ty.kind() {
                // TODO: `RegionsIn(T)` of a type parameter.
                ty::TyKind::Param(_) | ty::TyKind::Alias(..) => None,
                _ => {
                    let arg_ty = RustTyDecomposition::from_ty(arg_ty, decomp.args.context());
                    region_query(deps, arg_ty, r)?
                }
            };
            selected |= arg_query.is_some();
            args.push(arg_query.unwrap_or_else(|| query.q_none()));
        }
    }
    if !selected {
        return Ok(None);
    }
    let shape = deps.require_dep::<LifetimeQueryShapeEnc>((lifetimes.len(), args.len()))?;
    Ok(Some(shape(lifetimes.as_slice(), args.as_slice())))
}

/// Which region's reference encloses which region, found by walking the type
/// through references and fields.
#[derive(Default)]
struct Enclosure<'vir> {
    visited: HashSet<RustTyDecomposition<'vir>>,
    /// `(r', r)`: a reference `&'r' mut U` with `r` in `U`.
    pairs: HashSet<(PcgRegion<'vir>, PcgRegion<'vir>)>,
    /// The regions of shared references and of their referents.
    shared: HashSet<PcgRegion<'vir>>,
}

impl<'vir> Enclosure<'vir> {
    fn walk(&mut self, decomp: RustTyDecomposition<'vir>) {
        if !self.visited.insert(decomp) {
            return;
        }
        let ref_region = || PcgRegion::from(decomp.args.args()[0].expect_region());
        match &decomp.ty.specifics {
            TySpecifics::MutRef(data) => {
                let r = ref_region();
                let referent = data.referent.decompose_normalize(decomp.args);
                for inner in PrustiPcgCtxt.extract_regions(referent) {
                    if inner != r {
                        self.pairs.insert((r, inner));
                    }
                }
                self.walk(referent);
            }
            TySpecifics::ImmRef(data) => {
                let referent = data.referent.decompose_normalize(decomp.args);
                self.shared.insert(ref_region());
                self.shared.extend(PrustiPcgCtxt.extract_regions(referent));
            }
            TySpecifics::StructLike(data) => {
                for field in &data.fields {
                    self.walk(field.decompose_normalize(decomp.args));
                }
            }
            TySpecifics::EnumLike(data) => {
                for variant in &data.variants {
                    for field in &variant.inner.fields {
                        self.walk(field.decompose_normalize(decomp.args));
                    }
                }
            }
            _ => (),
        }
    }
}

/// TODO: remove! Temporary check of [`IndirectNodesEnc`] until the impure
/// encoder uses it: the bodiless method
/// `indirect_check_<m>(args..) requires <args' direct predicates> && <the
/// predicates of all lifetime projection nodes of the args>`. Silicon checks
/// that its precondition is well-defined, i.e. that every node's key is
/// definable given the nodes before it.
pub struct IndirectCheckEnc;

impl TaskEncoder for IndirectCheckEnc {
    task_encoder::encoder_cache!(IndirectCheckEnc);
    const ENCODER_NAME: &'static str = "indirect check encoder";

    type TaskDescription<'vir> = DefId;
    type OutputFullLocal<'vir> = vir::Method<'vir>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;
        let def_id = *task_key;
        let arg_defs = deps.require_dep::<MirLocalDefEnc>(MirLocalDefEncTask::Local {
            def_id,
            all_locals: false,
        })?;
        let params = GParams::from(def_id);
        let generics = deps.require_dep::<GenericParamsEnc>(params)?;
        vir::with_vcx(|vcx| {
            let tcx = vcx.tcx();
            let sig = tcx.liberate_late_bound_regions(def_id, tcx.fn_sig(def_id).instantiate_identity());
            let arg_count = arg_defs.arg_count;
            let mut pres = Vec::new();
            for i in 1..=sig.inputs().len() {
                pres.push(arg_defs[i.into()].impure_pred);
            }
            for (i, input) in sig.inputs().iter().enumerate() {
                let snap = arg_defs[(i + 1).into()].impure_snap;
                let decomp = RustTyDecomposition::from_ty(*input, params);
                let nodes = deps.require_dep::<IndirectNodesEnc>(decomp)?;
                pres.extend(nodes.ordered().map(|pred| pred.reify(vcx, snap)));
            }
            let args = (0..=arg_count)
                .map(|i| arg_defs[i.into()].local)
                .collect::<Vec<_>>();
            let ident: MethodIdn<'vir, (vir::ManyRef, vir::ManyTyVal, vir::ManyCSnap)> = MethodIdn::new(
                vir::vir_format_identifier!(
                    vcx,
                    "indirect_check_{}",
                    vir::ViperIdent::from_def_id(vcx, def_id)
                ),
                (
                    vcx.alloc_slice(&vec![vir::TYPE_REF; arg_count + 1]),
                    generics.ty_args(),
                    generics.const_args(),
                ),
            );
            let method = vcx.mk_method(
                ident,
                (args.as_slice(), generics.ty_decls(), generics.const_decls()),
                &[],
                vcx.alloc_slice(&pres),
                &[],
                None,
            );
            Ok((method, ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for method in Self::all_outputs_local_no_errors(program) {
            program.add_method(method);
        }
    }
}
