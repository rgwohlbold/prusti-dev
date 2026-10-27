use itertools::Itertools;
use pcg::borrow_pcg::region_projection::{
    ExtractRegionsCtxt, LifetimeProjection, LifetimeProjectionIdx, PcgRegion, Region,
};
use prusti_rustc_interface::{index::IndexVec, middle::ty};
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};

use super::{
    RustTyDecomposition,
    arity::TyArityEnc,
    generics::{GArgsTy, GArgsTyEnc, GParams, GenericParamsEnc},
    indirect_pred::{TyIndirectEnc, TyIndirectRef},
    occurrences::{End, Pos, Step, occurrences, param_position},
    range::{Range, int, pos_expr},
};

pub type RegionIdx = LifetimeProjectionIdx<Region>;

#[derive(Copy, Clone)]
pub struct PrustiPcgCtxt;

/// The regions of a type as the PCG extracts them (in the same order): a type
/// parameter `T` is the pseudo-region `RegionsIn(T)`.
impl<'tcx> ExtractRegionsCtxt<'tcx, RustTyDecomposition<'tcx>, PcgRegion<'tcx>> for PrustiPcgCtxt {
    fn extract_regions(
        self,
        data: RustTyDecomposition<'tcx>,
    ) -> IndexVec<LifetimeProjectionIdx<Region>, PcgRegion<'tcx>> {
        data.args
            .args()
            .iter()
            .flat_map(|arg| arg.walk())
            .filter_map(|arg| match arg.as_region() {
                Some(region) => Some(PcgRegion::from(region)),
                None => match arg.as_type()?.kind() {
                    ty::TyKind::Param(param) => Some(PcgRegion::RegionsIn(*param)),
                    _ => None,
                },
            })
            .unique()
            .collect()
    }
}

/// The node of `decomp` for the generalized lifetime `idx` of `ty`: a region,
/// or `RegionsIn(T)` of a type parameter (`None` for other opaque types).
pub fn node_for_generalized_idx<'tcx>(
    ty: ty::Ty<'tcx>,
    idx: LifetimeProjectionIdx<pcg::borrow_pcg::region_projection::Generalized>,
    decomp: RustTyDecomposition<'tcx>,
    tcx: ty::TyCtxt<'tcx>,
) -> Option<RegionIdx> {
    use pcg::borrow_pcg::GeneralizedLifetime;
    let lifetimes: IndexVec<_, GeneralizedLifetime<'tcx>> = tcx.extract_regions(ty);
    let region = match lifetimes.get(idx)? {
        GeneralizedLifetime::Region(r) => *r,
        GeneralizedLifetime::RegionsIn(opaque) => match opaque.ty(tcx).kind() {
            ty::TyKind::Param(param) => PcgRegion::RegionsIn(*param),
            // TODO: `RegionsIn` of an associated type
            _ => return None,
        },
    };
    LifetimeProjection::new(decomp, region, None, PrustiPcgCtxt).map(|lp| lp.region_idx())
}

/// Encodes, for a type `T` (the type of the base of PCG lifetime projection
/// nodes), the indirect pieces of each node `p↓r`: one per occurrence of the
/// region `r` in `T` (see `occurrences`), `p_T_ind(key, .., lo, hi)` for the
/// positions `lo .. hi` of the occurrence. The key of the piece at position
/// `lo` is the snapshot `S` of `p` extended by the pieces of the positions
/// `0 .. lo` in order: `K_0 = S`, `K_i+1 = ext_T(K_i, .., K_i, r_i)`.
pub struct IndirectNodesEnc;

/// The piece of one occurrence of a region (one position), or of a type
/// parameter (its `arity` positions).
#[derive(Clone, Debug)]
pub struct NodePiece {
    /// The PCG node `p↓node` the piece belongs to: a region, or the
    /// pseudo-region `RegionsIn(T)` of a type parameter.
    pub node: RegionIdx,
    /// Is it the piece of a type parameter (`RegionsIn(T)`)?
    pub all: bool,
    pub path: Vec<Step>,
    pos: Pos,
    /// The type parameter whose positions the piece covers (`all`).
    width: Option<ty::ParamTy>,
    /// The pieces enclosing this one.
    parents: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct IndirectNodes<'vir> {
    ind: TyIndirectRef<'vir>,
    tys: &'vir [vir::ExprTyVal<'vir>],
    consts: &'vir [vir::ExprCSnap<'vir>],
    /// The context of the type's arguments, and the type values of its type
    /// parameters (for the positions).
    context: GParams<'vir>,
    context_tys: &'vir [vir::ExprTyVal<'vir>],
    /// The pieces, enclosing ones first. A region without pieces holds
    /// nothing (e.g. one only behind shared references).
    pieces: Vec<NodePiece>,
    /// The region indices, enclosing regions first.
    order: Vec<RegionIdx>,
}

impl<'vir> IndirectNodes<'vir> {
    /// Does node `p↓idx` hold anything?
    pub fn holds(&self, idx: RegionIdx) -> bool {
        self.pieces.iter().any(|p| p.node == idx)
    }

    /// Is `p↓idx` the node `RegionsIn(T)` of a type parameter?
    pub fn is_regions_in(&self, idx: RegionIdx) -> bool {
        self.pieces.iter().any(|p| p.node == idx && p.all)
    }

    /// The pieces of node `p↓idx`, enclosing ones first.
    pub fn pieces_of(&self, idx: RegionIdx) -> Vec<usize> {
        (0..self.pieces.len())
            .filter(|i| self.pieces[*i].node == idx)
            .collect()
    }

    pub fn piece(&self, piece: usize) -> &NodePiece {
        &self.pieces[piece]
    }

    /// The number of pieces.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// The same pieces for other (Viper) type arguments of the type: e.g. a
    /// callee's view of its argument, instantiated with a caller's types
    /// (`context` the callee's type parameters instantiated).
    pub fn with_args(&self, args: GArgsTy<'vir>, context: GArgsTy<'vir>) -> Self {
        Self {
            tys: args.get_ty(),
            consts: args.get_const(),
            context_tys: context.get_ty(),
            ..self.clone()
        }
    }

    /// The positions of `piece`.
    pub fn range(&self, piece: usize) -> Range<'vir> {
        let piece = &self.pieces[piece];
        let ty_of = |param: ty::ParamTy| {
            self.context_tys[param_position(self.context, param.index, |arg| arg.as_type().is_some())]
        };
        let lo = pos_expr(&piece.pos, ty_of);
        let width = match piece.width {
            Some(param) => TyArityEnc::arity(ty_of(param)),
            None => int(1),
        };
        Range::starting_at(lo, width)
    }

    /// The type arguments of the type.
    pub fn args(&self) -> GArgsTy<'vir> {
        GArgsTy::new(self.tys, self.consts)
    }

    /// The pieces enclosing `piece`.
    pub fn parents(&self, piece: usize) -> &[usize] {
        &self.pieces[piece].parents
    }

    /// The piece of the occurrence at `path`.
    pub fn find(&self, path: &[Step]) -> Option<usize> {
        self.pieces.iter().position(|p| p.path == path)
    }

    /// `ext_T(k, .., kp, range)` in any expression context: the type
    /// arguments and the range are fully reified, so they are valid in any
    /// context (as in `GArgsTy::get_ty`), and reifying the result substitutes
    /// their locals.
    pub fn ext<Curr, Next>(
        &self,
        k: vir::ExprGenSnap<'vir, Curr, Next>,
        kp: vir::ExprGenSnap<'vir, Curr, Next>,
        range: Range<'vir>,
    ) -> vir::ExprGenSnap<'vir, Curr, Next> {
        let (tys, consts) = (self.args().get_ty(), self.args().get_const());
        let (lo, hi) = range.args();
        self.ind.ext.call()(k, tys, consts, kp, lo, hi)
    }

    /// The key of `piece`, for the snapshot `s` of `p`: `s` extended by the
    /// pieces of all lower positions (a chain key).
    pub fn key<Curr, Next>(
        &self,
        piece: usize,
        s: vir::ExprGenSnap<'vir, Curr, Next>,
    ) -> vir::ExprGenSnap<'vir, Curr, Next> {
        (0..piece).fold(s, |k, lower| self.ext(k, k, self.range(lower)))
    }

    /// The predicate application of `piece` with the given key.
    pub fn app(
        &self,
        piece: usize,
        key: vir::ExprSnap<'vir>,
        perm: Option<vir::ExprPerm<'vir>>,
    ) -> vir::PredicateApp<'vir> {
        self.app_range(self.range(piece), key, perm)
    }

    /// The predicate application of the positions `range` with the given key.
    pub fn app_range(
        &self,
        range: Range<'vir>,
        key: vir::ExprSnap<'vir>,
        perm: Option<vir::ExprPerm<'vir>>,
    ) -> vir::PredicateApp<'vir> {
        (self.ind.pred)(key, self.tys, self.consts, range.lo, range.hi)(perm)
    }

    /// The predicate applications of node `p↓idx`, enclosing ones first.
    pub fn apps(&self, idx: RegionIdx, s: vir::ExprSnap<'vir>) -> Vec<vir::PredicateApp<'vir>> {
        self.pieces_of(idx)
            .into_iter()
            .map(|piece| self.app(piece, self.key(piece, s), None))
            .collect()
    }

    /// The predicate of node `p↓idx`, `None` if it holds nothing.
    pub fn pred(&self, idx: RegionIdx, s: vir::ExprSnap<'vir>) -> Option<vir::ExprBool<'vir>> {
        self.pred_with(idx, s, |_| false)
    }

    /// As [Self::pred], with the key steps over the pieces of the nodes in
    /// `at_old` evaluated in the pre-state (`old`), e.g. those a result took.
    pub fn pred_with(
        &self,
        idx: RegionIdx,
        s: vir::ExprSnap<'vir>,
        at_old: impl Fn(RegionIdx) -> bool,
    ) -> Option<vir::ExprBool<'vir>> {
        let apps = self
            .pieces_of(idx)
            .into_iter()
            .map(|piece| {
                let key = (0..piece).fold(s, |k, lower| {
                    let ext = self.ext(k, k, self.range(lower));
                    match at_old(self.pieces[lower].node) {
                        true => vir::with_vcx(|vcx| vcx.mk_old_expr(ext)),
                        false => ext,
                    }
                });
                self.app(piece, key, None)
            })
            .collect::<Vec<_>>();
        vir::with_vcx(|vcx| {
            (!apps.is_empty()).then(|| {
                vcx.mk_conj(
                    &apps
                        .into_iter()
                        .map(|app| vcx.mk_predicate_app_expr(app))
                        .collect::<Vec<_>>(),
                )
            })
        })
    }

    /// The region indices, enclosing regions first (so that the keys of later
    /// nodes are well-defined given the earlier ones).
    pub fn order(&self) -> &[RegionIdx] {
        &self.order
    }

    /// `s` extended by the pieces of node `p↓idx` only (e.g. the own piece of
    /// a reference, to read its referent). Requires that node's pieces.
    pub fn extend_with<Curr, Next>(
        &self,
        idx: RegionIdx,
        s: vir::ExprGenSnap<'vir, Curr, Next>,
    ) -> vir::ExprGenSnap<'vir, Curr, Next> {
        self.pieces_of(idx)
            .into_iter()
            .fold(s, |k, piece| self.ext(k, self.key(piece, s), self.range(piece)))
    }

    /// `s` extended by all pieces: a snapshot whose `&mut` value slots are
    /// filled with the referents. Requires all pieces.
    pub fn extend_all<Curr, Next>(
        &self,
        s: vir::ExprGenSnap<'vir, Curr, Next>,
    ) -> vir::ExprGenSnap<'vir, Curr, Next> {
        (0..self.pieces.len()).fold(s, |k, piece| self.ext(k, self.key(piece, s), self.range(piece)))
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
        let regions = PrustiPcgCtxt.extract_regions(decomp);
        let occurrences =
            occurrences(decomp).map_err(|err| EncodeFullError::EncodingError(err, None))?;

        let pieces = occurrences
            .iter()
            .map(|occurrence| {
                let (region, all) = match occurrence.end {
                    End::Region(region) => (PcgRegion::from(region), false),
                    End::Param(param) => (PcgRegion::RegionsIn(param), true),
                };
                let node = regions.iter_enumerated().find(|(_, r)| **r == region).unwrap().0;
                NodePiece {
                    node,
                    all,
                    path: occurrence.path.clone(),
                    pos: occurrence.pos.clone(),
                    width: occurrence.width_param(),
                    parents: occurrence.parents.clone(),
                }
            })
            .collect::<Vec<_>>();

        // enclosing regions first
        let encloses = |a: RegionIdx, b: RegionIdx| {
            pieces.iter().any(|piece| {
                piece.node == b && piece.parents.iter().any(|p| pieces[*p].node == a)
            })
        };
        let mut order = Vec::new();
        while order.len() < regions.len() {
            let next = regions.indices().find(|idx| {
                !order.contains(idx)
                    && regions
                        .indices()
                        .all(|p| p == *idx || order.contains(&p) || !encloses(p, *idx))
            });
            let Some(next) = next else {
                return Err(EncodeFullError::EncodingError(
                    format!("regions enclosing each other are not supported ({decomp:?})"),
                    None,
                ));
            };
            order.push(next);
        }

        let ind = deps.require_ref::<TyIndirectEnc>(decomp.ty)?;
        let args = deps.require_dep::<GArgsTyEnc>(decomp.args)?;
        let context = decomp.args.context();
        let context_params = deps.require_dep::<GenericParamsEnc>(context)?;
        let context_tys = vir::with_vcx(|vcx| vcx.alloc_slice(context_params.ty_exprs()));
        Ok((
            (),
            IndirectNodes {
                ind,
                tys: args.get_ty(),
                consts: args.get_const(),
                context,
                context_tys,
                pieces,
                order,
            },
        ))
    }
}
