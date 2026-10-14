use itertools::Itertools;
use pcg::borrow_pcg::region_projection::{
    ExtractRegionsCtxt, LifetimeProjection, LifetimeProjectionIdx, PcgRegion, Region,
};
use prusti_rustc_interface::{index::IndexVec, middle::ty};
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};

use super::{
    RustTyDecomposition,
    generics::GArgsTyEnc,
    indirect_pred::{TyIndirectEnc, TyIndirectRef},
    lifetime_query::LifetimeQueryEnc,
    occurrences::{End, Step, occurrence_query, occurrences},
};

pub type RegionIdx = LifetimeProjectionIdx<Region>;

#[derive(Copy, Clone)]
pub struct PrustiPcgCtxt;

impl<'tcx> ExtractRegionsCtxt<'tcx, RustTyDecomposition<'tcx>, PcgRegion<'tcx>> for PrustiPcgCtxt {
    fn extract_regions(
        self,
        data: RustTyDecomposition<'tcx>,
    ) -> IndexVec<LifetimeProjectionIdx<Region>, PcgRegion<'tcx>> {
        data.args
            .args()
            .iter()
            .flat_map(|arg| arg.walk())
            .filter_map(|arg| arg.as_region().map(PcgRegion::from))
            .unique()
            .collect()
    }
}

/// The lifetime projection of `decomp` for the generalized lifetime `idx` of
/// `ty`, `None` for a `RegionsIn(T)` lifetime.
pub fn projection_for_generalized_idx<'tcx>(
    ty: ty::Ty<'tcx>,
    idx: LifetimeProjectionIdx<pcg::borrow_pcg::region_projection::Generalized>,
    decomp: RustTyDecomposition<'tcx>,
    tcx: ty::TyCtxt<'tcx>,
) -> Option<LifetimeProjection<'tcx, RustTyDecomposition<'tcx>, Region>> {
    use pcg::borrow_pcg::GeneralizedLifetime;
    let lifetimes: IndexVec<_, GeneralizedLifetime<'tcx>> = tcx.extract_regions(ty);
    let region = match lifetimes.get(idx)? {
        GeneralizedLifetime::Region(r) => *r,
        GeneralizedLifetime::RegionsIn(_) => return None,
    };
    LifetimeProjection::new(decomp, region, None, PrustiPcgCtxt)
}

/// Encodes, for a type `T` (the type of the base of PCG lifetime projection
/// nodes), the indirect pieces of each node `p↓r`: one per occurrence of the
/// region `r` in `T` (see `occurrences`), `p_T_ind(key, .., Q)` where `Q`
/// selects the occurrence. The key is the snapshot `S` of `p` if no other
/// occurrence encloses it; otherwise it is `S` extended by the pieces of the
/// enclosing occurrences, outer ones first:
/// `ext_T(.. ext_T(S, .., key(o_1), Q_o1) .., key(o_k), Q_ok)`.
pub struct IndirectNodesEnc;

/// The piece of one occurrence of a region.
#[derive(Clone, Debug)]
pub struct NodePiece<'vir> {
    pub region: RegionIdx,
    pub path: Vec<Step>,
    pub query: vir::ExprLifetimeQuery<'vir>,
    /// The pieces enclosing this one.
    parents: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct IndirectNodes<'vir> {
    ind: TyIndirectRef<'vir>,
    tys: &'vir [vir::ExprTyVal<'vir>],
    consts: &'vir [vir::ExprCSnap<'vir>],
    /// The pieces, enclosing ones first. A region without pieces holds
    /// nothing (e.g. one only behind shared references).
    pieces: Vec<NodePiece<'vir>>,
    /// The region indices, enclosing regions first.
    order: Vec<RegionIdx>,
}

impl<'vir> IndirectNodes<'vir> {
    /// Does node `p↓idx` hold anything?
    pub fn holds(&self, idx: RegionIdx) -> bool {
        self.pieces.iter().any(|p| p.region == idx)
    }

    /// The pieces of node `p↓idx`, enclosing ones first.
    pub fn pieces_of(&self, idx: RegionIdx) -> Vec<usize> {
        (0..self.pieces.len())
            .filter(|i| self.pieces[*i].region == idx)
            .collect()
    }

    pub fn piece(&self, piece: usize) -> &NodePiece<'vir> {
        &self.pieces[piece]
    }

    /// The number of pieces.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// The pieces enclosing `piece`.
    pub fn parents(&self, piece: usize) -> &[usize] {
        &self.pieces[piece].parents
    }

    /// The piece of the occurrence at `path`.
    pub fn find(&self, path: &[Step]) -> Option<usize> {
        self.pieces.iter().position(|p| p.path == path)
    }

    /// `ext_T(k, .., kp, q)` in any expression context: the type arguments
    /// and queries are fully reified, so they are valid in any context (as in
    /// `GArgsTy::get_ty`), and reifying the result substitutes their locals.
    fn ext<Curr, Next>(
        &self,
        k: vir::ExprGenSnap<'vir, Curr, Next>,
        kp: vir::ExprGenSnap<'vir, Curr, Next>,
        q: vir::ExprLifetimeQuery<'vir>,
    ) -> vir::ExprGenSnap<'vir, Curr, Next> {
        // SAFETY: see `GArgsTy::get_ty`.
        let (tys, consts, q) = unsafe {
            (
                &*(self.tys as *const [vir::ExprTyVal<'vir>]
                    as *const [vir::ExprGenTyVal<'vir, Curr, Next>]),
                &*(self.consts as *const [vir::ExprCSnap<'vir>]
                    as *const [vir::ExprGenCSnap<'vir, Curr, Next>]),
                &*(q as *const vir::ExprData<'vir, vir::LifetimeQuery>
                    as *const vir::ExprGenData<'vir, Curr, Next, vir::LifetimeQuery>),
            )
        };
        self.ind.ext.call()(k, tys, consts, kp, q)
    }

    /// The key of `piece`, for the snapshot `s` of `p`.
    pub fn key<Curr, Next>(
        &self,
        piece: usize,
        s: vir::ExprGenSnap<'vir, Curr, Next>,
    ) -> vir::ExprGenSnap<'vir, Curr, Next> {
        self.pieces[piece].parents.iter().fold(s, |k, parent| {
            let parent_key = self.key(*parent, s);
            self.ext(k, parent_key, self.pieces[*parent].query)
        })
    }

    /// The predicate application of `piece` with the given key.
    pub fn app(
        &self,
        piece: usize,
        key: vir::ExprSnap<'vir>,
        perm: Option<vir::ExprPerm<'vir>>,
    ) -> vir::PredicateApp<'vir> {
        (self.ind.pred)(key, self.tys, self.consts, self.pieces[piece].query)(perm)
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
        let apps = self.apps(idx, s);
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
        self.pieces_of(idx).into_iter().fold(s, |k, piece| {
            self.ext(k, self.key(piece, s), self.pieces[piece].query)
        })
    }

    /// `s` extended by all pieces: a snapshot whose `&mut` value slots are
    /// filled with the referents. Requires all pieces.
    pub fn extend_all<Curr, Next>(
        &self,
        s: vir::ExprGenSnap<'vir, Curr, Next>,
    ) -> vir::ExprGenSnap<'vir, Curr, Next> {
        (0..self.pieces.len()).fold(s, |k, piece| {
            self.ext(k, self.key(piece, s), self.pieces[piece].query)
        })
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
        let query = deps.require_dep::<LifetimeQueryEnc>(())?;
        let occurrences =
            occurrences(decomp).map_err(|err| EncodeFullError::EncodingError(err, None))?;

        // TODO: `RegionsIn(T)` of a type parameter.
        let kept = (0..occurrences.len())
            .filter(|i| matches!(occurrences[*i].end, End::Region(_)))
            .collect::<Vec<_>>();
        let pieces = kept
            .iter()
            .map(|i| {
                let occurrence = &occurrences[*i];
                let End::Region(region) = occurrence.end else { unreachable!() };
                let region = PcgRegion::from(region);
                NodePiece {
                    region: regions.iter_enumerated().find(|(_, r)| **r == region).unwrap().0,
                    path: occurrence.path.clone(),
                    query: occurrence_query(&query, &occurrence.path, None),
                    parents: occurrence
                        .parents
                        .iter()
                        .filter_map(|p| kept.iter().position(|k| k == p))
                        .collect(),
                }
            })
            .collect::<Vec<_>>();

        // enclosing regions first
        let encloses = |a: RegionIdx, b: RegionIdx| {
            pieces.iter().any(|piece| {
                piece.region == b && piece.parents.iter().any(|p| pieces[*p].region == a)
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
        Ok((
            (),
            IndirectNodes {
                ind,
                tys: args.get_ty(),
                consts: args.get_const(),
                pieces,
                order,
            },
        ))
    }
}
