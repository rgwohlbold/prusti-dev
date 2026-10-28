//! Indirect pieces of nested lifetimes in method bodies: opening them at PCG
//! lifetime-projection expansions and closing them when the expanded place is
//! folded again.
//!
//! - An expansion `{B↓r} -> {C↓r, ..}` (`B`'s base label is the state before
//!   the current phase) unfolds the piece `B↓r` with its key at that label;
//!   children at generic positions are cast to their concrete pieces
//!   (`make_concrete_ind`).
//! - When the place `B` is folded again (the deref of a reference removed, the
//!   fields of a struct collapsed), the PCG makes `B↓r` current again: the
//!   children are cast back right before the place fold (`make_generic_ind`,
//!   while they are still concrete) and `B↓r` is folded right after it, with
//!   its current key.
//! - The pieces of a value built from parts (`t = C { f: op }`, PCG
//!   `Aggregate` edges) are folded at the end of the phase.
//!
//! The nodes `p↓RegionsIn(T)` of type parameters are handled the same way.
//! Their parts are guarded by their ranges being nonempty: at an expansion, a
//! child's piece is made empty (`ind_empty` or a fold) if `arity(T)` is 0.

use pcg::{
    borrow_pcg::{
        borrow_pcg_expansion::BorrowPcgExpansion,
        edge::{borrow_flow::BorrowFlowEdgeKind, kind::BorrowPcgEdgeKind},
        state::BorrowsState,
    },
    pcg::{Pcg, PcgNode},
    utils::{HasPlace, Place, PrefixRelation, maybe_old::MaybeLabelledPlace},
};
use prusti_rustc_interface::{
    index::Idx,
    middle::{mir, ty},
};
use task_encoder::TaskEncoder;
use vir::CastType;

use crate::encoders::{
    EncodeResult, ImpureEncVisitor,
    mir_shared::PureRvalueEnc,
    ty::{
        LazyRustTy, RustTyDecomposition, RustTySpecial, TySpecifics,
        generics::{
            GArgsTy, GArgsTyEnc,
            ind_casters::{IndCasters, IndCastersEnc},
        },
        indirect_node::{IndirectNodesEnc, RegionIdx},
        indirect_pred::TyIndirectEnc,
        occurrences::Step,
        range::Range,
    },
};

/// A `RegionsIn(T)` node unfolded at `expansion`, with the keys its pieces
/// were unfolded with (the children's pieces are made from them).
#[derive(Clone)]
pub struct OpenedRegionsIn<'vir> {
    expansion: BorrowPcgExpansion<'vir>,
    keys: Vec<(usize, vir::ExprSnap<'vir>)>,
}

/// The bookkeeping of open nested pieces along one path: it is saved at the
/// end of a block and restored for its successors (a successor sees the state
/// of a predecessor, not of the block encoded last).
#[derive(Clone, Default)]
pub struct NestedPathState<'vir> {
    opened_regions_in: Vec<OpenedRegionsIn<'vir>>,
    applied_wands: Vec<mir::BasicBlock>,
}

/// The cast of the pieces of a part `source↓idx` of a value built from parts
/// to the generic field position they flow into.
#[derive(Clone)]
pub struct AggregateCast<'vir> {
    source: MaybeLabelledPlace<'vir>,
    source_idx: RegionIdx,
    casters: IndCasters<'vir>,
    ty_args: GArgsTy<'vir>,
}

/// The cast between the generic piece a child has inside its parent and its
/// concrete piece (one per piece of the child's node).
#[derive(Clone)]
struct ChildCast<'vir> {
    child: Place<'vir>,
    /// The child's piece in its own type.
    piece: usize,
    casters: IndCasters<'vir>,
    ty_args: GArgsTy<'vir>,
    range: Range<'vir>,
}

/// An expanded lifetime projection's place and region, and its children's.
type LpExpansion<'vir> = (Place<'vir>, RegionIdx, Vec<(Place<'vir>, RegionIdx)>);

impl<'vir, E: TaskEncoder> ImpureEncVisitor<'vir, '_, E> {
    /// At the start of `block`: the state at the end of an encoded
    /// predecessor (empty at the entry).
    pub(crate) fn nested_enter_block(&mut self, block: mir::BasicBlock) {
        let state = self.body.basic_blocks.predecessors()[block]
            .iter()
            .find_map(|pred| self.nested_at_exit.get(pred))
            .cloned()
            .unwrap_or_default();
        self.opened_regions_in = state.opened_regions_in;
        self.applied_wands = state.applied_wands;
    }

    /// At the end of `block`: save the state for its successors.
    pub(crate) fn nested_exit_block(&mut self, block: mir::BasicBlock) {
        let state = NestedPathState {
            opened_regions_in: self.opened_regions_in.clone(),
            applied_wands: self.applied_wands.clone(),
        };
        self.nested_at_exit.insert(block, state);
    }

    /// The predicate applications of the pieces of node `place↓idx`
    /// (enclosing ones first), their keys evaluated at `key_label` (or now).
    pub(crate) fn node_pred_apps(
        &mut self,
        place: Place<'vir>,
        idx: RegionIdx,
        key_label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, Vec<vir::PredicateApp<'vir>>, E> {
        let place = strip_downcasts(place);
        let ty = place.ty(self.pcg_ctxt()).ty;
        let decomp = RustTyDecomposition::from_ty(ty, self.def_id);
        let nodes = self.deps.require_dep::<IndirectNodesEnc>(decomp)?;
        if !nodes.holds(idx) {
            return Ok(Vec::new());
        }
        let snap = match key_label {
            Some(_) => self.path_snap(place)?,
            None => self.encode_place_with_snap(place)?.1,
        };
        Ok(nodes
            .pieces_of(idx)
            .into_iter()
            .map(|piece| {
                let key = self
                    .vcx
                    .maybe_apply_label(nodes.key(piece, snap), key_label);
                nodes.app(piece, key, None)
            })
            .collect())
    }

    /// The snapshot of `place`, read along its path from the local: through a
    /// `&mut`, from the reference's snapshot extended by its own piece. Unlike
    /// the place's own predicate (which an expansion may have unfolded or a
    /// deref may not have produced yet), this only needs the own pieces of the
    /// references on the path, as at the start of a phase.
    pub(crate) fn path_snap(
        &mut self,
        place: Place<'vir>,
    ) -> EncodeResult<'vir, vir::ExprSnap<'vir>, E> {
        let Some((parent, elem)) = place.last_projection() else {
            return Ok(self.encode_place_with_snap(place)?.1);
        };
        let parent_ty = parent.ty(self.pcg_ctxt());
        Ok(match elem {
            mir::ProjectionElem::Deref => {
                match parent_ty.ty.kind() {
                    ty::TyKind::Ref(.., ty::Mutability::Mut) => {
                        let snap = self.path_snap(parent)?;
                        let nodes = self.deps.require_dep::<IndirectNodesEnc>(
                            RustTyDecomposition::from_ty(parent_ty.ty, self.def_id),
                        )?;
                        // by the reference's own piece (position 0)
                        let extended = nodes.extend(1, snap);
                        let e_ty = self.ty_use_pure(parent_ty.ty);
                        e_ty.expect_mutref().value_access(extended.downcast_ty())
                    }
                    ty::TyKind::Ref(.., ty::Mutability::Not) => {
                        let snap = self.path_snap(parent)?;
                        let e_ty = self.ty_use_pure(parent_ty.ty);
                        e_ty.expect_immref().value_access(snap.downcast_ty())
                    }
                    _ if parent_ty.ty.is_box() => {
                        let snap = self.path_snap(parent)?;
                        let e_ty = self.ty_use_pure(parent_ty.ty);
                        e_ty.expect_structlike()
                            .box_value_access(snap.downcast_ty())
                    }
                    _ => self.encode_place_with_snap(place)?.1,
                }
            }
            mir::ProjectionElem::Field(field, _) => {
                let snap = self.path_snap(parent)?;
                let e_ty = self.ty_use_pure(parent_ty.ty);
                e_ty.expect_variant_opt(parent_ty.variant_index)[field].read(snap.downcast_ty())
            }
            mir::ProjectionElem::Downcast(..) => self.path_snap(parent)?,
            _ => self.encode_place_with_snap(place)?.1,
        })
    }

    /// The label of the state before the actions of the current phase.
    fn current_phase_label(&self) -> vir::OldLabel<'vir> {
        let (location, phase) = self.current_phase.unwrap();
        vir::OldLabel::Label(self.phase_label(
            location,
            phase,
            self.current_block_pres.as_ref().unwrap(),
        ))
    }

    /// The lifetime projection expanded by `expansion` and its children, if it
    /// opens a nested piece (not for a downcast, which views the same value).
    fn lp_expansion(expansion: &BorrowPcgExpansion<'vir>) -> Option<LpExpansion<'vir>> {
        let PcgNode::LifetimeProjection(base) = expansion.base() else {
            return None;
        };
        let MaybeLabelledPlace::Current(base_place) = base.base() else {
            return None;
        };
        let children = expansion
            .expansion()
            .into_iter()
            .filter_map(|child| match child {
                PcgNode::LifetimeProjection(lp) => match lp.base() {
                    MaybeLabelledPlace::Current(place) => Some((place, lp.region_idx())),
                    MaybeLabelledPlace::Labelled(_) => None,
                },
                PcgNode::Place(_) => None,
            })
            .collect::<Vec<_>>();
        if children.iter().any(|(child, _)| {
            matches!(
                child.last_projection(),
                Some((_, mir::ProjectionElem::Downcast(..)))
            )
        }) {
            return None;
        }
        Some((base_place, base.region_idx(), children))
    }

    /// As [Self::lp_expansion], but also for labelled places (an open
    /// expansion whose places were labelled since, e.g. at `StorageDead`).
    fn lp_expansion_any(expansion: &BorrowPcgExpansion<'vir>) -> Option<LpExpansion<'vir>> {
        let PcgNode::LifetimeProjection(base) = expansion.base() else {
            return None;
        };
        let children = expansion
            .expansion()
            .into_iter()
            .filter_map(|child| match child {
                PcgNode::LifetimeProjection(lp) => Some((lp.base().place(), lp.region_idx())),
                PcgNode::Place(_) => None,
            })
            .collect::<Vec<_>>();
        if children.iter().any(|(child, _)| {
            matches!(
                child.last_projection(),
                Some((_, mir::ProjectionElem::Downcast(..)))
            )
        }) {
            return None;
        }
        Some((base.base().place(), base.region_idx(), children))
    }

    /// The place whose piece `expansion` opens, without downcasts (a variant
    /// is the same value).
    pub(crate) fn lp_expansion_base(expansion: &BorrowPcgExpansion<'vir>) -> Option<Place<'vir>> {
        Self::lp_expansion(expansion).map(|(base, ..)| strip_downcasts(base))
    }

    /// The expansions of lifetime projections of `place`, enclosed lifetimes
    /// first (the reverse of `IndirectNodes::order`).
    pub(crate) fn innermost_first(
        &mut self,
        place: Place<'vir>,
        mut expansions: Vec<BorrowPcgExpansion<'vir>>,
    ) -> EncodeResult<'vir, Vec<BorrowPcgExpansion<'vir>>, E> {
        if expansions.len() < 2 {
            return Ok(expansions);
        }
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let order = nodes.order().to_vec();
        expansions.sort_by_key(|e| {
            let region = Self::lp_expansion(e).map(|(_, region, _)| region);
            std::cmp::Reverse(order.iter().position(|idx| Some(*idx) == region))
        });
        Ok(expansions)
    }

    /// Unfold the pieces opened by `expansion` (enclosed ones first, their
    /// keys read the enclosing ones), keys evaluated at `key_label` (or now).
    pub(crate) fn nested_expansion_unfold(
        &mut self,
        expansion: &BorrowPcgExpansion<'vir>,
        key_label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        let Some((base, region, _)) = Self::lp_expansion(expansion) else {
            return Ok(());
        };
        let base = strip_downcasts(base);
        let ty = base.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        if nodes.is_regions_in(region) {
            // the children's pieces are made from the keys: bind them
            let snap = match key_label {
                Some(_) => self.path_snap(base)?,
                None => self.encode_place_with_snap(base)?.1,
            };
            let keys = nodes
                .pieces_of(region)
                .into_iter()
                .map(|piece| {
                    (
                        piece,
                        self.bind(
                            self.vcx
                                .maybe_apply_label(nodes.key(piece, snap), key_label),
                        ),
                    )
                })
                .collect::<Vec<_>>();
            for (piece, key) in keys.iter().rev() {
                self.stmt(self.vcx.mk_unfold_stmt(nodes.app(*piece, key, None)));
            }
            self.opened_regions_in.push(OpenedRegionsIn {
                expansion: expansion.clone(),
                keys,
            });
            return Ok(());
        }
        for piece in self
            .node_pred_apps(base, region, key_label)?
            .into_iter()
            .rev()
        {
            self.stmt(self.vcx.mk_unfold_stmt(piece));
        }
        Ok(())
    }

    /// An expansion of a lifetime projection was added: unfold its piece
    /// (unless done before its place was unfolded, see
    /// `ImpureEncVisitor::pcg_actions`), with the key at the state before the
    /// current phase, and cast the children at generic positions.
    pub(crate) fn nested_expansion_added(
        &mut self,
        expansion: &BorrowPcgExpansion<'vir>,
        unfolded: bool,
    ) -> EncodeResult<'vir, (), E> {
        let Some((base_place, _, children)) = Self::lp_expansion(expansion) else {
            return Ok(());
        };
        if !unfolded {
            let label = self.current_phase_label();
            self.nested_expansion_unfold(expansion, Some(label))?;
        }
        if let Some(i) = self
            .opened_regions_in
            .iter()
            .position(|o| o.expansion == *expansion)
        {
            let opened = self.opened_regions_in.remove(i);
            self.regions_in_empty_children(base_place, &children, &opened.keys)?;
        }
        let mut casts = Vec::new();
        for (child, child_idx) in children {
            casts.extend(self.child_casts(base_place, child, child_idx)?);
        }
        // enclosing pieces first: the concrete keys of the enclosed ones read
        // them
        casts.sort_by_key(|cast| cast.piece);
        for cast in &casts {
            let key = self.child_key(cast)?;
            let generic = (cast.casters.make_generic_s)(
                key.downcast_ty(),
                cast.ty_args.get_ty(),
                cast.ty_args.get_const(),
            );
            self.stmt(
                (cast.casters.make_concrete)(
                    generic.upcast_ty(),
                    cast.ty_args.get_ty(),
                    cast.ty_args.get_const(),
                    cast.range.lo,
                    cast.range.hi,
                )
                .alloc(),
            );
        }
        Ok(())
    }

    /// The places of by-value arguments that are expanded in the owned PCG
    /// of `pcg` (deepest first), with their children.
    pub(crate) fn owed_expansions(
        &self,
        pcg: &Pcg<'_, 'vir>,
    ) -> Vec<(Place<'vir>, Vec<Place<'vir>>)> {
        use pcg::pcg::CapabilityKind;
        let with_capability = [
            CapabilityKind::Read,
            CapabilityKind::Write,
            CapabilityKind::Exclusive,
            CapabilityKind::ShallowExclusive,
        ]
        .into_iter()
        .flat_map(|cap| pcg.places_with_capapability(cap))
        .filter(|place| self.is_by_value_arg(place.local))
        .collect::<Vec<_>>();
        // proper prefixes of places with a capability, not through a dereference
        let mut expanded: Vec<Place<'vir>> = Vec::new();
        for leaf in &with_capability {
            let mut place = *leaf;
            while let Some((parent, _)) = place.last_projection() {
                if !expanded.contains(&parent) {
                    expanded.push(parent);
                }
                place = parent;
            }
        }
        expanded.retain(|place| {
            !place
                .projection()
                .iter()
                .any(|e| matches!(e, mir::ProjectionElem::Deref))
        });
        expanded.sort_by_key(|place| std::cmp::Reverse(place.projection().len()));
        expanded
            .iter()
            .map(|base| {
                let children = with_capability
                    .iter()
                    .chain(expanded.iter())
                    .filter(|c| {
                        c.last_projection()
                            .is_some_and(|(parent, _)| parent == *base)
                    })
                    .copied()
                    .collect();
                (*base, children)
            })
            .collect()
    }

    /// Fold all owed pieces of the expanded places `expansions`.
    fn fold_owed_expansions(
        &mut self,
        expansions: Vec<(Place<'vir>, Vec<Place<'vir>>)>,
    ) -> EncodeResult<'vir, (), E> {
        for (base, children) in expansions {
            let regions = base.regions(self.pcg_ctxt());
            for (idx, region) in regions.iter_enumerated() {
                let children = children
                    .iter()
                    .filter_map(|c| {
                        let i = c
                            .regions(self.pcg_ctxt())
                            .iter()
                            .position(|r| r == region)?;
                        Some((*c, RegionIdx::from(i)))
                    })
                    .collect::<Vec<_>>();
                self.fold_owed(base, idx, &children)?;
            }
        }
        Ok(())
    }

    /// At `return`: fold the pieces of the by-value arguments that are still
    /// expanded (the PCG does not collapse them at the end, and may have
    /// removed their lifetime projection expansions as dead).
    pub(crate) fn nested_fold_at_return(
        &mut self,
        pcg: &Pcg<'_, 'vir>,
    ) -> EncodeResult<'vir, (), E> {
        let owed = self.owed_expansions(pcg);
        self.fold_owed_expansions(owed)
    }

    /// A by-value argument (or a part of it) is collapsed for a drop: its
    /// pieces are still owed to the caller, so fold those of its expanded
    /// places (in the state before the phase) now, as at `return`.
    pub(crate) fn nested_drop(&mut self, place: Place<'vir>) -> EncodeResult<'vir, (), E> {
        let (owed, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.owed_expansions)
            .into_iter()
            .partition(|(base, _)| place.is_prefix_of(*base));
        self.owed_expansions = rest;
        self.fold_owed_expansions(owed)
    }

    fn is_by_value_arg(&self, local: mir::Local) -> bool {
        self.body.local_kind(local) == mir::LocalKind::Arg
            && !self.body.local_decls[local].ty.is_ref()
    }

    /// Fold the pieces of `base↓region` of a by-value argument that the
    /// postcondition takes, keyed by the pre-state snapshot: of a lifetime of
    /// the argument itself (`base` the argument), or of a `RegionsIn(T)` node
    /// the postcondition gives back unblocked (casting `children` at generic
    /// positions back first).
    fn fold_owed(
        &mut self,
        base: Place<'vir>,
        region: RegionIdx,
        children: &[(Place<'vir>, RegionIdx)],
    ) -> EncodeResult<'vir, (), E> {
        let place = strip_downcasts(base);
        if place
            .projection()
            .iter()
            .any(|e| matches!(e, mir::ProjectionElem::Deref))
        {
            return Ok(());
        }
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        if !nodes.is_regions_in(region) {
            if place.projection().is_empty() {
                let snap = self.vcx.mk_old_expr(self.encode_place_with_snap(place)?.1);
                for piece in nodes.apps(region, snap) {
                    self.stmt(self.vcx.mk_fold_stmt(piece));
                }
            }
            return Ok(());
        }
        let pcg_region = place.regions(self.pcg_ctxt())[region];
        let unblocked = self
            .wands
            .unblocked_regions_in(self.vcx, self.deps)
            .into_iter()
            .any(|(local, idx)| {
                local == place.local
                    && Place::from(local).regions(self.pcg_ctxt())[idx] == pcg_region
            });
        if !unblocked {
            return Ok(());
        }
        let mut casts = Vec::new();
        for (child, child_idx) in children {
            casts.extend(self.child_casts(base, *child, *child_idx)?);
        }
        casts.sort_by_key(|cast| std::cmp::Reverse(cast.piece));
        for cast in casts {
            let key = self.child_key(&cast)?;
            self.stmt(
                (cast.casters.make_generic)(
                    key,
                    cast.ty_args.get_ty(),
                    cast.ty_args.get_const(),
                    cast.range.lo,
                    cast.range.hi,
                )
                .alloc(),
            );
        }
        let snap = self.vcx.mk_old_expr(self.path_snap(place)?);
        for piece in nodes.apps(region, snap) {
            self.stmt(self.vcx.mk_fold_stmt(piece));
        }
        Ok(())
    }

    /// After unfolding the `RegionsIn(T_j)` pieces (with `keys`) of `base` at
    /// an expansion into `children`: a part is guarded by its range being
    /// nonempty, so for an empty range (`arity(T_j)` is 0), make the child's
    /// piece empty: `ind_empty` for a field of type `T_j` (`p_Param_ind`,
    /// before any cast), a fold for a constructor around `T_j`.
    fn regions_in_empty_children(
        &mut self,
        base: Place<'vir>,
        children: &[(Place<'vir>, RegionIdx)],
        keys: &[(usize, vir::ExprSnap<'vir>)],
    ) -> EncodeResult<'vir, (), E> {
        let vid = base.ty(self.pcg_ctxt()).variant_index;
        let base = strip_downcasts(base);
        let decomp = RustTyDecomposition::from_ty(base.ty(self.pcg_ctxt()).ty, self.def_id);
        let nodes = self.deps.require_dep::<IndirectNodesEnc>(decomp)?;
        let dep = self.deps.require_dep::<TyIndirectEnc>(decomp.ty)?;
        let args = self.deps.require_dep::<GArgsTyEnc>(decomp.args)?;
        let param_ind = self
            .deps
            .require_ref::<TyIndirectEnc>(RustTyDecomposition::param())?;
        for (g, key) in keys {
            let Some((Step::In(j), rest)) = nodes.piece(*g).path.split_first() else {
                continue;
            };
            for part in dep.ty_param_parts[*j].iter().filter(|p| p.vid == vid) {
                let Some(child) = children
                    .iter()
                    .map(|(c, _)| *c)
                    .find(|c| child_field(decomp, *c) == Some(part.field_idx))
                else {
                    continue;
                };
                let child_decomp =
                    RustTyDecomposition::from_ty(child.ty(self.pcg_ctxt()).ty, self.def_id);
                let child_nodes = self.deps.require_dep::<IndirectNodesEnc>(child_decomp)?;
                let path = part.path.iter().chain(rest).copied().collect::<Vec<_>>();
                let Some(piece) = child_nodes.find(&path) else {
                    continue;
                };
                let k = part.read.read(key);
                let tau = args.get_ty()[*j];
                let range = child_nodes.range(piece);
                let nonempty = range.nonempty();
                let is_empty = vir::expr! { !([nonempty]) };
                let empty = if part.path.is_empty() {
                    (param_ind.param.unwrap().empty)(k, &[tau], &[], range.lo, range.hi).alloc()
                } else {
                    self.vcx.mk_fold_stmt(child_nodes.app(piece, k, None))
                };
                self.stmt(
                    self.vcx
                        .mk_if_stmt(is_empty, self.vcx.alloc_slice(&[empty]), &[]),
                );
            }
        }
        Ok(())
    }

    /// The casts of the pieces of the child `child↓idx` of `base` if it sits
    /// at a generic position of `base`'s type constructor (and is concrete).
    fn child_casts(
        &mut self,
        base: Place<'vir>,
        child: Place<'vir>,
        idx: RegionIdx,
    ) -> EncodeResult<'vir, Vec<ChildCast<'vir>>, E> {
        let Some((parent, elem)) = child.last_projection() else {
            return Ok(Vec::new());
        };
        if parent != base {
            return Ok(Vec::new());
        }
        let base_ty = base.ty(self.pcg_ctxt());
        let decomp = RustTyDecomposition::from_ty(base_ty.ty, self.def_id);
        let field: Option<LazyRustTy<'vir>> = match (&decomp.ty.specifics, elem) {
            (TySpecifics::MutRef(data), mir::ProjectionElem::Deref) => Some(data.referent),
            (TySpecifics::StructLike(data), mir::ProjectionElem::Deref)
                if decomp.ty.special == RustTySpecial::Box =>
            {
                data.fields.last().map(|f| f.ty())
            }
            (TySpecifics::StructLike(data), mir::ProjectionElem::Field(f, _)) => {
                data.fields.get(f.as_usize()).map(|f| f.ty())
            }
            (TySpecifics::EnumLike(data), mir::ProjectionElem::Field(f, _)) => base_ty
                .variant_index
                .and_then(|v| data.variants.iter().find(|var| var.data.vid == v))
                .and_then(|var| var.inner.fields.get(f.as_usize()))
                .map(|f| f.ty()),
            _ => None,
        };
        let Some(field) = field else {
            return Ok(Vec::new());
        };
        let Some(normalized) = field.decompose_compare_normalize(decomp.ty.params, decomp.args)
        else {
            return Ok(Vec::new());
        };
        let child_ty = child.ty(self.pcg_ctxt()).ty;
        let child_decomp = RustTyDecomposition::from_ty(child_ty, self.def_id);
        if child_decomp.ty.specifics.is_param() {
            // `p_Param_ind` in the parent as well
            return Ok(Vec::new());
        }
        let child_nodes = self.deps.require_dep::<IndirectNodesEnc>(child_decomp)?;
        let casters = self
            .deps
            .require_ref::<IndCastersEnc>((normalized.param, normalized.concrete.ty))?;
        let ty_args = self
            .deps
            .require_dep::<GArgsTyEnc>(normalized.concrete.args)?;
        Ok(child_nodes
            .pieces_of(idx)
            .into_iter()
            .map(|piece| ChildCast {
                child,
                piece,
                casters,
                ty_args,
                range: child_nodes.range(piece),
            })
            .collect())
    }

    /// The current key of the child's (concrete) piece.
    fn child_key(&mut self, cast: &ChildCast<'vir>) -> EncodeResult<'vir, vir::ExprSnap<'vir>, E> {
        self.child_key_at(cast, None)
    }

    /// The key of the child's (concrete) piece, of the place at `label`.
    fn child_key_at(
        &mut self,
        cast: &ChildCast<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, vir::ExprSnap<'vir>, E> {
        let ty = cast.child.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let snap = self.place_snap_at(cast.child, label)?;
        Ok(nodes.key(cast.piece, snap))
    }

    /// The snapshot of `place` (labelled at `label`): of a reference, the one
    /// at the label (its key is stable, as for its own piece); otherwise read
    /// now at its address at the label (the place was just folded).
    fn place_snap_at(
        &mut self,
        place: Place<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, vir::ExprSnap<'vir>, E> {
        if label.is_none() || place.ty(self.pcg_ctxt()).ty.is_ref() {
            let snap = self.encode_place_with_snap(place)?.1;
            return Ok(self.vcx.maybe_apply_label(snap, label));
        }
        let place_ref = self.encode_place(place)?;
        let ty = place_ref.ty.ty;
        let place_ref = self
            .vcx
            .maybe_apply_label(place_ref.expr.expect_predicate(), label);
        let data = self.ty_use_impure(ty);
        Ok(data.ref_to_snap(place_ref))
    }

    /// The expansions of lifetime projections that are open in `state` (the
    /// state before the current phase, or the one a package unblocks): those
    /// whose place is still expanded by a Deref or Expansion edge or, for an
    /// owned place (expanded by repacks), that has no capability of its own in
    /// `pcg` (without it: whose children's places are current).
    pub(crate) fn nested_set_open_expansions(
        &mut self,
        state: &BorrowsState<'_, 'vir>,
        pcg: Option<&Pcg<'_, 'vir>>,
    ) {
        let mut place_edges: Vec<(Place<'vir>, Vec<Place<'vir>>)> = Vec::new();
        let mut lp_expansions = Vec::new();
        for edge in state.graph().edges() {
            match edge.kind() {
                BorrowPcgEdgeKind::Deref(deref) => place_edges.push((
                    deref.blocked_place().place(),
                    vec![deref.deref_place().place()],
                )),
                BorrowPcgEdgeKind::BorrowPcgExpansion(expansion) => match expansion.base() {
                    PcgNode::Place(base) => place_edges.push((
                        base.place(),
                        expansion
                            .expansion()
                            .iter()
                            .filter_map(|child| match child {
                                PcgNode::Place(child) => Some(child.place()),
                                PcgNode::LifetimeProjection(_) => None,
                            })
                            .collect(),
                    )),
                    PcgNode::LifetimeProjection(_) => {
                        if Self::lp_expansion_any(expansion).is_some() {
                            lp_expansions.push(expansion.clone());
                        }
                    }
                },
                _ => {}
            }
        }
        self.owed_expansions = pcg.map(|pcg| self.owed_expansions(pcg)).unwrap_or_default();
        self.open_expansions = lp_expansions
            .into_iter()
            .filter(|expansion| {
                let children = expansion
                    .expansion()
                    .iter()
                    .filter_map(|child| match child {
                        PcgNode::LifetimeProjection(lp) => Some(lp.base()),
                        PcgNode::Place(_) => None,
                    })
                    .collect::<Vec<_>>();
                let Some((base, _, _)) = Self::lp_expansion_any(expansion) else {
                    return false;
                };
                let owned = !base
                    .projection()
                    .iter()
                    .any(|e| matches!(e, mir::ProjectionElem::Deref));
                let owned_expanded = match pcg {
                    Some(pcg) => owned && pcg.place_capability(base, self.pcg_ctxt()).is_none(),
                    None => children
                        .iter()
                        .all(|c| matches!(c, MaybeLabelledPlace::Current(_))),
                };
                let place_expanded = place_edges.iter().any(|(blocked, targets)| {
                    *blocked == base && children.iter().all(|c| targets.contains(&c.place()))
                });
                owned_expanded || place_expanded
            })
            .collect();
    }

    /// Drop the open expansion of `place↓idx` (for the rest of the phase);
    /// whether there was one.
    fn forget_open_expansion(&mut self, place: Place<'vir>, idx: usize) -> bool {
        let before = self.open_expansions.len();
        self.open_expansions.retain(|e| {
            !Self::lp_expansion_any(e).is_some_and(|(base, region, _)| {
                strip_downcasts(base) == place && region.index() == idx
            })
        });
        self.open_expansions.len() != before
    }

    /// The open expansions of lifetime projections of `place`.
    fn open_expansions_of(&self, place: Place<'vir>) -> Vec<BorrowPcgExpansion<'vir>> {
        self.open_expansions
            .iter()
            .filter(|e| {
                Self::lp_expansion_any(e).is_some_and(|(base, ..)| strip_downcasts(base) == place)
            })
            .cloned()
            .collect()
    }

    /// Before the place `place` is folded (labelled at `label`): cast the
    /// children of its open expansions back to generic, while they are still
    /// concrete.
    pub(crate) fn nested_before_place_fold(
        &mut self,
        place: Place<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        // the children's pieces that are still to be folded (their parts were
        // collapsed in this phase): fold them first
        let (children, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending_aggregates)
            .into_iter()
            .partition(|(target, _)| {
                target
                    .place()
                    .last_projection()
                    .is_some_and(|(parent, _)| parent == place)
            });
        self.pending_aggregates = rest;
        self.fold_aggregate_targets(children)?;
        let mut casts = Vec::new();
        for expansion in self.open_expansions.clone() {
            let Some((base, _, children)) = Self::lp_expansion_any(&expansion) else {
                continue;
            };
            for (child, idx) in children {
                if child
                    .last_projection()
                    .is_some_and(|(parent, _)| parent == place)
                {
                    casts.extend(self.child_casts(base, child, idx)?);
                }
            }
        }
        // enclosed pieces first: their keys read the enclosing pieces
        casts.sort_by_key(|cast| std::cmp::Reverse(cast.piece));
        casts.dedup_by(|a, b| a.child == b.child && a.piece == b.piece);
        for cast in casts {
            let key = self.child_key_at(&cast, label)?;
            self.stmt(
                (cast.casters.make_generic)(
                    key,
                    cast.ty_args.get_ty(),
                    cast.ty_args.get_const(),
                    cast.range.lo,
                    cast.range.hi,
                )
                .alloc(),
            );
        }
        Ok(())
    }

    /// After the place `place` was folded (labelled at `label`): fold the
    /// lifetime projections of `place` that the PCG expanded together with it
    /// (enclosing lifetimes first), with their current keys.
    pub(crate) fn nested_after_place_fold(
        &mut self,
        place: Place<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        let open = self.open_expansions_of(place);
        self.open_expansions.retain(|e| {
            !Self::lp_expansion_any(e).is_some_and(|(base, ..)| strip_downcasts(base) == place)
        });
        let regions = open
            .iter()
            .filter_map(|e| Self::lp_expansion_any(e).map(|(_, region, _)| region))
            .collect::<Vec<_>>();
        if regions.is_empty() {
            return Ok(());
        }
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let snap = self.place_snap_at(place, label)?;
        for idx in nodes.order().to_vec() {
            if regions.contains(&idx) {
                for piece in nodes.apps(idx, snap) {
                    self.stmt(self.vcx.mk_fold_stmt(piece));
                }
            }
        }
        Ok(())
    }

    /// An `Aggregate` borrow-flow edge was removed: the loans the parts held
    /// expired, so the value's piece is unfolded (once per phase), and a part
    /// at a generic field position is cast back to its concrete pieces. If
    /// the target node is labelled (the value was expanded since), its fields
    /// hold the pieces already.
    pub(crate) fn nested_aggregate_removed(
        &mut self,
        source: MaybeLabelledPlace<'vir>,
        source_idx: RegionIdx,
        target: MaybeLabelledPlace<'vir>,
        target_idx: usize,
        target_labelled: bool,
        kind: &BorrowFlowEdgeKind<'vir>,
    ) -> EncodeResult<'vir, (), E> {
        let target_place = target.place();
        if source.place().is_prefix_of(target_place) || target_place.is_prefix_of(source.place()) {
            return Ok(());
        }
        if target_labelled {
            // the value is dead: its expansion is not folded again
            self.forget_open_expansion(target_place, target_idx);
            return Ok(());
        }
        if !self
            .released_aggregates
            .contains(&(target_place, target_idx))
        {
            let idx = RegionIdx::from(target_idx);
            let key_label = match target {
                MaybeLabelledPlace::Labelled(snap) => Some(self.get_location_label(snap.at())),
                MaybeLabelledPlace::Current(_) => None,
            };
            for piece in self
                .node_pred_apps(target_place, idx, key_label)?
                .into_iter()
                .rev()
            {
                self.stmt(self.vcx.mk_unfold_stmt(piece));
            }
            self.released_aggregates.push((target_place, target_idx));
        }

        // the part at a generic field position: back to its concrete pieces
        let BorrowFlowEdgeKind::Aggregate { field_idx, .. } = kind else {
            return Ok(());
        };
        if !self
            .released_aggregates
            .contains(&(target_place, target_idx))
            || self.released_parts.contains(&(source, source_idx))
        {
            return Ok(());
        }
        let Some(normalized) = self.generic_part_field(target_place, *field_idx) else {
            return Ok(());
        };
        self.released_parts.push((source, source_idx));
        let casters = self
            .deps
            .require_ref::<IndCastersEnc>((normalized.param, normalized.concrete.ty))?;
        let ty_args = self
            .deps
            .require_dep::<GArgsTyEnc>(normalized.concrete.args)?;
        let label = match source {
            MaybeLabelledPlace::Labelled(snap) => Some(self.get_location_label(snap.at())),
            MaybeLabelledPlace::Current(_) => None,
        };
        let place = source.place();
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let snap = self
            .vcx
            .maybe_apply_label(self.encode_place_with_snap(place)?.1, label);
        // enclosing pieces first: the concrete keys of the enclosed ones read them
        for piece in nodes.pieces_of(source_idx) {
            let key = nodes.key(piece, snap);
            let generic =
                (casters.make_generic_s)(key.downcast_ty(), ty_args.get_ty(), ty_args.get_const());
            self.stmt(
                (casters.make_concrete)(
                    generic.upcast_ty(),
                    ty_args.get_ty(),
                    ty_args.get_const(),
                    nodes.range(piece).lo,
                    nodes.range(piece).hi,
                )
                .alloc(),
            );
        }
        Ok(())
    }

    /// An `Aggregate` borrow-flow edge was added: either a value built from
    /// parts (`t = C { f: op }`) or an owned place collapsed into its base
    /// (`t.f↓r -> t↓r`). Either way the target's piece is folded at the end of
    /// the phase (unless its loans expired), after casting a part at a generic
    /// field position.
    pub(crate) fn nested_aggregate_added(
        &mut self,
        source: MaybeLabelledPlace<'vir>,
        source_idx: RegionIdx,
        target: MaybeLabelledPlace<'vir>,
        target_idx: usize,
        kind: &BorrowFlowEdgeKind<'vir>,
    ) -> EncodeResult<'vir, (), E> {
        let BorrowFlowEdgeKind::Aggregate { field_idx, .. } = kind else {
            return Ok(());
        };
        let MaybeLabelledPlace::Current(target_place) = target else {
            return Ok(());
        };
        if source.place().is_prefix_of(target_place) || target_place.is_prefix_of(source.place()) {
            // collapse of an owned place: its expansion was cast back when the
            // place was folded
            let was_expanded = self.forget_open_expansion(target_place, target_idx);
            if was_expanded && !self.pending_aggregates.contains(&(target, target_idx)) {
                self.pending_aggregates.push((target, target_idx));
            }
            return Ok(());
        }
        // the part, at its generic field position
        if let Some(normalized) = self.generic_part_field(target_place, *field_idx) {
            let casters = self
                .deps
                .require_ref::<IndCastersEnc>((normalized.param, normalized.concrete.ty))?;
            let ty_args = self
                .deps
                .require_dep::<GArgsTyEnc>(normalized.concrete.args)?;
            // one edge per target node: cast once, at the end of the phase
            if !self
                .pending_aggregate_casts
                .iter()
                .any(|c| c.source == source && c.source_idx == source_idx)
            {
                self.pending_aggregate_casts.push(AggregateCast {
                    source,
                    source_idx,
                    casters,
                    ty_args,
                });
            }
        }
        if !self.pending_aggregates.contains(&(target, target_idx)) {
            self.pending_aggregates.push((target, target_idx));
        }
        Ok(())
    }

    /// `place = C { .. }` was assigned: its pieces are folded at the end of
    /// the phase, also those no part flows into (e.g. of `None`, without PCG
    /// `Aggregate` edges).
    pub(crate) fn nested_aggregate_assigned(
        &mut self,
        place: Place<'vir>,
    ) -> EncodeResult<'vir, (), E> {
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let target = MaybeLabelledPlace::Current(place);
        for idx in nodes.order().to_vec() {
            if nodes.holds(idx) && !self.pending_aggregates.contains(&(target, idx.index())) {
                self.pending_aggregates.push((target, idx.index()));
            }
        }
        Ok(())
    }

    /// The field `field_idx` of the value at `target_place` built from parts,
    /// if it is at a generic position (and concrete).
    fn generic_part_field(
        &self,
        target_place: Place<'vir>,
        field_idx: usize,
    ) -> Option<crate::encoders::ty::RustTyNormalized<'vir>> {
        let target_ty = target_place.ty(self.pcg_ctxt());
        let decomp = RustTyDecomposition::from_ty(target_ty.ty, self.def_id);
        let field = match &decomp.ty.specifics {
            TySpecifics::StructLike(data) => data.fields.get(field_idx).map(|f| f.ty()),
            TySpecifics::EnumLike(data) => target_ty
                .variant_index
                .or(self
                    .aggregate_variants
                    .iter()
                    .find(|(place, _)| *place == target_place)
                    .map(|(_, v)| *v))
                .and_then(|v| data.variants.iter().find(|var| var.data.vid == v))
                .and_then(|var| var.inner.fields.get(field_idx))
                .map(|f| f.ty()),
            _ => None,
        };
        field.and_then(|f| f.decompose_compare_normalize(decomp.ty.params, decomp.args))
    }

    /// At the end of a phase: cast the parts at generic field positions, then
    /// fold the pieces of the values built from them.
    pub(crate) fn nested_fold_aggregates(&mut self) -> EncodeResult<'vir, (), E> {
        self.released_aggregates.clear();
        self.released_parts.clear();
        for cast in std::mem::take(&mut self.pending_aggregate_casts) {
            let label = match cast.source {
                MaybeLabelledPlace::Labelled(snap) => Some(self.get_location_label(snap.at())),
                MaybeLabelledPlace::Current(_) => None,
            };
            let place = cast.source.place();
            let ty = place.ty(self.pcg_ctxt()).ty;
            let nodes = self
                .deps
                .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
            let snap = self.encode_place_with_snap(place)?.1;
            // enclosed pieces first: their keys read the enclosing ones
            for piece in nodes.pieces_of(cast.source_idx).into_iter().rev() {
                let key = self.vcx.maybe_apply_label(nodes.key(piece, snap), label);
                self.stmt(
                    (cast.casters.make_generic)(
                        key,
                        cast.ty_args.get_ty(),
                        cast.ty_args.get_const(),
                        nodes.range(piece).lo,
                        nodes.range(piece).hi,
                    )
                    .alloc(),
                );
            }
        }
        let aggregates = std::mem::take(&mut self.pending_aggregates);
        self.fold_aggregate_targets(aggregates)
    }

    /// Fold the pieces of the values built from parts `aggregates`.
    fn fold_aggregate_targets(
        &mut self,
        aggregates: Vec<(MaybeLabelledPlace<'vir>, usize)>,
    ) -> EncodeResult<'vir, (), E> {
        let mut by_place: Vec<(Place<'vir>, Vec<usize>)> = Vec::new();
        for (target, idx) in aggregates {
            let place = target.place();
            match by_place.iter_mut().find(|(p, _)| *p == place) {
                Some((_, idxs)) => idxs.push(idx),
                None => by_place.push((place, vec![idx])),
            }
        }
        for (place, idxs) in by_place {
            let ty = place.ty(self.pcg_ctxt()).ty;
            let nodes = self
                .deps
                .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
            for idx in nodes.order().to_vec() {
                if idxs.contains(&idx.index()) {
                    for piece in self.node_pred_apps(place, idx, None)? {
                        self.stmt(self.vcx.mk_fold_stmt(piece));
                    }
                }
            }
        }
        Ok(())
    }
}

/// The field of `decomp` (the type of `child`'s parent) that `child` is.
fn child_field(decomp: RustTyDecomposition<'_>, child: Place<'_>) -> Option<usize> {
    match child.last_projection()?.1 {
        mir::ProjectionElem::Field(f, _) => Some(f.as_usize()),
        mir::ProjectionElem::Deref => match &decomp.ty.specifics {
            TySpecifics::MutRef(_) => Some(0),
            TySpecifics::StructLike(data) if decomp.ty.special == RustTySpecial::Box => {
                data.fields.len().checked_sub(1)
            }
            _ => None,
        },
        _ => None,
    }
}

/// `place` without trailing downcasts (a variant of an enum place is the same
/// value).
fn strip_downcasts(mut place: Place<'_>) -> Place<'_> {
    while let Some((parent, mir::ProjectionElem::Downcast(..))) = place.last_projection() {
        place = parent;
    }
    place
}
