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

use pcg::{
    borrow_pcg::{
        borrow_pcg_expansion::BorrowPcgExpansion, edge::borrow_flow::BorrowFlowEdgeKind,
    },
    pcg::PcgNode,
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
    },
};

/// A lifetime projection unfolded at an expansion, to be folded when its
/// place is folded again.
#[derive(Clone)]
pub struct PendingExpansion<'vir> {
    base: Place<'vir>,
    region: RegionIdx,
    casts: Vec<ChildCast<'vir>>,
}

/// The bookkeeping of open nested pieces along one path: it is saved at the
/// end of a block and restored for its successors (a successor sees the state
/// of a predecessor, not of the block encoded last).
#[derive(Clone, Default)]
pub struct NestedPathState<'vir> {
    pending_expansions: Vec<PendingExpansion<'vir>>,
    early_expansions: Vec<BorrowPcgExpansion<'vir>>,
    expired_aggregates: Vec<(Place<'vir>, usize)>,
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
    query: vir::ExprLifetimeQuery<'vir>,
}

impl<'vir, E: TaskEncoder> ImpureEncVisitor<'vir, '_, E> {
    /// At the start of `block`: the state at the end of an encoded
    /// predecessor (empty at the entry).
    pub(crate) fn nested_enter_block(&mut self, block: mir::BasicBlock) {
        let state = self.body.basic_blocks.predecessors()[block]
            .iter()
            .find_map(|pred| self.nested_at_exit.get(pred))
            .cloned()
            .unwrap_or_default();
        self.pending_expansions = state.pending_expansions;
        self.early_expansions = state.early_expansions;
        self.expired_aggregates = state.expired_aggregates;
    }

    /// At the end of `block`: save the state for its successors.
    pub(crate) fn nested_exit_block(&mut self, block: mir::BasicBlock) {
        let state = NestedPathState {
            pending_expansions: self.pending_expansions.clone(),
            early_expansions: self.early_expansions.clone(),
            expired_aggregates: self.expired_aggregates.clone(),
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
                let key = self.vcx.maybe_apply_label(nodes.key(piece, snap), key_label);
                nodes.app(piece, key, None)
            })
            .collect())
    }

    /// The snapshot of `place`, read along its path from the local: through a
    /// `&mut`, from the reference's snapshot extended by its own piece. Unlike
    /// the place's own predicate (which an expansion may have unfolded or a
    /// deref may not have produced yet), this only needs the own pieces of the
    /// references on the path, as at the start of a phase.
    fn path_snap(&mut self, place: Place<'vir>) -> EncodeResult<'vir, vir::ExprSnap<'vir>, E> {
        let Some((parent, elem)) = place.last_projection() else {
            return Ok(self.encode_place_with_snap(place)?.1);
        };
        let parent_ty = parent.ty(self.pcg_ctxt());
        Ok(match elem {
            mir::ProjectionElem::Deref => match parent_ty.ty.kind() {
                ty::TyKind::Ref(.., ty::Mutability::Mut) => {
                    let snap = self.path_snap(parent)?;
                    let nodes = self.deps.require_dep::<IndirectNodesEnc>(
                        RustTyDecomposition::from_ty(parent_ty.ty, self.def_id),
                    )?;
                    let extended = nodes.extend_with(RegionIdx::from(0), snap);
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
                    e_ty.expect_structlike().box_value_access(snap.downcast_ty())
                }
                _ => self.encode_place_with_snap(place)?.1,
            },
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
    fn lp_expansion(
        expansion: &BorrowPcgExpansion<'vir>,
    ) -> Option<(Place<'vir>, RegionIdx, Vec<(Place<'vir>, RegionIdx)>)> {
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
            matches!(child.last_projection(), Some((_, mir::ProjectionElem::Downcast(..))))
        }) {
            return None;
        }
        Some((base_place, base.region_idx(), children))
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
        for piece in self.node_pred_apps(base, region, key_label)?.into_iter().rev() {
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
        let Some((base_place, region, children)) = Self::lp_expansion(expansion) else {
            return Ok(());
        };
        if !unfolded {
            let label = self.current_phase_label();
            self.nested_expansion_unfold(expansion, Some(label))?;
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
                    cast.query,
                )
                .alloc(),
            );
        }
        self.pending_expansions.push(PendingExpansion {
            base: strip_downcasts(base_place),
            region,
            casts,
        });
        Ok(())
    }

    /// At `return`: fold the pending pieces of the arguments (owned values
    /// whose expansions the PCG does not collapse at the end), keyed by their
    /// pre-state snapshots as in the postcondition.
    pub(crate) fn nested_fold_at_return(&mut self) -> EncodeResult<'vir, (), E> {
        let pending = std::mem::take(&mut self.pending_expansions);
        for p in pending.iter().rev() {
            if p.base.projection().is_empty()
                && self.body.local_kind(p.base.local) == mir::LocalKind::Arg
            {
                let ty = p.base.ty(self.pcg_ctxt()).ty;
                let nodes = self
                    .deps
                    .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
                let snap = self.vcx.mk_old_expr(self.encode_place_with_snap(p.base)?.1);
                for piece in nodes.apps(p.region, snap) {
                    self.stmt(self.vcx.mk_fold_stmt(piece));
                }
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
        let child_nodes = self.deps.require_dep::<IndirectNodesEnc>(child_decomp)?;
        let casters = self
            .deps
            .require_ref::<IndCastersEnc>((normalized.param, normalized.concrete.ty))?;
        let ty_args = self.deps.require_dep::<GArgsTyEnc>(normalized.concrete.args)?;
        Ok(child_nodes
            .pieces_of(idx)
            .into_iter()
            .map(|piece| ChildCast {
                child,
                piece,
                casters,
                ty_args,
                query: child_nodes.piece(piece).query,
            })
            .collect())
    }

    /// The current key of the child's (concrete) piece.
    fn child_key(&mut self, cast: &ChildCast<'vir>) -> EncodeResult<'vir, vir::ExprSnap<'vir>, E> {
        let ty = cast.child.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let snap = self.encode_place_with_snap(cast.child)?.1;
        Ok(nodes.key(cast.piece, snap))
    }

    /// Before the place `place` is folded: cast the children of its pending
    /// expansions back to generic, while they are still concrete.
    pub(crate) fn nested_before_place_fold(&mut self, place: Place<'vir>) -> EncodeResult<'vir, (), E> {
        let casts = self
            .pending_expansions
            .iter()
            .flat_map(|p| p.casts.clone())
            .filter(|c| c.child.last_projection().is_some_and(|(parent, _)| parent == place))
            .collect::<Vec<_>>();
        // enclosed pieces first: their keys read the enclosing pieces
        let mut casts = casts;
        casts.sort_by_key(|cast| std::cmp::Reverse(cast.piece));
        for cast in casts {
            let key = self.child_key(&cast)?;
            self.stmt(
                (cast.casters.make_generic)(
                    key,
                    cast.ty_args.get_ty(),
                    cast.ty_args.get_const(),
                    cast.query,
                )
                .alloc(),
            );
        }
        Ok(())
    }

    /// After the place `place` was folded: fold the pieces of its pending
    /// expansions (enclosing lifetimes first) with their current keys.
    pub(crate) fn nested_after_place_fold(&mut self, place: Place<'vir>) -> EncodeResult<'vir, (), E> {
        let (pending, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending_expansions)
            .into_iter()
            .partition(|p| p.base == place);
        self.pending_expansions = rest;
        if pending.is_empty() {
            return Ok(());
        }
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        for idx in nodes.order().to_vec() {
            if pending.iter().any(|p| p.region == idx) {
                for piece in self.node_pred_apps(place, idx, None)? {
                    self.stmt(self.vcx.mk_fold_stmt(piece));
                }
            }
        }
        Ok(())
    }

    /// An `Aggregate` borrow-flow edge was removed. Removing an edge that
    /// built a value from parts means that the loans the parts held expire:
    /// their pieces go back to the lenders, so the value's piece is not
    /// refolded when it is collapsed.
    pub(crate) fn nested_aggregate_removed(
        &mut self,
        source: MaybeLabelledPlace<'vir>,
        target: MaybeLabelledPlace<'vir>,
        target_idx: usize,
    ) {
        let target_place = target.place();
        if source.place().is_prefix_of(target_place) || target_place.is_prefix_of(source.place()) {
            return;
        }
        if !self.expired_aggregates.contains(&(target_place, target_idx)) {
            self.expired_aggregates.push((target_place, target_idx));
        }
        self.pending_expansions
            .retain(|p| !(p.base == target_place && p.region.index() == target_idx));
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
            let was_expanded = self
                .pending_expansions
                .iter()
                .any(|p| p.base == target_place && p.region.index() == target_idx);
            self.pending_expansions
                .retain(|p| !(p.base == target_place && p.region.index() == target_idx));
            if was_expanded
                && !self.expired_aggregates.contains(&(target_place, target_idx))
                && !self.pending_aggregates.contains(&(target, target_idx))
            {
                self.pending_aggregates.push((target, target_idx));
            }
            return Ok(());
        }
        // the part, at its generic field position
        let target_ty = target_place.ty(self.pcg_ctxt());
        let decomp = RustTyDecomposition::from_ty(target_ty.ty, self.def_id);
        let field = match &decomp.ty.specifics {
            TySpecifics::StructLike(data) => data.fields.get(*field_idx).map(|f| f.ty()),
            TySpecifics::EnumLike(data) => target_ty
                .variant_index
                .or(self
                    .aggregate_variant
                    .filter(|(place, _)| *place == target_place)
                    .map(|(_, v)| v))
                .and_then(|v| data.variants.iter().find(|var| var.data.vid == v))
                .and_then(|var| var.inner.fields.get(*field_idx))
                .map(|f| f.ty()),
            _ => None,
        };
        if let Some(normalized) =
            field.and_then(|f| f.decompose_compare_normalize(decomp.ty.params, decomp.args))
        {
            let casters = self
                .deps
                .require_ref::<IndCastersEnc>((normalized.param, normalized.concrete.ty))?;
            let ty_args = self.deps.require_dep::<GArgsTyEnc>(normalized.concrete.args)?;
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
        // a new value: its loans are alive
        self.expired_aggregates
            .retain(|e| *e != (target_place, target_idx));
        if !self.pending_aggregates.contains(&(target, target_idx)) {
            self.pending_aggregates.push((target, target_idx));
        }
        Ok(())
    }

    /// `place = C { .. }` was assigned: its pieces are folded at the end of
    /// the phase, also those no part flows into (e.g. of `None`, without PCG
    /// `Aggregate` edges).
    pub(crate) fn nested_aggregate_assigned(&mut self, place: Place<'vir>) -> EncodeResult<'vir, (), E> {
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let target = MaybeLabelledPlace::Current(place);
        for idx in nodes.order().to_vec() {
            self.expired_aggregates.retain(|e| *e != (place, idx.index()));
            if nodes.holds(idx) && !self.pending_aggregates.contains(&(target, idx.index())) {
                self.pending_aggregates.push((target, idx.index()));
            }
        }
        Ok(())
    }

    /// At the end of a phase: cast the parts at generic field positions, then
    /// fold the pieces of the values built from them.
    pub(crate) fn nested_fold_aggregates(&mut self) -> EncodeResult<'vir, (), E> {
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
                        nodes.piece(piece).query,
                    )
                    .alloc(),
                );
            }
        }
        let aggregates = std::mem::take(&mut self.pending_aggregates);
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

/// `place` without trailing downcasts (a variant of an enum place is the same
/// value).
fn strip_downcasts(mut place: Place<'_>) -> Place<'_> {
    while let Some((parent, mir::ProjectionElem::Downcast(..))) = place.last_projection() {
        place = parent;
    }
    place
}
