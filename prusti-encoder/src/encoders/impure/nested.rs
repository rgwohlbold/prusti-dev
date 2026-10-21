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
//! Their parts are guarded by `has_ind(T)`: at an expansion, a child's piece
//! is made empty (`ind_empty` or a fold) if `T` holds nothing.

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
        has_ind::TyHasIndEnc,
        indirect_node::{IndirectNodesEnc, RegionIdx},
        indirect_pred::TyIndirectEnc,
        occurrences::Step,
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
    pending_expansions: Vec<PendingExpansion<'vir>>,
    early_expansions: Vec<BorrowPcgExpansion<'vir>>,
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
        self.opened_regions_in = state.opened_regions_in;
        self.applied_wands = state.applied_wands;
    }

    /// At the end of `block`: save the state for its successors.
    pub(crate) fn nested_exit_block(&mut self, block: mir::BasicBlock) {
        let state = NestedPathState {
            pending_expansions: self.pending_expansions.clone(),
            early_expansions: self.early_expansions.clone(),
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
    pub(crate) fn path_snap(&mut self, place: Place<'vir>) -> EncodeResult<'vir, vir::ExprSnap<'vir>, E> {
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
                .map(|piece| (piece, self.bind(self.vcx.maybe_apply_label(nodes.key(piece, snap), key_label))))
                .collect::<Vec<_>>();
            for (piece, key) in keys.iter().rev() {
                self.stmt(self.vcx.mk_unfold_stmt(nodes.app(*piece, *key, None)));
            }
            self.opened_regions_in.push(OpenedRegionsIn {
                expansion: expansion.clone(),
                keys,
            });
            return Ok(());
        }
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
        if let Some(i) = self.opened_regions_in.iter().position(|o| o.expansion == *expansion) {
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
    /// pre-state snapshots as in the postcondition. Of `RegionsIn(T)` nodes,
    /// only those the postcondition gives back unblocked (the others moved
    /// with the values), including expansions inside the argument (deepest
    /// first); the others stay pending for the wand packages.
    pub(crate) fn nested_fold_at_return(&mut self) -> EncodeResult<'vir, (), E> {
        let pending = std::mem::take(&mut self.pending_expansions);
        let unblocked = self
            .wands
            .unblocked_regions_in(self.vcx, self.deps)
            .into_iter()
            .map(|(local, idx)| (local, Place::from(local).regions(self.pcg_ctxt())[idx]))
            .collect::<Vec<_>>();
        let mut regions_in = Vec::new();
        for p in pending.iter().rev() {
            let ty = p.base.ty(self.pcg_ctxt()).ty;
            let nodes = self
                .deps
                .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
            if nodes.is_regions_in(p.region) {
                regions_in.push(p.clone());
            } else if p.base.projection().is_empty()
                && self.body.local_kind(p.base.local) == mir::LocalKind::Arg
            {
                let snap = self.vcx.mk_old_expr(self.encode_place_with_snap(p.base)?.1);
                for piece in nodes.apps(p.region, snap) {
                    self.stmt(self.vcx.mk_fold_stmt(piece));
                }
            }
        }
        // deepest first
        regions_in.sort_by_key(|p| std::cmp::Reverse(p.base.projection().len()));
        let mut keep = Vec::new();
        for p in regions_in {
            let local = p.base.local;
            let region = p.base.regions(self.pcg_ctxt())[p.region];
            if self.body.local_kind(local) != mir::LocalKind::Arg
                || self.body.local_decls[local].ty.is_ref()
                || p.base.projection().iter().any(|e| matches!(e, mir::ProjectionElem::Deref))
                || !unblocked.contains(&(local, region))
            {
                keep.push(p);
                continue;
            }
            let mut casts = p.casts.clone();
            casts.sort_by_key(|cast| std::cmp::Reverse(cast.piece));
            for cast in casts {
                let key = self.child_key(&cast)?;
                self.stmt(
                    (cast.casters.make_generic)(key, cast.ty_args.get_ty(), cast.ty_args.get_const(), cast.query)
                        .alloc(),
                );
            }
            let ty = p.base.ty(self.pcg_ctxt()).ty;
            let nodes = self
                .deps
                .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
            let snap = self.vcx.mk_old_expr(self.path_snap(p.base)?);
            for piece in nodes.apps(p.region, snap) {
                self.stmt(self.vcx.mk_fold_stmt(piece));
            }
        }
        self.pending_expansions = keep;
        Ok(())
    }

    /// A place was collapsed without capability (moved out or dropped): the
    /// pieces of its open `RegionsIn(T)` expansions went with the moved
    /// values. Those of a by-value argument stay for the fold at `return`.
    pub(crate) fn nested_drop(&mut self, place: Place<'vir>) -> EncodeResult<'vir, (), E> {
        let mut keep = Vec::new();
        for p in std::mem::take(&mut self.pending_expansions) {
            let ty = p.base.ty(self.pcg_ctxt()).ty;
            let nodes = self
                .deps
                .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
            let by_value_arg = self.body.local_kind(p.base.local) == mir::LocalKind::Arg
                && !self.body.local_decls[p.base.local].ty.is_ref();
            if !nodes.is_regions_in(p.region) || !place.is_prefix_of(p.base) || by_value_arg {
                keep.push(p);
            }
        }
        self.pending_expansions = keep;
        Ok(())
    }

    /// After unfolding the `RegionsIn(T_j)` pieces (with `keys`) of `base` at
    /// an expansion into `children`: a part is guarded by `has_ind(T_j)`, so
    /// without it, make the child's piece empty: `ind_empty` for a field of
    /// type `T_j` (`p_Param_ind`, before any cast), a fold for a constructor
    /// around `T_j`.
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
                let child_decomp = RustTyDecomposition::from_ty(child.ty(self.pcg_ctxt()).ty, self.def_id);
                let child_nodes = self.deps.require_dep::<IndirectNodesEnc>(child_decomp)?;
                let path = part.path.iter().chain(rest).copied().collect::<Vec<_>>();
                let Some(piece) = child_nodes.find(&path) else {
                    continue;
                };
                let k = part.read.read(*key);
                let tau = args.get_ty()[*j];
                let no_ind = self
                    .vcx
                    .mk_unary_op_expr(vir::UnOpKind::Not, TyHasIndEnc::has_ind_fn().call()(tau).upcast_ty())
                    .downcast_ty::<vir::Bool>();
                let empty = if part.path.is_empty() {
                    let query = child_nodes.piece(piece).query;
                    (param_ind.empty.unwrap())(k, &[tau], &[], query).alloc()
                } else {
                    self.vcx.mk_fold_stmt(child_nodes.app(piece, k, None))
                };
                self.stmt(self.vcx.mk_if_stmt(no_ind, self.vcx.alloc_slice(&[empty]), &[]));
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
        let place_ref = self.vcx.maybe_apply_label(place_ref.expr.expect_predicate(), label);
        let data = self.ty_use_impure(ty);
        Ok(data.ref_to_snap(place_ref))
    }

    /// Before the place `place` is folded (labelled at `label`): cast the
    /// children of its pending expansions back to generic, while they are
    /// still concrete.
    pub(crate) fn nested_before_place_fold(
        &mut self,
        place: Place<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
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
            let key = self.child_key_at(&cast, label)?;
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

    /// After the place `place` was folded (labelled at `label`): fold the
    /// pieces of its pending expansions (enclosing lifetimes first) with
    /// their current keys.
    pub(crate) fn nested_after_place_fold(
        &mut self,
        place: Place<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
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
        let snap = self.place_snap_at(place, label)?;
        for idx in nodes.order().to_vec() {
            if pending.iter().any(|p| p.region == idx) {
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
            self.pending_expansions
                .retain(|p| !(p.base == target_place && p.region.index() == target_idx));
            return Ok(());
        }
        if !self.released_aggregates.contains(&(target_place, target_idx)) {
            let idx = RegionIdx::from(target_idx);
            let key_label = match target {
                MaybeLabelledPlace::Labelled(snap) => Some(self.get_location_label(snap.at())),
                MaybeLabelledPlace::Current(_) => None,
            };
            for piece in self.node_pred_apps(target_place, idx, key_label)?.into_iter().rev() {
                self.stmt(self.vcx.mk_unfold_stmt(piece));
            }
            self.released_aggregates.push((target_place, target_idx));
        }

        // the part at a generic field position: back to its concrete pieces
        let BorrowFlowEdgeKind::Aggregate { field_idx, .. } = kind else {
            return Ok(());
        };
        if !self.released_aggregates.contains(&(target_place, target_idx))
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
        let ty_args = self.deps.require_dep::<GArgsTyEnc>(normalized.concrete.args)?;
        let label = match source {
            MaybeLabelledPlace::Labelled(snap) => Some(self.get_location_label(snap.at())),
            MaybeLabelledPlace::Current(_) => None,
        };
        let place = source.place();
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let snap = self.vcx.maybe_apply_label(self.encode_place_with_snap(place)?.1, label);
        // enclosing pieces first: the concrete keys of the enclosed ones read them
        for piece in nodes.pieces_of(source_idx) {
            let key = nodes.key(piece, snap);
            let generic = (casters.make_generic_s)(key.downcast_ty(), ty_args.get_ty(), ty_args.get_const());
            self.stmt(
                (casters.make_concrete)(
                    generic.upcast_ty(),
                    ty_args.get_ty(),
                    ty_args.get_const(),
                    nodes.piece(piece).query,
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
            let was_expanded = self
                .pending_expansions
                .iter()
                .any(|p| p.base == target_place && p.region.index() == target_idx);
            self.pending_expansions
                .retain(|p| !(p.base == target_place && p.region.index() == target_idx));
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
