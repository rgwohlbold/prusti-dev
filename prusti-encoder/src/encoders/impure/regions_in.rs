//! `RegionsIn(T)` pieces in method bodies. The pieces of type parameters
//! (query `path ++ q_all()`) have no PCG nodes; they follow the place
//! expansions instead, like the direct predicates:
//!
//! - Before a place `p` is unfolded, its pieces are unfolded (keys bound now).
//!   A part `p_Param_ind(read(K), T_j, rest)` (a field that is `T_j`) becomes
//!   the child's piece, through `make_concrete_ind` if `T_j` is instantiated
//!   with a constructor; a part is only there if `has_ind(T_j)`, otherwise
//!   the child's (empty) piece is made by `ind_empty` or a fold. For an enum,
//!   the children are made at the expansion of the variant.
//! - Before the children are folded into `p` (or its variant), the casts are
//!   undone; after `p` is folded, its pieces are folded with the current keys.

use pcg::utils::{Place, PrefixRelation};
use prusti_rustc_interface::{abi::VariantIdx, middle::mir};
use task_encoder::TaskEncoder;
use vir::CastType;

use crate::encoders::{
    EncodeResult, ImpureEncVisitor,
    ty::{
        RustTyDecomposition, RustTySpecial, TySpecifics,
        generics::{
            GArgsTy, GArgsTyEnc,
            ind_casters::{IndCasters, IndCastersEnc},
        },
        has_ind::TyHasIndEnc,
        indirect_node::{IndirectNodesEnc, NodeId},
        indirect_pred::TyIndirectEnc,
        occurrences::Step,
    },
};

/// The `RegionsIn(T)` pieces of an unfolded place, to be folded when the
/// place is folded again.
#[derive(Clone)]
pub struct PendingRegionsIn<'vir> {
    base: Place<'vir>,
    /// The pieces of `base` and their keys at the unfold.
    keys: Vec<(usize, vir::ExprSnap<'vir>)>,
    /// For an enum: the variant whose expansion makes the children.
    awaiting: Option<VariantIdx>,
    children: Vec<ChildPiece<'vir>>,
}

/// A child's piece made from a part of its parent's piece.
#[derive(Clone)]
struct ChildPiece<'vir> {
    /// The parent's piece.
    from: usize,
    /// The child's key at the unfold.
    key: vir::ExprSnap<'vir>,
    child: Place<'vir>,
    /// The piece in the child's type.
    piece: usize,
    query: vir::ExprLifetimeQuery<'vir>,
    /// The cast between `p_Param_ind` and the child's constructor.
    cast: Option<(IndCasters<'vir>, GArgsTy<'vir>)>,
}

impl<'vir, E: TaskEncoder> ImpureEncVisitor<'vir, '_, E> {
    /// Before `place` is unfolded into `children`: unfold its pieces and make
    /// the children's.
    pub(crate) fn regions_in_unfold(
        &mut self,
        place: Place<'vir>,
        children: Vec<Place<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        if place.is_shared_ref(self.pcg_ctxt()) || place.projects_shared_ref(self.pcg_ctxt()) {
            return Ok(());
        }
        let base = strip_downcasts(place);
        if base != place {
            // the expansion of a variant: the children of the enum's pieces
            let vid = place.ty(self.pcg_ctxt()).variant_index;
            if let Some(i) = self
                .pending_regions_in
                .iter()
                .rposition(|p| p.base == base && p.awaiting.is_some() && p.awaiting == vid)
            {
                self.pending_regions_in[i].awaiting = None;
                self.regions_in_children(i, vid, &children)?;
            }
            return Ok(());
        }
        let ty = place.ty(self.pcg_ctxt()).ty;
        let decomp = RustTyDecomposition::from_ty(ty, self.def_id);
        let Ok(nodes) = self.deps.require_dep::<IndirectNodesEnc>(decomp) else {
            return Ok(());
        };
        let pieces = (0..nodes.len())
            .filter(|i| {
                matches!(nodes.piece(*i).node, NodeId::RegionsIn(_)) && !nodes.piece(*i).path.is_empty()
            })
            .collect::<Vec<_>>();
        if pieces.is_empty() {
            return Ok(());
        }
        let snap = self.encode_place_with_snap(place)?.1;
        let keys = pieces
            .into_iter()
            .map(|i| (i, self.bind(nodes.key(i, snap))))
            .collect::<Vec<_>>();
        // enclosed ones first (their keys read the enclosing ones)
        for (i, key) in keys.iter().rev() {
            self.stmt(self.vcx.mk_unfold_stmt(nodes.app(*i, *key, None)));
        }
        let awaiting = match decomp.ty.specifics {
            TySpecifics::EnumLike(_) => children.iter().find_map(|c| {
                match c.last_projection() {
                    Some((_, mir::ProjectionElem::Downcast(_, vid))) => Some(vid),
                    _ => None,
                }
            }),
            _ => None,
        };
        let enum_ty = matches!(decomp.ty.specifics, TySpecifics::EnumLike(_));
        self.pending_regions_in.push(PendingRegionsIn {
            base,
            keys,
            awaiting,
            children: Vec::new(),
        });
        if !enum_ty {
            let i = self.pending_regions_in.len() - 1;
            self.regions_in_children(i, None, &children)?;
        }
        Ok(())
    }

    /// Make the pieces of `children` (of the variant `vid`) from the parts of
    /// the pending entry `i`.
    fn regions_in_children(
        &mut self,
        i: usize,
        vid: Option<VariantIdx>,
        children: &[Place<'vir>],
    ) -> EncodeResult<'vir, (), E> {
        let entry = self.pending_regions_in[i].clone();
        let ty = entry.base.ty(self.pcg_ctxt()).ty;
        let decomp = RustTyDecomposition::from_ty(ty, self.def_id);
        let nodes = self.deps.require_dep::<IndirectNodesEnc>(decomp)?;
        let dep = self.deps.require_dep::<TyIndirectEnc>(decomp.ty)?;
        let args = self.deps.require_dep::<GArgsTyEnc>(decomp.args)?;
        let param_ind = self
            .deps
            .require_ref::<TyIndirectEnc>(RustTyDecomposition::param())?;
        let mut made = Vec::new();
        for (g, key) in &entry.keys {
            let Some((Step::In(j), rest)) = nodes.piece(*g).path.split_first() else {
                continue;
            };
            for part in dep.ty_param_parts[*j].iter().filter(|p| p.vid == vid) {
                let Some(child) = children
                    .iter()
                    .copied()
                    .find(|c| self.child_field(decomp, *c) == Some(part.field_idx))
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
                let query = child_nodes.piece(piece).query;
                let k = part.read.read(*key);
                let tau = args.get_ty()[*j];
                let no_ind = self.vcx.mk_unary_op_expr(
                    vir::UnOpKind::Not,
                    TyHasIndEnc::has_ind_fn().call()(tau).upcast_ty(),
                );
                let no_ind = no_ind.downcast_ty::<vir::Bool>();
                let mut cast = None;
                let mut child_key = k;
                if part.path.is_empty() {
                    // the field is `T_j`: `p_Param_ind(k, tau, rest)`
                    let empty = (param_ind.empty.unwrap())(k, &[tau], &[], query).alloc();
                    self.stmt(self.vcx.mk_if_stmt(no_ind, self.vcx.alloc_slice(&[empty]), &[]));
                    if !child_decomp.ty.specifics.is_param() {
                        let casters = self
                            .deps
                            .require_ref::<IndCastersEnc>((RustTyDecomposition::param(), child_decomp.ty))?;
                        let ty_args = self.deps.require_dep::<GArgsTyEnc>(child_decomp.args)?;
                        self.stmt(
                            (casters.make_concrete)(k, ty_args.get_ty(), ty_args.get_const(), query)
                                .alloc(),
                        );
                        child_key = casters.make_concrete_s.call()(k.downcast_ty()).upcast_ty();
                        cast = Some((casters, ty_args));
                    }
                } else {
                    // a constructor around `T_j`: its piece, empty without `T_j`'s
                    let fold = self.vcx.mk_fold_stmt(child_nodes.app(piece, k, None));
                    self.stmt(self.vcx.mk_if_stmt(no_ind, self.vcx.alloc_slice(&[fold]), &[]));
                }
                made.push(ChildPiece { from: *g, key: child_key, child, piece, query, cast });
            }
        }
        self.pending_regions_in[i].children.extend(made);
        Ok(())
    }

    /// The field of `decomp` (the parent's type) that `child` is.
    fn child_field(&self, decomp: RustTyDecomposition<'vir>, child: Place<'vir>) -> Option<usize> {
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

    /// Before `place` is folded: cast its children's pieces back to the
    /// parts of the parent's pieces.
    pub(crate) fn regions_in_before_fold(
        &mut self,
        place: Place<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        let children = self
            .pending_regions_in
            .iter()
            .flat_map(|p| p.children.iter())
            .filter(|c| c.cast.is_some())
            .filter(|c| c.child.last_projection().is_some_and(|(parent, _)| parent == place))
            .cloned()
            .collect::<Vec<_>>();
        // enclosed ones first: their keys read the enclosing ones
        for c in children.iter().rev() {
            let (casters, ty_args) = c.cast.unwrap();
            let child_ty = c.child.ty(self.pcg_ctxt()).ty;
            let nodes = self
                .deps
                .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(child_ty, self.def_id))?;
            let snap = self.place_snap_at(c.child, label)?;
            let key = nodes.key(c.piece, snap);
            self.stmt((casters.make_generic)(key, ty_args.get_ty(), ty_args.get_const(), c.query).alloc());
        }
        Ok(())
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

    /// After `place` was folded: fold its pieces with their current keys (of
    /// the place at `label`, if it is labelled).
    pub(crate) fn regions_in_after_fold(
        &mut self,
        place: Place<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        let Some(i) = self.pending_regions_in.iter().rposition(|p| p.base == place) else {
            return Ok(());
        };
        let entry = self.pending_regions_in.remove(i);
        let ty = place.ty(self.pcg_ctxt()).ty;
        let nodes = self
            .deps
            .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
        let snap = self.place_snap_at(place, label)?;
        for (g, _) in entry.keys {
            let app = nodes.app(g, nodes.key(g, snap), None);
            self.stmt(self.vcx.mk_fold_stmt(app));
        }
        Ok(())
    }

    /// At `return`: fold the pieces of the (by-value) arguments that the
    /// postcondition gives back unblocked (see `regions_in_fold_args`).
    pub(crate) fn regions_in_fold_at_return(&mut self) -> EncodeResult<'vir, (), E> {
        let unblocked = self.wands.unblocked_regions_in(self.vcx, self.deps);
        self.regions_in_fold_args(&unblocked)
    }

    /// Fold the pieces `nodes` of by-value arguments that were unfolded,
    /// keyed by their pre-state snapshots (as in the contract): the pieces of
    /// expanded places inside the argument first (deepest first), whose
    /// children are cast back with their keys at the unfold.
    pub(crate) fn regions_in_fold_args(
        &mut self,
        nodes: &[(mir::Local, NodeId)],
    ) -> EncodeResult<'vir, (), E> {
        let mut locals = nodes.iter().map(|(l, _)| *l).collect::<Vec<_>>();
        locals.dedup();
        for local in locals {
            if self.body.local_kind(local) != mir::LocalKind::Arg
                || self.body.local_decls[local].ty.is_ref()
            {
                // a reference's pieces are folded when its deref is
                continue;
            }
            let mut entries = self
                .pending_regions_in
                .iter()
                .filter(|p| {
                    p.base.local == local
                        && !p.base.projection().iter().any(|e| matches!(e, mir::ProjectionElem::Deref))
                })
                .cloned()
                .collect::<Vec<_>>();
            // deepest first; of several entries of a place, the latest
            entries.reverse();
            entries.sort_by_key(|p| std::cmp::Reverse(p.base.projection().len()));
            let top = entries.iter().find(|p| p.base.projection().is_empty()).cloned();
            let ty = self.body.local_decls[local].ty;
            let local_nodes = self
                .deps
                .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
            // the top-level piece whose part a nested place is in
            let selected = |base: Place<'vir>| match &top {
                None => true,
                Some(top) => top.children.iter().any(|c| {
                    c.child.is_prefix_of(base) && nodes.contains(&(local, local_nodes.piece(c.from).node))
                }),
            };
            let mut done = Vec::new();
            let mut folded: Vec<(Place<'vir>, usize)> = Vec::new();
            for entry in entries {
                if done.contains(&entry.base) {
                    continue;
                }
                done.push(entry.base);
                let is_top = entry.base.projection().is_empty();
                if !is_top && !selected(entry.base) {
                    continue;
                }
                let ty = entry.base.ty(self.pcg_ctxt()).ty;
                let base_nodes = self
                    .deps
                    .require_dep::<IndirectNodesEnc>(RustTyDecomposition::from_ty(ty, self.def_id))?;
                // read along the path from the argument (folded in the pre-state)
                let snap = self.vcx.mk_old_expr(self.path_snap(entry.base)?);
                for (g, _) in &entry.keys {
                    if is_top && !nodes.contains(&(local, base_nodes.piece(*g).node)) {
                        continue;
                    }
                    for c in entry.children.iter().filter(|c| c.from == *g).rev() {
                        if let Some((casters, ty_args)) = c.cast {
                            self.stmt(
                                (casters.make_generic)(c.key, ty_args.get_ty(), ty_args.get_const(), c.query)
                                    .alloc(),
                            );
                        }
                    }
                    let app = base_nodes.app(*g, base_nodes.key(*g, snap), None);
                    self.stmt(self.vcx.mk_fold_stmt(app));
                    folded.push((entry.base, *g));
                }
            }
            // the folded pieces are no longer open
            for p in self.pending_regions_in.iter_mut() {
                p.keys.retain(|(g, _)| !folded.contains(&(p.base, *g)));
            }
            self.pending_regions_in.retain(|p| !p.keys.is_empty());
        }
        Ok(())
    }

    /// `dest = C { .. }` was assigned: fold its pieces from the parts (the
    /// moved operands' pieces, cast to `p_Param_ind` where a field is a type
    /// parameter instantiated with a constructor).
    pub(crate) fn regions_in_aggregate(
        &mut self,
        dest: Place<'vir>,
        vid: Option<VariantIdx>,
    ) -> EncodeResult<'vir, (), E> {
        let ty = dest.ty(self.pcg_ctxt()).ty;
        let decomp = RustTyDecomposition::from_ty(ty, self.def_id);
        let Ok(nodes) = self.deps.require_dep::<IndirectNodesEnc>(decomp) else {
            return Ok(());
        };
        let pieces = (0..nodes.len())
            .filter(|i| {
                matches!(nodes.piece(*i).node, NodeId::RegionsIn(_)) && !nodes.piece(*i).path.is_empty()
            })
            .collect::<Vec<_>>();
        if pieces.is_empty() {
            return Ok(());
        }
        let dep = self.deps.require_dep::<TyIndirectEnc>(decomp.ty)?;
        let vid = match decomp.ty.specifics {
            TySpecifics::EnumLike(_) => vid,
            _ => None,
        };
        let snap = self.encode_place_with_snap(dest)?.1;
        for g in pieces {
            let key = nodes.key(g, snap);
            let Some((Step::In(j), rest)) = nodes.piece(g).path.split_first() else {
                continue;
            };
            for part in dep.ty_param_parts[*j].iter().filter(|p| p.vid == vid) {
                if !part.path.is_empty() {
                    continue;
                }
                let arg = decomp.args.args().iter().filter_map(|a| a.as_type()).nth(*j).unwrap();
                let child = RustTyDecomposition::from_ty(arg, decomp.args.context());
                if child.ty.specifics.is_param() {
                    continue;
                }
                let child_nodes = self.deps.require_dep::<IndirectNodesEnc>(child)?;
                let Some(piece) = child_nodes.find(rest) else {
                    continue;
                };
                let casters = self
                    .deps
                    .require_ref::<IndCastersEnc>((RustTyDecomposition::param(), child.ty))?;
                let ty_args = self.deps.require_dep::<GArgsTyEnc>(child.args)?;
                let child_snap = casters.make_concrete_s.call()(part.read.read(key).downcast_ty());
                let child_key = child_nodes.key(piece, child_snap.upcast_ty());
                self.stmt(
                    (casters.make_generic)(
                        child_key,
                        ty_args.get_ty(),
                        ty_args.get_const(),
                        child_nodes.piece(piece).query,
                    )
                    .alloc(),
                );
            }
            self.stmt(self.vcx.mk_fold_stmt(nodes.app(g, key, None)));
        }
        Ok(())
    }

    /// `place` is collapsed without capability (moved out or dropped): its
    /// pieces went with the moved values. Those of a by-value argument stay
    /// for the fold at `return` (or in a wand package).
    pub(crate) fn regions_in_drop(&mut self, place: Place<'vir>) {
        let body = self.body;
        self.pending_regions_in.retain(|p| {
            !place.is_prefix_of(p.base)
                || (body.local_kind(p.base.local) == mir::LocalKind::Arg
                    && !body.local_decls[p.base.local].ty.is_ref())
        });
    }
}

/// `place` without trailing downcasts.
fn strip_downcasts(mut place: Place<'_>) -> Place<'_> {
    while let Some((parent, mir::ProjectionElem::Downcast(..))) = place.last_projection() {
        place = parent;
    }
    place
}
