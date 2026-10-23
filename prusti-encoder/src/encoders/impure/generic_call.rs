//! Calls to generic callees: a callee's contract names one opaque piece per
//! type parameter position (`RegionsIn(T)`, query `path ++ q_all()`), while
//! the caller holds one piece per lifetime occurrence of the instantiation.
//! Before the call the caller's pieces under such a position are regrouped
//! into the callee's piece, after the call it is split again (with the keys
//! the callee returned).
//!
//! Both call a generated method per position (see `RegroupEnc`), which
//! converts one level of the type and calls the methods of the levels below.

use prusti_rustc_interface::middle::mir;
use task_encoder::TaskEncoder;
use vir::CastType;

use crate::encoders::{
    EncodeResult, ImpureEncVisitor,
    impure::fn_wand::GenericNode,
    ty::{
        RustTyDecomposition,
        generics::{GArgsTyEnc, GenericParamsEnc, ind_casters::IndCastersEnc, regroup::RegroupEnc},
        indirect_node::IndirectNodesEnc,
        lifetime_query::LifetimeQueryEnc,
        occurrences::Step,
    },
};

impl<'vir, E: TaskEncoder> ImpureEncVisitor<'vir, '_, E> {
    /// Before a call: regroup the pieces of the arguments at the callee's
    /// generic positions.
    pub(crate) fn generic_call_regroup(
        &mut self,
        generic: &[GenericNode<'vir>],
        args: &[(prusti_rustc_interface::middle::ty::Ty<'vir>, vir::ExprSnap<'vir>)],
    ) -> EncodeResult<'vir, (), E> {
        for node in generic.iter().filter(|n| n.blocked.is_some()) {
            let (ty, snap) = args[node.local.as_usize() - 1];
            self.convert_node(ty, snap, node, true)?;
        }
        Ok(())
    }

    /// After a call: split the pieces the callee returned at its generic
    /// positions (the arguments' at the label before the call, the result's
    /// now). Pieces blocked by the result come back through its wand.
    pub(crate) fn generic_call_split(
        &mut self,
        generic: &[GenericNode<'vir>],
        args: &[Option<mir::Place<'vir>>],
        dest: mir::Place<'vir>,
        label_pre: &'vir str,
    ) -> EncodeResult<'vir, (), E> {
        for node in generic {
            let (place, snap) = match node.blocked {
                Some(true) => continue,
                // the post-state of an argument: given back with the argument
                None if node.local != mir::RETURN_PLACE => continue,
                Some(false) => {
                    let Some(place) = args[node.local.as_usize() - 1] else { continue };
                    let snap = self.encode_place_with_snap(place.into())?.1;
                    (place, self.vcx.mk_local_labelled_old_expr(snap, label_pre))
                }
                None => (dest, self.encode_place_with_snap(dest.into())?.1),
            };
            let ty = place.ty(self.local_decls, self.vcx.tcx()).ty;
            self.convert_node(ty, snap, node, false)?;
        }
        Ok(())
    }

    /// Before applying the wand of a call: regroup the pieces of the result
    /// (of type `ty`, snapshot `snap` after the call) at the callee's generic
    /// positions.
    pub(crate) fn generic_wand_regroup(
        &mut self,
        generic: &[GenericNode<'vir>],
        ty: prusti_rustc_interface::middle::ty::Ty<'vir>,
        snap: vir::ExprSnap<'vir>,
    ) -> EncodeResult<'vir, (), E> {
        for node in generic.iter().filter(|n| n.blocked.is_none() && n.local == mir::RETURN_PLACE) {
            self.convert_node(ty, snap, node, true)?;
        }
        Ok(())
    }

    /// After applying the wand of a call: split the pieces it gave back at
    /// the callee's generic positions (of the arguments, typed and with their
    /// snapshots before the call).
    pub(crate) fn generic_wand_split(
        &mut self,
        generic: &[GenericNode<'vir>],
        args: &[(prusti_rustc_interface::middle::ty::Ty<'vir>, vir::ExprSnap<'vir>)],
    ) -> EncodeResult<'vir, (), E> {
        for node in generic.iter().filter(|n| n.blocked == Some(true)) {
            let (ty, snap) = args[node.local.as_usize() - 1];
            self.convert_node(ty, snap, node, false)?;
        }
        Ok(())
    }

    /// Regroup (`into_callee`) or split the caller's pieces of a value of
    /// type `ty` (with snapshot `snap`) at the positions of the callee's `node`.
    fn convert_node(
        &mut self,
        ty: prusti_rustc_interface::middle::ty::Ty<'vir>,
        snap: vir::ExprSnap<'vir>,
        node: &GenericNode<'vir>,
        into_callee: bool,
    ) -> EncodeResult<'vir, (), E> {
        let decomp = RustTyDecomposition::from_ty(ty, self.def_id);
        let caller = self.deps.require_dep::<IndirectNodesEnc>(decomp)?;
        for g in node.nodes.pieces_of(node.node) {
            let path = node.nodes.piece(g).path.clone();
            // generic in the caller as well: the same piece
            if caller
                .find(&path)
                .is_some_and(|c| caller.piece(c).all)
            {
                continue;
            }
            if path.is_empty() {
                // the whole value is generic (`x: T`): through the cast of the
                // concrete type (as its direct predicate)
                self.convert_top(decomp, snap, into_callee)?;
                continue;
            }
            let key = self.bind(node.nodes.key(g, snap));
            self.call_regroup(decomp, path, into_callee, key)?;
        }
        Ok(())
    }

    /// A whole value at a generic position: `q_all()` of its concrete type,
    /// cast to `p_Param_ind(make_generic_s(snap), .., q_all())`.
    fn convert_top(
        &mut self,
        c: RustTyDecomposition<'vir>,
        snap: vir::ExprSnap<'vir>,
        into_callee: bool,
    ) -> EncodeResult<'vir, (), E> {
        if c.ty.specifics.is_param() {
            return Ok(());
        }
        let casters = self
            .deps
            .require_ref::<IndCastersEnc>((RustTyDecomposition::param(), c.ty))?;
        let args = self.deps.require_dep::<GArgsTyEnc>(c.args)?;
        let query = self.deps.require_dep::<LifetimeQueryEnc>(())?;
        let snap = self.bind(snap);
        if into_callee {
            self.call_regroup(c, Vec::new(), true, snap)?;
            self.stmt((casters.make_generic)(snap, args.get_ty(), args.get_const(), query.all()).alloc());
        } else {
            let generic = (casters.make_generic_s)(snap.downcast_ty(), args.get_ty(), args.get_const());
            self.stmt(
                (casters.make_concrete)(generic.upcast_ty(), args.get_ty(), args.get_const(), query.all())
                    .alloc(),
            );
            self.call_regroup(c, Vec::new(), false, snap)?;
        }
        Ok(())
    }

    /// The call of the regroup (`into`) or split method of the position
    /// `rest` of `x`, on the key `k`.
    fn call_regroup(
        &mut self,
        x: RustTyDecomposition<'vir>,
        rest: Vec<Step>,
        into: bool,
        k: vir::ExprSnap<'vir>,
    ) -> EncodeResult<'vir, (), E> {
        let method = self.deps.require_ref::<RegroupEnc>((x, rest, into))?.method;
        let generics = self.deps.require_dep::<GenericParamsEnc>(x.args.context())?;
        let tys = self.vcx.alloc_slice(generics.ty_exprs());
        let consts = self.vcx.alloc_slice(generics.const_exprs());
        self.stmt(method(k, tys, consts).alloc());
        Ok(())
    }

    /// A fresh variable holding `expr` (a key read before the pieces it reads
    /// are unfolded).
    pub(crate) fn bind(&mut self, expr: vir::ExprSnap<'vir>) -> vir::ExprSnap<'vir> {
        let tmp = self.new_tmp(expr.ty());
        self.stmt(self.vcx.mk_pure_assign_stmt(tmp, expr));
        tmp
    }
}
