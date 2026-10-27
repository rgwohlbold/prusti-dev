//! Calls to generic callees: a callee's contract names one piece per type
//! parameter occurrence (`RegionsIn(T)`, the positions `lo .. hi` of `T`),
//! while the caller holds one piece per lifetime occurrence of the
//! instantiation (its pieces under the occurrence's path, in position order).
//! Before the call the caller's pieces are merged into the callee's piece:
//! re-keyed to chain keys (the first piece's key, extended by the previous
//! pieces), cast to `p_Param_ind` of the argument's type, merged with
//! `merge_Param` (right to left) and, below the top of the argument, cast
//! back. After the call they are split again in reverse (`split_Param` left
//! to right) and re-keyed to their own keys (extended by their enclosing
//! pieces only).

use prusti_rustc_interface::middle::mir;
use task_encoder::TaskEncoder;
use vir::CastType;

use crate::encoders::{
    EncodeResult, ImpureEncVisitor,
    impure::fn_wand::GenericNode,
    ty::{
        RustTyDecomposition,
        generics::{
            GArgsTy, GArgsTyEnc,
            ind_casters::{IndCasters, IndCastersEnc},
        },
        indirect_node::{IndirectNodes, IndirectNodesEnc},
        indirect_pred::{ParamRanges, TyIndirectEnc, TyIndirectExt},
        lifted::TyConstructorEnc,
        range::Range,
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

    /// Merge (`into_callee`) or split the caller's pieces of a value of type
    /// `ty` (with snapshot `snap`) at the positions of the callee's `node`.
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
            if decomp.ty.specifics.is_param() || caller.find(&path).is_some_and(|c| caller.piece(c).all) {
                continue;
            }
            let group = (0..caller.len())
                .filter(|c| caller.piece(*c).path.starts_with(&path))
                .collect::<Vec<_>>();
            // At the top (`x: T`) the callee's piece is the cast of the
            // caller's, otherwise of the caller's type (exactly the caller's
            // piece if there is one).
            let top = path.is_empty();
            if group.len() == 1 && !top || group.is_empty() && !into_callee {
                continue;
            }
            let range = node.nodes.range(g);
            let c = self.group_casts(decomp)?;
            if group.is_empty() {
                // no positions: make the callee's empty piece
                if top {
                    let generic = c.generic(snap);
                    self.stmt((c.param.empty)(generic, &[c.x], &[], range.lo, range.hi).alloc());
                } else {
                    let key = node.nodes.key(g, snap);
                    self.stmt(self.vcx.mk_fold_stmt(caller.app_range(range, key, None)));
                }
            } else if into_callee {
                self.merge_pieces(&c, &caller, &group, snap, range, top);
            } else {
                let key = (!top).then(|| self.bind(node.nodes.key(g, snap)));
                self.split_pieces(&c, &caller, &group, snap, range, key);
            }
        }
        Ok(())
    }

    fn group_casts(&mut self, decomp: RustTyDecomposition<'vir>) -> EncodeResult<'vir, GroupCasts<'vir>, E> {
        let param = RustTyDecomposition::param();
        let param_ind = self.deps.require_ref::<TyIndirectEnc>(param)?;
        let constructor = self.deps.require_ref::<TyConstructorEnc>(decomp.ty)?.ty_constructor;
        let args = self.deps.require_dep::<GArgsTyEnc>(decomp.args)?;
        Ok(GroupCasts {
            casters: self.deps.require_ref::<IndCastersEnc>((param, decomp.ty))?,
            param: param_ind.param.unwrap(),
            param_ext: param_ind.ext,
            args,
            x: constructor(args.get_ty(), args.get_const()),
        })
    }

    /// Re-key the caller's piece `c` from `from` to `to` (unfold, fold).
    fn rekey(&mut self, caller: &IndirectNodes<'vir>, c: usize, from: vir::ExprSnap<'vir>, to: vir::ExprSnap<'vir>) {
        let range = caller.range(c);
        self.stmt(self.vcx.mk_unfold_stmt(caller.app_range(range, from, None)));
        self.stmt(self.vcx.mk_fold_stmt(caller.app_range(range, to, None)));
    }

    /// Before a call: merge the caller's pieces `group` (of a value with
    /// snapshot `snap`) into the callee's piece of `range`: of the caller's
    /// type (keyed by the first piece's key), or at the `top` its cast.
    fn merge_pieces(
        &mut self,
        c: &GroupCasts<'vir>,
        caller: &IndirectNodes<'vir>,
        group: &[usize],
        snap: vir::ExprSnap<'vir>,
        range: Range<'vir>,
        top: bool,
    ) {
        // the pieces' own keys, while all are held
        let own_keys = group.iter().map(|p| self.bind(caller.key(*p, snap))).collect::<Vec<_>>();
        // re-key to chain keys, in position order
        let mut chain = vec![own_keys[0]];
        for (i, p) in group.iter().enumerate().skip(1) {
            let prev = chain[i - 1];
            let key = self.bind(caller.ext(prev, prev, caller.range(group[i - 1])));
            self.rekey(caller, *p, own_keys[i], key);
            chain.push(key);
        }
        for (p, k) in group.iter().zip(&chain) {
            let r = caller.range(*p);
            self.stmt(c.make_generic(*k, r));
        }
        // right to left: `[l_i, l_i+1)` with the merged `[l_i+1, hi)`
        for (p, k) in group.iter().zip(&chain).rev().skip(1) {
            let r = caller.range(*p);
            self.stmt((c.param.merge)(c.generic(*k), &[c.x], &[], r.lo, r.hi, range.hi).alloc());
        }
        if !top {
            self.stmt(c.make_concrete(c.generic(chain[0]), range));
        }
    }

    /// After a call: split the callee's piece of `range` (keyed by `key`, or
    /// at the top the cast of the value) into the caller's pieces `group`.
    fn split_pieces(
        &mut self,
        c: &GroupCasts<'vir>,
        caller: &IndirectNodes<'vir>,
        group: &[usize],
        snap: vir::ExprSnap<'vir>,
        range: Range<'vir>,
        key: Option<vir::ExprSnap<'vir>>,
    ) {
        if let Some(key) = key {
            self.stmt(c.make_generic(key, range));
        }
        // left to right: `[l_i, hi)` into `[l_i, l_i+1)` and the rest
        let mut chain = vec![self.bind(c.generic(key.unwrap_or(snap)))];
        for p in &group[..group.len() - 1] {
            let (r, k) = (caller.range(*p), *chain.last().unwrap());
            self.stmt((c.param.split)(k, &[c.x], &[], r.lo, r.hi, range.hi).alloc());
            chain.push(self.bind((c.param_ext)(k, &[c.x], &[], k, r.lo, r.hi)));
        }
        for (p, k) in group.iter().zip(&chain) {
            self.stmt(c.make_concrete(*k, caller.range(*p)));
        }
        // re-key to the pieces' own keys, enclosing ones first
        for (p, k) in group.iter().zip(&chain).skip(1) {
            let own_key = self.bind(caller.key(*p, snap));
            self.rekey(caller, *p, c.casters.make_concrete_s.call()(k.downcast_ty()).upcast_ty(), own_key);
        }
    }

    /// A fresh variable holding `expr` (a key read before the pieces it reads
    /// are unfolded).
    pub(crate) fn bind(&mut self, expr: vir::ExprSnap<'vir>) -> vir::ExprSnap<'vir> {
        let tmp = self.new_tmp(expr.ty());
        self.stmt(self.vcx.mk_pure_assign_stmt(tmp, expr));
        tmp
    }
}

/// The casts and range methods for a group of pieces of the type `x`.
struct GroupCasts<'vir> {
    casters: IndCasters<'vir>,
    param: ParamRanges<'vir>,
    param_ext: TyIndirectExt<'vir>,
    args: GArgsTy<'vir>,
    x: vir::ExprTyVal<'vir>,
}

impl<'vir> GroupCasts<'vir> {
    /// `make_generic_s_X(k, ..)`
    fn generic(&self, k: vir::ExprSnap<'vir>) -> vir::ExprSnap<'vir> {
        (self.casters.make_generic_s)(k.downcast_ty(), self.args.get_ty(), self.args.get_const()).upcast_ty()
    }

    fn make_generic(&self, k: vir::ExprSnap<'vir>, r: Range<'vir>) -> vir::Stmt<'vir> {
        (self.casters.make_generic)(k, self.args.get_ty(), self.args.get_const(), r.lo, r.hi).alloc()
    }

    fn make_concrete(&self, p: vir::ExprSnap<'vir>, r: Range<'vir>) -> vir::Stmt<'vir> {
        (self.casters.make_concrete)(p, self.args.get_ty(), self.args.get_const(), r.lo, r.hi).alloc()
    }
}
