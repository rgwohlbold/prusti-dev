//! Calls to generic callees: a callee's contract names one opaque piece per
//! type parameter position (`RegionsIn(T)`, query `path ++ q_all()`), while
//! the caller holds one piece per lifetime occurrence of the instantiation.
//! Before the call the caller's pieces under such a position are regrouped
//! into the callee's piece, after the call it is split again (with the keys
//! the callee returned).
//!
//! Both follow the constructors on the path to the position: the caller's
//! pieces are unfolded (`q_in(j, ..)` holds the parts where `T_j` occurs in
//! the fields, see `TyParamPart`), cast to the concrete instantiation where a
//! field is `T_j` itself, regrouped there (`q_all()` of a concrete type is the
//! conjunction of the pieces of its own parameters), cast back and folded.

use prusti_rustc_interface::middle::{mir, ty};
use task_encoder::TaskEncoder;
use vir::CastType;

use crate::encoders::{
    EncodeResult, ImpureEncVisitor,
    impure::fn_wand::GenericNode,
    ty::{
        RustTyDecomposition,
        generics::{GArgsTyEnc, ind_casters::IndCastersEnc},
        indirect_node::{IndirectNodes, IndirectNodesEnc, NodeId},
        indirect_pred::TyIndirectEnc,
        lifetime_query::{LifetimeQueryEnc, LifetimeQueryFns},
        occurrences::{End, Step, occurrence_query, occurrences, param_position},
    },
};

/// A caller's piece at one level of the type: its key and its path relative
/// to the level's type.
#[derive(Clone)]
struct LevelPiece<'vir> {
    key: vir::ExprSnap<'vir>,
    path: Vec<Step>,
    /// A `RegionsIn(T)` piece of the caller (query `path ++ q_all()`).
    all: bool,
}

impl<'vir, E: TaskEncoder> ImpureEncVisitor<'vir, '_, E> {
    /// Before a call: regroup the pieces of the arguments at the callee's
    /// generic positions.
    pub(crate) fn generic_call_regroup(
        &mut self,
        generic: &[GenericNode<'vir>],
        args: &[(ty::Ty<'vir>, vir::ExprSnap<'vir>)],
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
        ty: ty::Ty<'vir>,
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
        args: &[(ty::Ty<'vir>, vir::ExprSnap<'vir>)],
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
        ty: ty::Ty<'vir>,
        snap: vir::ExprSnap<'vir>,
        node: &GenericNode<'vir>,
        into_callee: bool,
    ) -> EncodeResult<'vir, (), E> {
        let decomp = RustTyDecomposition::from_ty(ty, self.def_id);
        let caller = self.deps.require_dep::<IndirectNodesEnc>(decomp)?;
        for g in node.nodes.pieces_of_node(node.node) {
            let path = node.nodes.piece(g).path.clone();
            // generic in the caller as well: the same piece
            if caller
                .find(&path)
                .is_some_and(|c| matches!(caller.piece(c).node, NodeId::RegionsIn(_)))
            {
                continue;
            }
            let pieces = self.caller_pieces(&caller, &path, snap);
            if path.is_empty() {
                // the whole value is generic (`x: T`): through the cast of the
                // concrete type (as its direct predicate)
                self.convert_top(decomp, snap, pieces, into_callee)?;
                continue;
            }
            let key = self.bind(node.nodes.key(g, snap));
            if into_callee {
                self.regroup_level(decomp, key, &path, pieces)?;
            } else {
                self.split_level(decomp, key, &path, pieces)?;
            }
        }
        Ok(())
    }

    /// A whole value at a generic position: `q_all()` of its concrete type,
    /// cast to `p_Param_ind(make_generic_s(snap), .., q_all())`.
    fn convert_top(
        &mut self,
        c: RustTyDecomposition<'vir>,
        snap: vir::ExprSnap<'vir>,
        pieces: Vec<LevelPiece<'vir>>,
        into_callee: bool,
    ) -> EncodeResult<'vir, (), E> {
        if c.ty.specifics.is_param() {
            return Ok(());
        }
        let casters = self
            .deps
            .require_ref::<IndCastersEnc>((RustTyDecomposition::param(), c.ty))?;
        let args = self.deps.require_dep::<GArgsTyEnc>(c.args)?;
        let query = self.query_fns()?;
        let snap = self.bind(snap);
        if into_callee {
            self.regroup_all(c, snap, pieces)?;
            self.stmt((casters.make_generic)(snap, args.get_ty(), args.get_const(), query.all()).alloc());
        } else {
            let generic = (casters.make_generic_s)(snap.downcast_ty(), args.get_ty(), args.get_const());
            self.stmt(
                (casters.make_concrete)(generic.upcast_ty(), args.get_ty(), args.get_const(), query.all())
                    .alloc(),
            );
            self.split_all(c, snap, pieces)?;
        }
        Ok(())
    }

    /// The caller's pieces under `path` (whole paths), keys bound now.
    fn caller_pieces(
        &mut self,
        caller: &IndirectNodes<'vir>,
        path: &[Step],
        snap: vir::ExprSnap<'vir>,
    ) -> Vec<LevelPiece<'vir>> {
        (0..caller.len())
            .filter(|c| caller.piece(*c).path.starts_with(path))
            .map(|c| LevelPiece {
                key: self.bind(caller.key(c, snap)),
                path: caller.piece(c).path.clone(),
                all: matches!(caller.piece(c).node, NodeId::RegionsIn(_)),
            })
            .collect()
    }

    /// A fresh variable holding `expr` (a key read before the pieces it reads
    /// are unfolded).
    pub(crate) fn bind(&mut self, expr: vir::ExprSnap<'vir>) -> vir::ExprSnap<'vir> {
        let tmp = self.new_tmp(expr.ty());
        self.stmt(self.vcx.mk_pure_assign_stmt(tmp, expr));
        tmp
    }

    fn query_fns(&mut self) -> EncodeResult<'vir, LifetimeQueryFns<'vir>, E> {
        Ok(self.deps.require_dep::<LifetimeQueryEnc>(())?)
    }

    /// `p_X_ind(key, .., Q(path))`, `Q(path ++ q_all())` if `all`.
    fn level_app(
        &mut self,
        x: RustTyDecomposition<'vir>,
        key: vir::ExprSnap<'vir>,
        path: &[Step],
        all: bool,
    ) -> EncodeResult<'vir, vir::PredicateApp<'vir>, E> {
        let query = self.query_fns()?;
        let ind = self.deps.require_ref::<TyIndirectEnc>(x.ty)?;
        let args = self.deps.require_dep::<GArgsTyEnc>(x.args)?;
        let rest = all.then(|| query.all());
        let q = occurrence_query(&query, path, rest);
        Ok((ind.pred)(key, args.get_ty(), args.get_const(), q)(None))
    }

    /// Does the (caller's, concrete) type reach anything through mutable
    /// references (`has_ind`)?
    fn has_pieces(decomp: RustTyDecomposition<'vir>) -> bool {
        !occurrences(decomp).unwrap_or_default().is_empty()
    }

    /// Build `p_X_ind(k, .., rest ++ q_all())` from the caller's `pieces` at
    /// this level (paths starting with `rest`).
    fn regroup_level(
        &mut self,
        x: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        rest: &[Step],
        pieces: Vec<LevelPiece<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        let Some((Step::In(j), r)) = rest.split_first().map(|(s, r)| (*s, r)) else {
            return self.regroup_all(x, k, pieces);
        };
        for p in &pieces {
            let app = self.level_app(x, p.key, &p.path, p.all)?;
            self.stmt(self.vcx.mk_unfold_stmt(app));
        }
        self.parts(x, k, j, r, &pieces, true)?;
        let app = self.level_app(x, k, rest, true)?;
        self.stmt(self.vcx.mk_fold_stmt(app));
        Ok(())
    }

    /// Split `p_X_ind(k, .., rest ++ q_all())` into the caller's `pieces`.
    fn split_level(
        &mut self,
        x: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        rest: &[Step],
        pieces: Vec<LevelPiece<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        let Some((Step::In(j), r)) = rest.split_first().map(|(s, r)| (*s, r)) else {
            return self.split_all(x, k, pieces);
        };
        let app = self.level_app(x, k, rest, true)?;
        self.stmt(self.vcx.mk_unfold_stmt(app));
        self.parts(x, k, j, r, &pieces, false)?;
        for p in &pieces {
            let app = self.level_app(x, p.key, &p.path, p.all)?;
            self.stmt(self.vcx.mk_fold_stmt(app));
        }
        Ok(())
    }

    /// The parts of `X`'s body for `q_in(j, ..)`: per field where `T_j`
    /// occurs, convert the field's pieces one level down.
    fn parts(
        &mut self,
        x: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        j: usize,
        r: &[Step],
        pieces: &[LevelPiece<'vir>],
        into_callee: bool,
    ) -> EncodeResult<'vir, (), E> {
        let dep = self.deps.require_dep::<TyIndirectEnc>(x.ty)?;
        for part in dep.ty_param_parts[j].clone() {
            let stmts = self.block(|v| {
                let sub_k = part.read.read(k);
                let sub_pieces = pieces
                    .iter()
                    .map(|p| LevelPiece {
                        key: part.read.read(p.key),
                        path: part.path.iter().chain(&p.path[1..]).copied().collect(),
                        all: p.all,
                    })
                    .collect::<Vec<_>>();
                if part.path.is_empty() {
                    // the field is `T_j`: its instantiation, through the casts
                    let arg = x.args.args().iter().filter_map(|a| a.as_type()).nth(j).unwrap();
                    let c = RustTyDecomposition::from_ty(arg, x.args.context());
                    if c.ty.specifics.is_param() || !Self::has_pieces(c) {
                        return Ok(());
                    }
                    v.cast_level(c, sub_k, r, sub_pieces, into_callee)
                } else {
                    let f = part.field.decompose_normalize(x.args);
                    let rest = part.path.iter().chain(r).copied().collect::<Vec<_>>();
                    if into_callee {
                        v.regroup_level(f, sub_k, &rest, sub_pieces)
                    } else {
                        v.split_level(f, sub_k, &rest, sub_pieces)
                    }
                }
            })?;
            let stmt = match part.variant {
                Some((discr, value)) => {
                    let cond = self.vcx.mk_eq_expr(discr.call()(k.downcast_ty()), value);
                    self.vcx.mk_if_stmt(cond, self.vcx.alloc_slice(&stmts), &[])
                }
                None => {
                    self.stmts(stmts);
                    continue;
                }
            };
            self.stmt(stmt);
        }
        Ok(())
    }

    /// At a field of type `T_j` instantiated with `c`: cast the generic pieces
    /// to concrete, convert at `c`, and cast back.
    fn cast_level(
        &mut self,
        c: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        r: &[Step],
        pieces: Vec<LevelPiece<'vir>>,
        into_callee: bool,
    ) -> EncodeResult<'vir, (), E> {
        let casters = self
            .deps
            .require_ref::<IndCastersEnc>((RustTyDecomposition::param(), c.ty))?;
        let args = self.deps.require_dep::<GArgsTyEnc>(c.args)?;
        let query = self.query_fns()?;
        let concrete = |key: vir::ExprSnap<'vir>| -> vir::ExprSnap<'vir> {
            casters.make_concrete_s.call()(key.downcast_ty()).upcast_ty()
        };
        let c_k = concrete(k);
        let c_pieces = pieces
            .iter()
            .map(|p| LevelPiece { key: concrete(p.key), path: p.path.clone(), all: p.all })
            .collect::<Vec<_>>();
        let rest_all = occurrence_query(&query, r, Some(query.all()));
        if into_callee {
            // enclosing pieces first (their keys are read before)
            for p in &pieces {
                let q = occurrence_query(&query, &p.path, p.all.then(|| query.all()));
                self.stmt((casters.make_concrete)(p.key, args.get_ty(), args.get_const(), q).alloc());
            }
            self.regroup_level(c, c_k, r, c_pieces)?;
            self.stmt((casters.make_generic)(c_k, args.get_ty(), args.get_const(), rest_all).alloc());
        } else {
            self.stmt((casters.make_concrete)(k, args.get_ty(), args.get_const(), rest_all).alloc());
            self.split_level(c, c_k, r, c_pieces.clone())?;
            // enclosed pieces first: their keys read the enclosing concrete ones
            for p in c_pieces.iter().rev() {
                let q = occurrence_query(&query, &p.path, p.all.then(|| query.all()));
                self.stmt((casters.make_generic)(p.key, args.get_ty(), args.get_const(), q).alloc());
            }
        }
        Ok(())
    }

    /// The parts of `q_all()` of the concrete `c` (keyed through the
    /// enclosing ones): the pieces of its lifetime parameters are the
    /// caller's; those inside its type parameters are converted.
    fn all_parts(
        &mut self,
        c: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
    ) -> EncodeResult<'vir, Vec<(usize, vir::ExprSnap<'vir>)>, E> {
        let query = self.query_fns()?;
        let ind = self.deps.require_ref::<TyIndirectEnc>(c.ty)?;
        let args = self.deps.require_dep::<GArgsTyEnc>(c.args)?;
        let all = occurrences(RustTyDecomposition::identity(c.ty)).unwrap_or_default();
        let mut keys: Vec<vir::ExprSnap<'vir>> = Vec::new();
        let mut params = Vec::new();
        for o in &all {
            let mut key = k;
            for parent in &o.parents {
                let q = occurrence_query(&query, &all[*parent].path, Some(query.all()));
                key = (ind.ext)(key, args.get_ty(), args.get_const(), keys[*parent], q);
            }
            keys.push(key);
            if let End::Param(param) = o.end {
                let j = param_position(c.ty.params, param.index, |a| a.as_type().is_some());
                params.push((j, key));
            }
        }
        Ok(params)
    }

    fn regroup_all(
        &mut self,
        c: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        pieces: Vec<LevelPiece<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        for (j, key) in self.all_parts(c, k)? {
            if !self.param_has_pieces(c, j) {
                continue;
            }
            let key = self.bind(key);
            let sub = pieces
                .iter()
                .filter(|p| p.path.first() == Some(&Step::In(j)))
                .cloned()
                .collect();
            self.regroup_level(c, key, &[Step::In(j)], sub)?;
        }
        let app = self.level_app(c, k, &[], true)?;
        self.stmt(self.vcx.mk_fold_stmt(app));
        Ok(())
    }

    fn split_all(
        &mut self,
        c: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        pieces: Vec<LevelPiece<'vir>>,
    ) -> EncodeResult<'vir, (), E> {
        let app = self.level_app(c, k, &[], true)?;
        self.stmt(self.vcx.mk_unfold_stmt(app));
        // the parts' keys read the enclosing parts, unfolded now
        for (j, key) in self.all_parts(c, k)? {
            if !self.param_has_pieces(c, j) {
                continue;
            }
            let key = self.bind(key);
            let sub = pieces
                .iter()
                .filter(|p| p.path.first() == Some(&Step::In(j)))
                .cloned()
                .collect();
            self.split_level(c, key, &[Step::In(j)], sub)?;
        }
        Ok(())
    }

    /// Does `c`'s type argument `j` reach anything through mutable references?
    fn param_has_pieces(&self, c: RustTyDecomposition<'vir>, j: usize) -> bool {
        let arg = c.args.args().iter().filter_map(|a| a.as_type()).nth(j).unwrap();
        Self::has_pieces(RustTyDecomposition::from_ty(arg, c.args.context()))
    }
}
