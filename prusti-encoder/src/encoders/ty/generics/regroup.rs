//! The methods converting between a caller's pieces under a callee's generic
//! position and the callee's piece there (design doc, "Generic Calls Need
//! Regrouping"):
//! ```text
//! method regroup_ind_X_..(k: s_X, C..)
//!   requires p_X_ind(key_o(k), .., Q_o)  // per occurrence o under `rest`
//!   ensures  p_X_ind(k, .., rest ++ q_all())
//! ```
//! and `split_ind_X_..` for the reverse direction. `C..` are the generic
//! parameters of the context of `X`'s arguments (the caller's). The key of an
//! occurrence is `k` extended by the occurrences under `rest` enclosing it.
//!
//! Each method converts one level of the type (unfold, convert the parts
//! where the type parameter occurs, fold) and calls the methods of the levels
//! below; at a field of type `T_j` itself it casts to the instantiation and
//! back. Recursive types thus give (mutually) recursive methods. A position
//! without occurrences holds nothing; its methods have no body.

use std::hash::{Hash, Hasher};

use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, MethodIdn};

use crate::encoders::ty::{
    RustTyDecomposition,
    generics::{GArgsTyEnc, GenericParamsEnc, ind_casters::IndCastersEnc},
    indirect_pred::TyIndirectEnc,
    lifetime_query::{LifetimeQueryEnc, LifetimeQueryFns},
    occurrences::{End, Step, occurrence_query, occurrences, param_position},
    pure::TyPureEnc,
};

pub(crate) struct RegroupEnc;

/// `(x, rest, into)`: regroup (`into`) or split the pieces of `x` under the
/// position `rest`.
pub(crate) type RegroupTask<'vir> = (RustTyDecomposition<'vir>, Vec<Step>, bool);

pub(crate) type RegroupMethod<'vir> =
    MethodIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap)>;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RegroupRef<'vir> {
    pub(crate) method: RegroupMethod<'vir>,
}

impl task_encoder::OutputRefAny for RegroupRef<'_> {}

impl TaskEncoder for RegroupEnc {
    task_encoder::encoder_cache!(RegroupEnc);
    const ENCODER_NAME: &'static str = "regroup encoder";

    type TaskDescription<'vir> = RegroupTask<'vir>;
    type OutputRef<'vir> = RegroupRef<'vir>;
    type OutputFullLocal<'vir> = vir::Method<'vir>;
    type EncodingError = String;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        task.clone()
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        let (x, rest, into) = task_key.clone();
        let generics = deps.require_dep::<GenericParamsEnc>(x.args.context())?;
        let snap = deps.require_ref::<TyPureEnc>(x.ty)?.snapshot;
        let query = deps.require_dep::<LifetimeQueryEnc>(())?;
        vir::with_vcx(|vcx| {
            let name = method_name(&x, &rest, into);
            let method = MethodIdn::new(
                vir::vir_format_identifier!(vcx, "{name}"),
                (snap, generics.ty_args(), generics.const_args()),
            );
            deps.emit_output_ref(task_key.clone(), RegroupRef { method })?;

            let k_decl = vcx.mk_local_decl("k", snap);
            let mut builder = Builder {
                vcx,
                deps,
                query,
                ctx_tys: vcx.alloc_slice(generics.ty_exprs()),
                ctx_consts: vcx.alloc_slice(generics.const_exprs()),
                stmts: Vec::new(),
                locals: 0,
            };
            let k = vcx.mk_local_ex(k_decl);
            let pieces = builder.pieces(x, &rest, k)?;
            let apps = pieces
                .iter()
                .map(|p| Ok(vcx.mk_predicate_app_expr(builder.app(x, p.key, &p.path, p.all)?)))
                .collect::<EncResult<'vir, Vec<_>>>()?;
            let target = vcx.mk_predicate_app_expr(builder.app(x, k, &rest, true)?);
            // Nothing under the position (e.g. `T := i32`): the predicate holds
            // nothing, so the method is trusted (as `ind_empty`).
            if pieces.is_empty() {
                let (pres, posts) = if into { (vec![], vec![target]) } else { (vec![target], vec![]) };
                return Ok((
                    vcx.mk_method(
                        method,
                        (k_decl, generics.ty_decls(), generics.const_decls()),
                        &[],
                        vcx.alloc_slice(&pres),
                        vcx.alloc_slice(&posts),
                        None,
                    ),
                    (),
                ));
            }
            if into {
                builder.regroup(x, k, &rest, &pieces)?;
            } else {
                builder.split(x, k, &rest, &pieces)?;
            }
            let (pres, posts) = if into {
                (apps, vec![target])
            } else {
                (vec![target], apps)
            };
            let body = vcx.mk_cfg_block(
                &vir::CfgBlockLabelData::Start,
                &[],
                vcx.alloc_slice(&builder.stmts),
                vcx.alloc(vir::TerminatorStmtData::Exit),
            );
            Ok((
                vcx.mk_method(
                    method,
                    (k_decl, generics.ty_decls(), generics.const_decls()),
                    &[],
                    vcx.alloc_slice(&pres),
                    vcx.alloc_slice(&posts),
                    Some(vcx.alloc_slice(&[body])),
                ),
                (),
            ))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for method in Self::all_outputs_local_no_errors(program) {
            program.add_method(method);
        }
    }
}

/// A unique name per task (the hash distinguishes instantiations and
/// contexts; `DefaultHasher::new` is deterministic).
fn method_name(x: &RustTyDecomposition<'_>, rest: &[Step], into: bool) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{x:?} {rest:?}").hash(&mut hasher);
    let dir = if into { "regroup" } else { "split" };
    format!("{dir}_ind_{}_{:016x}", x.ty.name(), hasher.finish())
}

/// A piece of the level's type: its key and its path in the type.
#[derive(Clone)]
struct Piece<'vir> {
    key: vir::ExprSnap<'vir>,
    path: Vec<Step>,
    /// A `RegionsIn(T)` piece (query `path ++ q_all()`).
    all: bool,
}

type EncResult<'vir, T> = Result<T, EncodeFullError<'vir, RegroupEnc>>;

struct Builder<'a, 'vir> {
    vcx: &'vir vir::VirCtxt<'vir>,
    deps: &'a mut TaskEncoderDependencies<'vir, RegroupEnc>,
    query: LifetimeQueryFns<'vir>,
    /// The generic parameters of the context (the method's).
    ctx_tys: &'vir [vir::ExprTyVal<'vir>],
    ctx_consts: &'vir [vir::ExprCSnap<'vir>],
    stmts: Vec<vir::Stmt<'vir>>,
    locals: usize,
}

impl<'vir> Builder<'_, 'vir> {
    /// The pieces of `x` under `rest`, keyed from `k`: extended by the
    /// pieces under `rest` enclosing them (the others are in `k` already).
    fn pieces(
        &mut self,
        x: RustTyDecomposition<'vir>,
        rest: &[Step],
        k: vir::ExprSnap<'vir>,
    ) -> EncResult<'vir, Vec<Piece<'vir>>> {
        let all = occurrences(x).map_err(|err| EncodeFullError::EncodingError(err, None))?;
        let mut keys: Vec<Option<vir::ExprSnap<'vir>>> = vec![None; all.len()];
        let mut pieces = Vec::new();
        for (idx, occurrence) in all.iter().enumerate() {
            if !occurrence.path.starts_with(rest) {
                continue;
            }
            let mut key = k;
            for parent in &occurrence.parents {
                if let Some(parent_key) = keys[*parent] {
                    let parent = &all[*parent];
                    let q = self.piece_query(&parent.path, matches!(parent.end, End::Param(_)));
                    key = self.ext(x, key, parent_key, q)?;
                }
            }
            keys[idx] = Some(key);
            pieces.push(Piece {
                key,
                path: occurrence.path.clone(),
                all: matches!(occurrence.end, End::Param(_)),
            });
        }
        Ok(pieces)
    }

    fn piece_query(&self, path: &[Step], all: bool) -> vir::ExprLifetimeQuery<'vir> {
        occurrence_query(&self.query, path, all.then(|| self.query.all()))
    }

    /// `p_X_ind(key, .., Q(path))`, `Q(path ++ q_all())` if `all`.
    fn app(
        &mut self,
        x: RustTyDecomposition<'vir>,
        key: vir::ExprSnap<'vir>,
        path: &[Step],
        all: bool,
    ) -> EncResult<'vir, vir::PredicateApp<'vir>> {
        let ind = self.deps.require_ref::<TyIndirectEnc>(x.ty)?;
        let args = self.deps.require_dep::<GArgsTyEnc>(x.args)?;
        let q = self.piece_query(path, all);
        Ok((ind.pred)(key, args.get_ty(), args.get_const(), q)(None))
    }

    fn ext(
        &mut self,
        x: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        kp: vir::ExprSnap<'vir>,
        q: vir::ExprLifetimeQuery<'vir>,
    ) -> EncResult<'vir, vir::ExprSnap<'vir>> {
        let ind = self.deps.require_ref::<TyIndirectEnc>(x.ty)?;
        let args = self.deps.require_dep::<GArgsTyEnc>(x.args)?;
        Ok((ind.ext)(k, args.get_ty(), args.get_const(), kp, q))
    }

    fn stmt(&mut self, stmt: vir::Stmt<'vir>) {
        self.stmts.push(stmt);
    }

    /// A fresh local holding `expr` (a key read before the pieces it reads
    /// are unfolded).
    fn bind(&mut self, expr: vir::ExprSnap<'vir>) -> vir::ExprSnap<'vir> {
        let name = vir::vir_format!(self.vcx, "key_{}", self.locals);
        self.locals += 1;
        let decl = self.vcx.mk_local_decl(name, expr.ty());
        self.stmt(self.vcx.mk_local_decl_stmt(decl, Some(expr)));
        self.vcx.mk_local_ex(decl)
    }

    /// The statements `f` adds, separately.
    fn block(
        &mut self,
        f: impl FnOnce(&mut Self) -> EncResult<'vir, ()>,
    ) -> EncResult<'vir, Vec<vir::Stmt<'vir>>> {
        let outer = std::mem::take(&mut self.stmts);
        let result = f(self);
        let inner = std::mem::replace(&mut self.stmts, outer);
        result.map(|()| inner)
    }

    /// The call of the method of `(x, rest, into)` on `k`.
    fn call(
        &mut self,
        x: RustTyDecomposition<'vir>,
        rest: Vec<Step>,
        into: bool,
        k: vir::ExprSnap<'vir>,
    ) -> EncResult<'vir, ()> {
        let method = self.deps.require_ref::<RegroupEnc>((x, rest, into))?.method;
        self.stmt(method(k, self.ctx_tys, self.ctx_consts).alloc());
        Ok(())
    }

    /// The body of a regroup method.
    fn regroup(
        &mut self,
        x: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        rest: &[Step],
        pieces: &[Piece<'vir>],
    ) -> EncResult<'vir, ()> {
        let Some((Step::In(j), r)) = rest.split_first().map(|(s, r)| (*s, r)) else {
            return self.regroup_all(x, k);
        };
        // the keys read the pieces they are extended by
        let pieces = pieces
            .iter()
            .map(|p| Piece { key: self.bind(p.key), ..p.clone() })
            .collect::<Vec<_>>();
        for p in &pieces {
            let app = self.app(x, p.key, &p.path, p.all)?;
            self.stmt(self.vcx.mk_unfold_stmt(app));
        }
        self.parts(x, k, j, r, &pieces, true)?;
        let app = self.app(x, k, rest, true)?;
        self.stmt(self.vcx.mk_fold_stmt(app));
        Ok(())
    }

    /// The body of a split method.
    fn split(
        &mut self,
        x: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        rest: &[Step],
        pieces: &[Piece<'vir>],
    ) -> EncResult<'vir, ()> {
        let Some((Step::In(j), r)) = rest.split_first().map(|(s, r)| (*s, r)) else {
            return self.split_all(x, k);
        };
        let app = self.app(x, k, rest, true)?;
        self.stmt(self.vcx.mk_unfold_stmt(app));
        self.parts(x, k, j, r, pieces, false)?;
        for p in pieces {
            let app = self.app(x, p.key, &p.path, p.all)?;
            self.stmt(self.vcx.mk_fold_stmt(app));
        }
        Ok(())
    }

    /// The parts of `X`'s body for `q_arg(j, ..)`: per field where `T_j`
    /// occurs, convert the field's pieces one level down.
    fn parts(
        &mut self,
        x: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        j: usize,
        r: &[Step],
        pieces: &[Piece<'vir>],
        into: bool,
    ) -> EncResult<'vir, ()> {
        let dep = self.deps.require_dep::<TyIndirectEnc>(x.ty)?;
        for part in dep.ty_param_parts[j].clone() {
            let stmts = self.block(|b| {
                let sub_k = part.read.read(k);
                if part.path.is_empty() {
                    // the field is `T_j`: its instantiation, through the casts
                    let arg = x.args.args().iter().filter_map(|a| a.as_type()).nth(j).unwrap();
                    let c = RustTyDecomposition::from_ty(arg, x.args.context());
                    if c.ty.specifics.is_param() || !has_pieces(c) {
                        return Ok(());
                    }
                    let sub_pieces = pieces
                        .iter()
                        .map(|p| Piece {
                            key: part.read.read(p.key),
                            path: p.path[1..].to_vec(),
                            all: p.all,
                        })
                        .collect::<Vec<_>>();
                    b.cast(c, sub_k, r, &sub_pieces, into)
                } else {
                    let f = part.field.decompose_normalize(x.args);
                    let rest = part.path.iter().chain(r).copied().collect::<Vec<_>>();
                    b.call(f, rest, into, sub_k)
                }
            })?;
            match part.variant {
                Some((discr, value)) => {
                    let cond = self.vcx.mk_eq_expr(discr.call()(k.downcast_ty()), value);
                    self.stmt(self.vcx.mk_if_stmt(cond, self.vcx.alloc_slice(&stmts), &[]));
                }
                None => self.stmts.extend(stmts),
            }
        }
        Ok(())
    }

    /// At a field of type `T_j` instantiated with `c`: cast the pieces to
    /// concrete, convert at `c`, and cast back.
    fn cast(
        &mut self,
        c: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        r: &[Step],
        pieces: &[Piece<'vir>],
        into: bool,
    ) -> EncResult<'vir, ()> {
        let casters = self
            .deps
            .require_ref::<IndCastersEnc>((RustTyDecomposition::param(), c.ty))?;
        let args = self.deps.require_dep::<GArgsTyEnc>(c.args)?;
        let concrete = |key: vir::ExprSnap<'vir>| -> vir::ExprSnap<'vir> {
            casters.make_concrete_s.call()(key.downcast_ty()).upcast_ty()
        };
        let c_k = concrete(k);
        let rest_all = self.piece_query(r, true);
        if into {
            // enclosing pieces first (their keys are read before)
            for p in pieces {
                let q = self.piece_query(&p.path, p.all);
                self.stmt((casters.make_concrete)(p.key, args.get_ty(), args.get_const(), q).alloc());
            }
            self.call(c, r.to_vec(), true, c_k)?;
            self.stmt((casters.make_generic)(c_k, args.get_ty(), args.get_const(), rest_all).alloc());
        } else {
            self.stmt((casters.make_concrete)(k, args.get_ty(), args.get_const(), rest_all).alloc());
            self.call(c, r.to_vec(), false, c_k)?;
            // enclosed pieces first: their keys read the enclosing concrete ones
            for p in pieces.iter().rev() {
                let q = self.piece_query(&p.path, p.all);
                self.stmt(
                    (casters.make_generic)(concrete(p.key), args.get_ty(), args.get_const(), q)
                        .alloc(),
                );
            }
        }
        Ok(())
    }

    /// The parts of `q_all()` of the concrete `c` inside its type parameters:
    /// `(j, key)`, keyed through the enclosing parts.
    fn all_parts(
        &mut self,
        c: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
    ) -> EncResult<'vir, Vec<(usize, vir::ExprSnap<'vir>)>> {
        let identity = RustTyDecomposition::identity(c.ty);
        let all = occurrences(identity).unwrap_or_default();
        let mut keys: Vec<vir::ExprSnap<'vir>> = Vec::new();
        let mut params = Vec::new();
        for o in &all {
            let mut key = k;
            for parent in &o.parents {
                let q = self.piece_query(&all[*parent].path, true);
                key = self.ext(c, key, keys[*parent], q)?;
            }
            keys.push(key);
            if let End::Param(param) = o.end {
                let j = param_position(c.ty.params, param.index, |a| a.as_type().is_some());
                params.push((j, key));
            }
        }
        Ok(params)
    }

    /// `q_all()` of the concrete `x` from its own pieces.
    fn regroup_all(&mut self, x: RustTyDecomposition<'vir>, k: vir::ExprSnap<'vir>) -> EncResult<'vir, ()> {
        for (j, key) in self.all_parts(x, k)? {
            if !param_has_pieces(x, j) {
                continue;
            }
            let key = self.bind(key);
            self.call(x, vec![Step::In(j)], true, key)?;
        }
        let app = self.app(x, k, &[], true)?;
        self.stmt(self.vcx.mk_fold_stmt(app));
        Ok(())
    }

    /// The own pieces of the concrete `x` from its `q_all()`.
    fn split_all(&mut self, x: RustTyDecomposition<'vir>, k: vir::ExprSnap<'vir>) -> EncResult<'vir, ()> {
        let app = self.app(x, k, &[], true)?;
        self.stmt(self.vcx.mk_unfold_stmt(app));
        // the parts' keys read the enclosing parts, unfolded now
        for (j, key) in self.all_parts(x, k)? {
            if !param_has_pieces(x, j) {
                continue;
            }
            let key = self.bind(key);
            self.call(x, vec![Step::In(j)], false, key)?;
        }
        Ok(())
    }
}

/// Does the (concrete) type reach anything through mutable references?
pub(crate) fn has_pieces(decomp: RustTyDecomposition<'_>) -> bool {
    !occurrences(decomp).unwrap_or_default().is_empty()
}

/// Does `c`'s type argument `j` reach anything through mutable references?
fn param_has_pieces(c: RustTyDecomposition<'_>, j: usize) -> bool {
    let arg = c.args.args().iter().filter_map(|a| a.as_type()).nth(j).unwrap();
    has_pieces(RustTyDecomposition::from_ty(arg, c.args.context()))
}
