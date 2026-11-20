use prusti_rustc_interface::{hir::attrs::LangItem, span::DUMMY_SP};
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, FunctionIdn, MethodIdn, PredicateIdn};

use crate::encoders::TyUseImpureEnc;

use super::{
    RustTy, RustTyDecomposition,
    data::TySpecifics,
    generics::{
        GArgsTyEnc, GenericParamsEnc, IndirectCastIdn, ParamTypEnc,
        lifetime_position::{
            EnclosureGraph, GeneralizedLifetime, LifetimePosition, LifetimePositionEnc,
        },
        selector::{SelectorArg, SelectorEnc, path_selector},
        r#trait::TraitEnc,
    },
    impure::TyImpureEnc,
    pure::TyPureEnc,
};

pub struct IndirectPredEnc;

#[derive(Debug, Clone, Copy)]
pub struct IndirectPredRef<'vir> {
    pub predicate: PredicateIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::Sel)>,
    /// `extend_X(k, T.., sel)`: `k` extended by the positions `sel`: the references
    /// there hold their current referents, read through `p_X_ind(k, T.., sel)`.
    pub extend: FunctionIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::Sel), vir::Snap>,
    /// `graft_X(s, T.., t, sel)`: `s` with the references at `sel` holding `t`'s
    /// referents, in which the references at other positions are collapsed.
    pub graft: FunctionIdn<
        'vir,
        (
            vir::Snap,
            vir::ManyTyVal,
            vir::ManyCSnap,
            vir::Snap,
            vir::Sel,
        ),
        vir::Snap,
    >,
    /// `shallow_X(s, T..)`: `s` with every reference collapsed, the shallow snapshot.
    pub shallow: FunctionIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap), vir::Snap>,
    /// For an array, `p_X_elem_ind(i, v, T.., sel)`: the predicates of the
    /// element `v` at index `i`. The array's predicate holds one per element;
    /// the index makes them distinct even where elements are equal.
    pub element: Option<ElementPredIdn<'vir>>,
    /// Whether the predicate has a body; that of a type parameter, or of a
    /// type whose predicate is not defined yet (e.g. a slice), is abstract
    /// and can be neither folded nor unfolded.
    pub has_body: bool,
    /// For a type parameter, `copy_ind_X(k, T, lo, hi)`: the predicate of a
    /// copy of the value `k`, given that the type is `Copy`. A `Copy` type
    /// contains no mutable references, so its predicate holds no
    /// permissions, and a copy may hold it as well as the original.
    pub copy: Option<IndirectCastIdn<'vir>>,
}

pub type ElementPredIdn<'vir> = PredicateIdn<
    'vir,
    (
        vir::CSnap,
        vir::PSnap,
        vir::ManyTyVal,
        vir::ManyCSnap,
        vir::Sel,
    ),
>;

impl<'vir> task_encoder::OutputRefAny for IndirectPredRef<'vir> {}

impl<'vir> IndirectPredRef<'vir> {
    /// `snap` with the positions `sel` holding what extending `key` by them reads
    /// through the predicate on `key`.
    pub fn extend_from(
        &self,
        snap: vir::ExprSnap<'vir>,
        key: vir::ExprSnap<'vir>,
        tys: &'vir [vir::ExprTyVal<'vir>],
        consts: &'vir [vir::ExprCSnap<'vir>],
        sel: vir::ExprSel<'vir>,
    ) -> vir::ExprSnap<'vir> {
        if vir::simplify::syntactic_eq(snap.as_dyn(), key.as_dyn()) {
            self.extend.call()(snap, tys, consts, sel)
        } else {
            let extended = self.extend.call()(key, tys, consts, sel);
            self.graft.call()(snap, tys, consts, extended, sel)
        }
    }
}

/// The positions `[first, end)` of an instantiated type, merged on the
/// snapshot `k`.
pub(crate) struct LifetimePositionBlock<'vir, 'a> {
    pub(crate) graph: &'a EnclosureGraph<'vir>,
    pub(crate) first: LifetimePosition,
    pub(crate) end: LifetimePosition,
    pub(crate) ind: IndirectPredRef<'vir>,
    pub(crate) tys: &'vir [vir::ExprTyVal<'vir>],
    pub(crate) consts: &'vir [vir::ExprCSnap<'vir>],
    pub(crate) k: vir::ExprSnap<'vir>,
}

impl<'vir> LifetimePositionBlock<'vir, '_> {
    fn contains(&self, p: LifetimePosition) -> bool {
        self.first <= p && p < self.end
    }

    /// The positions of the block, each after the positions enclosing it.
    pub(crate) fn positions(&self) -> Vec<LifetimePosition> {
        self.graph
            .positions_outermost_first()
            .filter(|&p| self.contains(p))
            .collect()
    }

    /// The key of `p`'s predicate: `k` with what the positions of the block
    /// enclosing `p` read grafted in, as in the body of the merged predicate.
    pub(crate) fn key(&self, p: LifetimePosition) -> vir::ExprSnap<'vir> {
        let ancestors = self
            .graph
            .ancestors(p)
            .iter()
            .copied()
            .filter(|&q| self.contains(q))
            .collect::<Vec<_>>();
        self.fold(&ancestors)
    }

    /// `k` with the data at the positions of the block enclosing `p` taken
    /// from `from`, another key of `p`.
    pub(crate) fn graft_enclosing(
        &self,
        p: LifetimePosition,
        from: vir::ExprSnap<'vir>,
    ) -> vir::ExprSnap<'vir> {
        let mut snap = self.k;
        for &q in self
            .graph
            .ancestors(p)
            .iter()
            .filter(|&&q| self.contains(q))
        {
            let sel = self.graph.selector(q);
            snap = self.ind.graft.call()(snap, self.tys, self.consts, from, sel);
        }
        snap
    }

    /// `k` with what each of `positions` reads through its key grafted in, in
    /// turn.
    pub(crate) fn fold(&self, positions: &[LifetimePosition]) -> vir::ExprSnap<'vir> {
        let mut snap = self.k;
        for &q in positions {
            let sel = self.graph.selector(q);
            snap = self
                .ind
                .extend_from(snap, self.key(q), self.tys, self.consts, sel);
        }
        snap
    }

    pub(crate) fn predicate(
        &self,
        key: vir::ExprSnap<'vir>,
        sel: vir::ExprSel<'vir>,
    ) -> vir::ExprBool<'vir> {
        let pred = self.ind.predicate;
        let (tys, consts) = (self.tys, self.consts);
        vir::expr! { acc([pred](key, [..[tys]], [..[consts]], sel)) }
    }
}

/// The part of a query `sel` at one own position.
#[derive(Clone, Copy)]
struct Subquery<'vir> {
    lifetime: GeneralizedLifetime<'vir>,
    /// The lifetimes of the own positions enclosing this one.
    enclosing: &'vir [GeneralizedLifetime<'vir>],
    /// Whether the query selects this position, or positions of it for a type parameter.
    guard: vir::ExprBool<'vir>,
    /// For a type parameter, the positions of it that the query selects, where `guard` holds.
    tail: Option<vir::ExprSel<'vir>>,
}

impl<'vir> Subquery<'vir> {
    /// The selector of this subquery at position `p` of the field type `field_ty`, one of
    /// this subquery's lifetime, where `guard` holds: the whole position for an own
    /// lifetime, and the selected positions of an own type parameter.
    fn in_field<E: TaskEncoder>(
        self,
        field_ty: RustTyDecomposition<'vir>,
        graph: &EnclosureGraph<'vir>,
        p: LifetimePosition,
        deps: &mut TaskEncoderDependencies<'vir, E>,
    ) -> Result<vir::ExprSel<'vir>, EncodeFullError<'vir, E>> {
        path_selector(field_ty, graph.path(p), self.tail, deps)
    }

    /// `then` where `guard` holds, `otherwise` elsewhere.
    fn if_selected(
        self,
        then: vir::ExprSnap<'vir>,
        otherwise: vir::ExprSnap<'vir>,
    ) -> vir::ExprSnap<'vir> {
        vir::with_vcx(|vcx| vcx.mk_ternary_expr(self.guard, then, otherwise))
    }
}

/// A field of a struct or of an enum variant: its type and its reader.
type Field<'vir> = (
    RustTyDecomposition<'vir>,
    vir::AdtDestructor<'vir, vir::CSnap, vir::Snap>,
);

impl IndirectPredEnc {
    /// The subqueries of the query `sel` for lifetime and type parameters of `ty`.
    fn subqueries<'vir>(
        ty: RustTy<'vir>,
        sel: vir::ExprSel<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<Subquery<'vir>>, EncodeFullError<'vir, Self>> {
        let graph = deps.require_dep::<LifetimePositionEnc>(RustTyDecomposition::identity(ty))?;
        let selector = deps.require_ref::<SelectorEnc>(ty)?;
        let param_all = deps
            .require_ref::<SelectorEnc>(RustTyDecomposition::param())?
            .all;
        vir::with_vcx(|vcx| {
            let all = vcx.mk_eq_expr(sel, (selector.all)());
            // A position's predicate is on its key extended by the positions enclosing it,
            // so their predicates and extensions come first.
            Ok(graph
                .positions_outermost_first()
                .map(|p| {
                    let lifetime = graph.lifetime(p);
                    let enclosing = graph
                        .ancestors(p)
                        .iter()
                        .map(|&q| graph.lifetime(q))
                        .collect::<Vec<_>>();
                    let enclosing = vcx.alloc_slice(&enclosing);
                    // Each own position is a generic argument of the identity
                    // decomposition; the query selects it structurally,
                    // independent of what a type parameter is instantiated with.
                    let arg = selector.args[graph.path(p)[0]];
                    let guard = arg.selects(sel);
                    let guard = vir::expr! { vcx; (all) || (guard) };
                    let tail = match arg {
                        SelectorArg::Type { index, .. } => {
                            Some(vcx.mk_ternary_expr(all, param_all(), index.call()(sel)))
                        }
                        _ => None,
                    };
                    Subquery {
                        lifetime,
                        enclosing,
                        guard,
                        tail,
                    }
                })
                .collect())
        })
    }

    /// The snapshot of the field predicate of `p` for `subquery`: `field_snap`, restricted,
    /// then extended by `p`'s ancestors in the query; the caller extended it by the others.
    fn position_snap<'vir>(
        subqueries: &[Subquery<'vir>],
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        graph: &EnclosureGraph<'vir>,
        p: LifetimePosition,
        field_snap: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        let mut snap = Self::restricted_snap(subquery, field_ty, graph, p, field_snap, deps)?;
        for &enclosing in graph.ancestors(p) {
            let lifetime = graph.lifetime(enclosing);
            let Some(&outer) = subqueries.iter().find(|s| s.lifetime == lifetime) else {
                continue;
            };
            let key = Self::position_snap(
                subqueries, outer, field_ty, graph, enclosing, field_snap, deps,
            )?;
            let sel = outer.in_field(field_ty, graph, enclosing, deps)?;
            let extended = Self::extend_from(field_ty, snap, key, sel, deps)?;
            snap = outer.if_selected(extended, snap);
        }
        Ok(snap)
    }

    /// `snap` extended by the positions `sel` of `field_ty`, whose permissions are on `key`.
    fn extend_from<'vir>(
        field_ty: RustTyDecomposition<'vir>,
        snap: vir::ExprSnap<'vir>,
        key: vir::ExprSnap<'vir>,
        sel: vir::ExprSel<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        let field_ind = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?;
        let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
        Ok(field_ind.extend_from(snap, key, args.get_ty(), args.get_const(), sel))
    }

    /// `field_snap` restricted to `p`'s ancestors in `field_ty` and, for a type parameter,
    /// all of `p`'s block, if the container may have extended other positions of
    /// `field_ty`: the field made shallow, with those positions grafted back in from
    /// `field_snap`. Which positions of a type parameter enclose the subquery is up to its
    /// instantiation, so its block keeps what the container's key has there.
    fn restricted_snap<'vir>(
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        graph: &EnclosureGraph<'vir>,
        p: LifetimePosition,
        field_snap: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        let spurious = graph.positions().any(|q| {
            subquery.enclosing.contains(&graph.lifetime(q)) && !graph.ancestors(p).contains(&q)
        });
        if !spurious {
            return Ok(field_snap);
        }
        let field_ind = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?;
        let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
        let mut keep = graph
            .ancestors(p)
            .iter()
            .map(|&q| graph.selector(q))
            .collect::<Vec<_>>();
        if let GeneralizedLifetime::RegionsIn(_) = graph.lifetime(p) {
            keep.push(graph.selector(p));
        }
        let mut snap = field_ind.shallow.call()(field_snap, args.get_ty(), args.get_const());
        for sel in keep {
            snap = field_ind.graft.call()(snap, args.get_ty(), args.get_const(), field_snap, sel);
        }
        Ok(snap)
    }

    /// `s_field` with the positions of `subquery` in `field_ty` grafted in from `t_field`;
    /// unchanged where the query does not select them.
    fn field_graft<'vir>(
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        s_field: vir::ExprSnap<'vir>,
        t_field: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        let graph = deps.require_dep::<LifetimePositionEnc>(field_ty)?;
        if graph
            .positions()
            .all(|p| graph.lifetime(p) != subquery.lifetime)
        {
            return Ok(s_field);
        }
        let graft = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.graft;
        let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
        let mut snap = s_field;
        for p in graph.positions() {
            if graph.lifetime(p) != subquery.lifetime {
                continue;
            }
            let sel = subquery.in_field(field_ty, &graph, p, deps)?;
            let grafted = graft.call()(snap, args.get_ty(), args.get_const(), t_field, sel);
            snap = subquery.if_selected(grafted, snap);
        }
        Ok(snap)
    }

    /// The fields `fields`, given by their types and readers, of `s`, with all
    /// `subqueries` grafted in from the fields of `t`.
    fn fields_graft<'vir>(
        subqueries: &[Subquery<'vir>],
        fields: &[Field<'vir>],
        s: vir::ExprSnap<'vir>,
        t: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprSnap<'vir>>, EncodeFullError<'vir, Self>> {
        let mut field_snaps = Vec::new();
        for &(field_ty, read) in fields {
            let mut field_snap = read.call()(s.downcast_ty());
            let t_field = read.call()(t.downcast_ty());
            for &subquery in subqueries {
                field_snap = Self::field_graft(subquery, field_ty, field_snap, t_field, deps)?;
            }
            field_snaps.push(field_snap);
        }
        Ok(field_snaps)
    }

    /// The fields `fields`, given by their types and readers, of `s`, each made shallow.
    fn fields_shallow<'vir>(
        fields: &[Field<'vir>],
        s: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprSnap<'vir>>, EncodeFullError<'vir, Self>> {
        let mut field_snaps = Vec::new();
        for &(field_ty, read) in fields {
            let field_snap = read.call()(s.downcast_ty());
            field_snaps.push(Self::field_shallow(field_ty, field_snap, deps)?);
        }
        Ok(field_snaps)
    }

    /// `field_snap` of type `field_ty` made shallow.
    fn field_shallow<'vir>(
        field_ty: RustTyDecomposition<'vir>,
        field_snap: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        if deps
            .require_dep::<LifetimePositionEnc>(field_ty)?
            .is_empty()
        {
            return Ok(field_snap);
        }
        let shallow = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.shallow;
        let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
        Ok(shallow.call()(field_snap, args.get_ty(), args.get_const()))
    }

    /// The field predicates for `subquery`: one per position of its lifetime in `field_ty`.
    fn field_preds<'vir>(
        subqueries: &[Subquery<'vir>],
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        field_snap: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprBool<'vir>>, EncodeFullError<'vir, Self>> {
        let graph = deps.require_dep::<LifetimePositionEnc>(field_ty)?;
        let mut preds = Vec::new();
        for p in graph.positions_outermost_first() {
            if graph.lifetime(p) != subquery.lifetime {
                continue;
            }
            let pred = deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.predicate;
            let args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
            let snap =
                Self::position_snap(subqueries, subquery, field_ty, &graph, p, field_snap, deps)?;
            let sel = subquery.in_field(field_ty, &graph, p, deps)?;
            preds.push(vir::expr! {
                acc([pred](snap, [..[args.get_ty()]], [..[args.get_const()]], sel))
            });
        }
        Ok(preds)
    }

    /// `field_snap` extended by `subquery` at each position of its lifetime in `field_ty`;
    /// the field predicates are on `field_pred`.
    fn field_extend<'vir>(
        subqueries: &[Subquery<'vir>],
        subquery: Subquery<'vir>,
        field_ty: RustTyDecomposition<'vir>,
        field_snap: vir::ExprSnap<'vir>,
        field_pred: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<vir::ExprSnap<'vir>, EncodeFullError<'vir, Self>> {
        let graph = deps.require_dep::<LifetimePositionEnc>(field_ty)?;
        if graph
            .positions()
            .all(|p| graph.lifetime(p) != subquery.lifetime)
        {
            return Ok(field_snap);
        }
        let mut snap = field_snap;
        for p in graph.positions_outermost_first() {
            if graph.lifetime(p) != subquery.lifetime {
                continue;
            }
            let key =
                Self::position_snap(subqueries, subquery, field_ty, &graph, p, field_pred, deps)?;
            let sel = subquery.in_field(field_ty, &graph, p, deps)?;
            let extended = Self::extend_from(field_ty, snap, key, sel, deps)?;
            snap = subquery.if_selected(extended, snap);
        }
        Ok(snap)
    }

    /// The field predicates of the fields `fields`, given by their types and
    /// readers, of the snapshot `k` for `subquery`.
    fn fields_preds<'vir>(
        subqueries: &[Subquery<'vir>],
        subquery: Subquery<'vir>,
        fields: &[Field<'vir>],
        k: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprBool<'vir>>, EncodeFullError<'vir, Self>> {
        let mut preds = Vec::new();
        for &(field_ty, read) in fields {
            let field_snap = read.call()(k.downcast_ty());
            preds.extend(Self::field_preds(
                subqueries, subquery, field_ty, field_snap, deps,
            )?);
        }
        Ok(preds)
    }

    /// The fields `fields`, given by their types and readers, of `k`, each
    /// extended by all `subqueries`, whose field predicates are on the fields
    /// of `k`.
    fn fields_extend<'vir>(
        subqueries: &[Subquery<'vir>],
        fields: &[Field<'vir>],
        k: vir::ExprSnap<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<Vec<vir::ExprSnap<'vir>>, EncodeFullError<'vir, Self>> {
        let mut field_snaps = Vec::new();
        for &(field_ty, read) in fields {
            let field_pred = read.call()(k.downcast_ty());
            let mut field_snap = field_pred;
            for &subquery in subqueries {
                field_snap = Self::field_extend(
                    subqueries, subquery, field_ty, field_snap, field_pred, deps,
                )?;
            }
            field_snaps.push(field_snap);
        }
        Ok(field_snaps)
    }
}

/// The declarations of `graft` and `shallow` of a type.
#[derive(Clone, Copy)]
pub enum GraftShallowDecls<'vir> {
    /// Functions with bodies, or, for an array, abstract ones with their
    /// results given element by element, and a domain with facts about them.
    Functions(
        vir::Function<'vir>,
        vir::Function<'vir>,
        Option<vir::Domain<'vir>>,
    ),
    /// At a type parameter, where they are abstract and do not depend on the
    /// heap: domain functions, with their facts as axioms.
    Domain(vir::Domain<'vir>),
}

impl IndirectPredEnc {
    /// The domain of `graft` and `shallow` of the type parameter `task_key`.
    /// Their results are values of the parameter's type, as the snapshots of
    /// its predicate are, so that they can be cast to the instantiated type.
    /// Grafting changes only what references hold, not the shallow snapshot,
    /// as for every type whose `graft` and `shallow` have bodies.
    fn param_graft_shallow<'vir>(
        task_key: &RustTy<'vir>,
        graft: FunctionIdn<
            'vir,
            (
                vir::Snap,
                vir::ManyTyVal,
                vir::ManyCSnap,
                vir::Snap,
                vir::Sel,
            ),
            vir::Snap,
        >,
        shallow: FunctionIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap), vir::Snap>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<GraftShallowDecls<'vir>, EncodeFullError<'vir, Self>> {
        let params = deps.require_dep::<GenericParamsEnc>(task_key.params)?;
        let snapshot = deps.require_dep::<TyPureEnc>(*task_key)?.data.snapshot;
        let selector = deps.require_ref::<SelectorEnc>(*task_key)?;
        let typ = deps.require_dep::<ParamTypEnc>(())?.typ;
        vir::with_vcx(|vcx| {
            let s_decl = vcx.mk_local_decl("s", snapshot);
            let t_decl = vcx.mk_local_decl("t", snapshot);
            let sel_decl = vcx.mk_local_decl("sel", selector.ty);
            let s: vir::ExprSnap<'vir> = vcx.mk_local_ex(s_decl);
            let t: vir::ExprSnap<'vir> = vcx.mk_local_ex(t_decl);
            let sel = vcx.mk_local_ex(sel_decl);
            let (tys, consts) = (params.ty_exprs(), params.const_exprs());
            let param = tys[0];
            let grafted = graft.call()(s, tys, consts, t, sel);
            let shallow_s = shallow.call()(s, tys, consts);
            let shallow_grafted = shallow.call()(grafted, tys, consts);
            let grafted_typ = vcx.mk_eq_expr(typ(grafted.downcast_ty()), param);
            let shallow_typ = vcx.mk_eq_expr(typ(shallow_s.downcast_ty()), param);
            let graft_facts = vir::expr! { vcx;
                forall [s_decl], ..[params.ty_decls()], ..[params.const_decls()],
                    [t_decl], [sel_decl]
                    :: { [grafted] }
                    [grafted_typ]
            };
            let shallow_facts = vir::expr! { vcx;
                forall [s_decl], ..[params.ty_decls()], ..[params.const_decls()]
                    :: { [shallow_s] }
                    [shallow_typ]
            };
            let shallow_graft = vir::expr! { vcx;
                forall [s_decl], ..[params.ty_decls()], ..[params.const_decls()],
                    [t_decl], [sel_decl]
                    :: { [shallow_grafted] }
                    ([shallow_grafted]) == ([shallow_s])
            };
            let name = task_key.name();
            Ok(GraftShallowDecls::Domain(vcx.mk_domain(
                vir::vir_format_identifier!(vcx, "Indirect_{name}"),
                &[],
                vcx.alloc_slice(&[
                    vcx.mk_domain_axiom(
                        vir::vir_format_identifier!(vcx, "graft_{name}_facts"),
                        graft_facts,
                    ),
                    vcx.mk_domain_axiom(
                        vir::vir_format_identifier!(vcx, "shallow_{name}_facts"),
                        shallow_facts,
                    ),
                    vcx.mk_domain_axiom(
                        vir::vir_format_identifier!(vcx, "shallow_graft_{name}"),
                        shallow_graft,
                    ),
                ]),
                vcx.alloc_slice(&[
                    vcx.mk_domain_function(graft, false, None),
                    vcx.mk_domain_function(shallow, false, None),
                ]),
                None,
            )))
        })
    }
}

impl TaskEncoder for IndirectPredEnc {
    task_encoder::encoder_cache!(IndirectPredEnc);
    const ENCODER_NAME: &'static str = "indirect predicate encoder";

    type TaskDescription<'vir> = RustTy<'vir>;

    type OutputRef<'vir> = IndirectPredRef<'vir>;
    type OutputFullLocal<'vir> = (
        vir::Predicate<'vir>,
        vir::Function<'vir>,
        GraftShallowDecls<'vir>,
        Option<vir::Predicate<'vir>>,
        Option<vir::Method<'vir>>,
    );

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        let pure = deps.require_dep::<TyPureEnc>(*task_key)?;
        let params = deps.require_dep::<GenericParamsEnc>(task_key.params)?;
        let selector = deps.require_ref::<SelectorEnc>(*task_key)?;
        let ty = task_key.zip(pure);
        vir::with_vcx(|vcx| {
            let snapshot = pure.data.snapshot;
            let predicate = PredicateIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "p_{}_ind", task_key.name())),
                (snapshot, params.ty_args(), params.const_args(), selector.ty),
            );
            let extend = FunctionIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "extend_{}", task_key.name())),
                (snapshot, params.ty_args(), params.const_args(), selector.ty),
                snapshot,
            );
            let graft = FunctionIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "graft_{}", task_key.name())),
                (
                    snapshot,
                    params.ty_args(),
                    params.const_args(),
                    snapshot,
                    selector.ty,
                ),
                snapshot,
            );
            let shallow = FunctionIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "shallow_{}", task_key.name())),
                (snapshot, params.ty_args(), params.const_args()),
                snapshot,
            );
            // Array elements are indexed by `usize` snapshots.
            let usize_ty = RustTyDecomposition::from_prim_ty(vcx.tcx().types.usize).ty;
            let usize_enc = deps.require_dep::<TyPureEnc>(usize_ty)?;
            let usize_snap = usize_enc.snapshot.downcast_ty();
            let element = match &ty.specifics {
                TySpecifics::ArrayLike(data) if !data.slice => Some(PredicateIdn::new(
                    vir::ViperIdent::new(vir::vir_format!(vcx, "p_{}_elem_ind", task_key.name())),
                    (
                        usize_snap,
                        vir::TYPE_PSNAP,
                        params.ty_args(),
                        params.const_args(),
                        selector.ty,
                    ),
                )),
                _ => None,
            };
            let positionless = !ty.specifics.is_param()
                && deps
                    .require_dep::<LifetimePositionEnc>(RustTyDecomposition::identity(task_key))?
                    .is_empty();
            let has_body = positionless
                || match &ty.specifics {
                    TySpecifics::StructLike(_)
                    | TySpecifics::EnumLike(_)
                    | TySpecifics::MutRef(_)
                    | TySpecifics::ImmRef(_)
                    | TySpecifics::Raw(_) => true,
                    TySpecifics::ArrayLike(data) => !data.slice,
                    _ => false,
                };
            let copy = ty.specifics.is_param().then(|| {
                MethodIdn::new(
                    vir::ViperIdent::new(vir::vir_format!(vcx, "copy_ind_{}", task_key.name())),
                    (snapshot, params.ty_args(), params.const_args(), selector.ty),
                )
            });
            deps.emit_output_ref(
                *task_key,
                IndirectPredRef {
                    predicate,
                    extend,
                    graft,
                    shallow,
                    element,
                    has_body,
                    copy,
                },
            )?;

            let k_decl = vcx.mk_local_decl("k", snapshot);
            let sel_decl = vcx.mk_local_decl("sel", selector.ty);
            let k: vir::ExprSnap<'vir> = vcx.mk_local_ex(k_decl);
            let sel = vcx.mk_local_ex(sel_decl);
            // The snapshot arguments of `graft` and `shallow`.
            let s_decl = vcx.mk_local_decl("s", snapshot);
            let t_decl = vcx.mk_local_decl("t", snapshot);
            let s: vir::ExprSnap<'vir> = vcx.mk_local_ex(s_decl);
            let t: vir::ExprSnap<'vir> = vcx.mk_local_ex(t_decl);
            // The postconditions of `extend`, `graft` and `shallow` of an
            // array, which state their results element by element.
            let mut array_posts = (Vec::new(), Vec::new(), Vec::new());
            let mut array_axioms = Vec::new();
            let mut element_decl = None;

            let (body, extend_body, graft_body, shallow_body) = match &ty.specifics {
                TySpecifics::StructLike(data) => {
                    let subqueries = Self::subqueries(task_key, sel, deps)?;
                    let fields = data
                        .fields
                        .iter()
                        .map(|(field, pure_field)| {
                            (field.decompose(task_key.params), pure_field.read)
                        })
                        .collect::<Vec<_>>();
                    let mut conjuncts = Vec::new();
                    for &subquery in &subqueries {
                        let preds = Self::fields_preds(&subqueries, subquery, &fields, k, deps)?;
                        if !preds.is_empty() {
                            let guard = subquery.guard;
                            let preds = vcx.mk_conj(&preds);
                            conjuncts.push(vir::expr! { vcx; (guard) ==> (preds) });
                        }
                    }
                    let cons = data.data.1.field_snaps_to_snap;
                    let field_snaps = Self::fields_extend(&subqueries, &fields, k, deps)?;
                    let extend_body = cons.call()(&field_snaps).upcast_ty();
                    let field_snaps = Self::fields_graft(&subqueries, &fields, s, t, deps)?;
                    let graft_body = cons.call()(&field_snaps).upcast_ty();
                    let field_snaps = Self::fields_shallow(&fields, s, deps)?;
                    let shallow_body = cons.call()(&field_snaps).upcast_ty();
                    (
                        Some(vcx.mk_conj(&conjuncts)),
                        Some(extend_body),
                        Some(graft_body),
                        Some(shallow_body),
                    )
                }
                TySpecifics::EnumLike(data) => {
                    let subqueries = Self::subqueries(task_key, sel, deps)?;
                    let discr = data.data.1.snap_to_discr_snap;
                    // The variants with lifetime positions.
                    let mut variants = Vec::new();
                    for variant in &data.variants {
                        let fields = variant
                            .inner
                            .fields
                            .iter()
                            .map(|(field, pure_field)| {
                                (field.decompose(task_key.params), pure_field.read)
                            })
                            .collect::<Vec<_>>();
                        let mut has_positions = false;
                        for &(field_ty, _) in &fields {
                            has_positions |= !deps
                                .require_dep::<LifetimePositionEnc>(field_ty)?
                                .is_empty();
                        }
                        if has_positions {
                            let cons = variant.inner.data.1.field_snaps_to_snap;
                            variants.push((variant.data.1.discr, fields, cons));
                        }
                    }
                    // The field predicates of each variant are guarded by its
                    // discriminant.
                    let mut conjuncts = Vec::new();
                    for &subquery in &subqueries {
                        let mut preds = Vec::new();
                        for (variant_discr, fields, _) in &variants {
                            let variant_preds =
                                Self::fields_preds(&subqueries, subquery, fields, k, deps)?;
                            if !variant_preds.is_empty() {
                                let in_variant =
                                    vcx.mk_eq_expr(discr.call()(k.downcast_ty()), *variant_discr);
                                let variant_preds = vcx.mk_conj(&variant_preds);
                                preds.push(vir::expr! { vcx; (in_variant) ==> (variant_preds) });
                            }
                        }
                        if !preds.is_empty() {
                            let guard = subquery.guard;
                            let preds = vcx.mk_conj(&preds);
                            conjuncts.push(vir::expr! { vcx; (guard) ==> (preds) });
                        }
                    }
                    let mut extend_body = k;
                    for (variant_discr, fields, cons) in variants.iter().rev() {
                        let field_snaps = Self::fields_extend(&subqueries, fields, k, deps)?;
                        let extended: vir::ExprSnap<'vir> = cons.call()(&field_snaps).upcast_ty();
                        let in_variant =
                            vcx.mk_eq_expr(discr.call()(k.downcast_ty()), *variant_discr);
                        extend_body = vir::expr! { vcx;
                            (in_variant) ? (extended) : (extend_body)
                        };
                    }
                    // Plug only if both snapshots have the variant; `s` must not change
                    // otherwise.
                    let mut graft_body = s;
                    for (variant_discr, fields, cons) in variants.iter().rev() {
                        let field_snaps = Self::fields_graft(&subqueries, fields, s, t, deps)?;
                        let grafted: vir::ExprSnap<'vir> = cons.call()(&field_snaps).upcast_ty();
                        let s_in_variant =
                            vcx.mk_eq_expr(discr.call()(s.downcast_ty()), *variant_discr);
                        let t_in_variant =
                            vcx.mk_eq_expr(discr.call()(t.downcast_ty()), *variant_discr);
                        graft_body = vir::expr! { vcx;
                            ((s_in_variant) && (t_in_variant)) ? (grafted) : (graft_body)
                        };
                    }
                    let mut shallow_body = s;
                    for (variant_discr, fields, cons) in variants.iter().rev() {
                        let field_snaps = Self::fields_shallow(fields, s, deps)?;
                        let collapsed: vir::ExprSnap<'vir> = cons.call()(&field_snaps).upcast_ty();
                        let in_variant =
                            vcx.mk_eq_expr(discr.call()(s.downcast_ty()), *variant_discr);
                        shallow_body = vir::expr! { vcx;
                            (in_variant) ? (collapsed) : (shallow_body)
                        };
                    }
                    (
                        Some(vcx.mk_conj(&conjuncts)),
                        Some(extend_body),
                        Some(graft_body),
                        Some(shallow_body),
                    )
                }
                TySpecifics::MutRef((data, pure_data)) => {
                    let referent_ty = data.referent.decompose(task_key.params);
                    let referent = deps.require_dep::<TyUseImpureEnc>(referent_ty)?;
                    let addr = pure_data.deref_access.call()(k.downcast_ty());
                    let referent_pred = referent.ref_to_pred(vcx, addr, None);
                    let subqueries = Self::subqueries(task_key, sel, deps)?;
                    // With 'a in the query, the referent's snapshot is read from the heap,
                    // otherwise it is the stored one.
                    let in_query = subqueries
                        .iter()
                        .find(|subquery| {
                            matches!(subquery.lifetime, GeneralizedLifetime::Region(_))
                        })
                        .unwrap()
                        .guard;
                    let current_snap = referent.ref_to_snap(addr);
                    let stored_snap: vir::ExprSnap<'vir> =
                        pure_data.value_access.call()(k.downcast_ty()).upcast_ty();
                    let referent_snap = vir::expr! { vcx;
                        (in_query) ? (current_snap) : (stored_snap)
                    };
                    let mut conjuncts = Vec::new();
                    for &subquery in &subqueries {
                        let mut preds = Vec::new();
                        // The own lifetime of the reference gives access to the referent.
                        if matches!(subquery.lifetime, GeneralizedLifetime::Region(_)) {
                            preds.push(referent_pred);
                        }
                        // The referent's positions are passed on like a field's.
                        preds.extend(Self::field_preds(
                            &subqueries,
                            subquery,
                            referent_ty,
                            referent_snap,
                            deps,
                        )?);
                        if !preds.is_empty() {
                            let guard = subquery.guard;
                            let preds = vcx.mk_conj(&preds);
                            conjuncts.push(vir::expr! { vcx; (guard) ==> (preds) });
                        }
                    }
                    // The value is read from the heap if the query contains 'a and stored
                    // otherwise, then extended at the referent's positions.
                    let mut value = referent_snap;
                    for &subquery in &subqueries {
                        value = Self::field_extend(
                            &subqueries,
                            subquery,
                            referent_ty,
                            value,
                            referent_snap,
                            deps,
                        )?;
                    }
                    let metadata = pure_data.metadata_access.call()(k.downcast_ty());
                    let extend_body =
                        pure_data.prim_to_snap.call()(addr, metadata, value.downcast_ty());
                    // The referent's positions in the query are grafted in like a
                    // field's: into `t`'s referent made shallow if 'a is in the
                    // query, into `s`'s otherwise. A position behind 'a that is not
                    // in the query is thus collapsed, even if `t` extended it.
                    let s_addr = pure_data.deref_access.call()(s.downcast_ty());
                    let s_metadata = pure_data.metadata_access.call()(s.downcast_ty());
                    let s_value: vir::ExprSnap<'vir> =
                        pure_data.value_access.call()(s.downcast_ty()).upcast_ty();
                    let t_value: vir::ExprSnap<'vir> =
                        pure_data.value_access.call()(t.downcast_ty()).upcast_ty();
                    let t_shallow = Self::field_shallow(referent_ty, t_value, deps)?;
                    let mut value = vir::expr! { vcx; (in_query) ? (t_shallow) : (s_value) };
                    for &subquery in &subqueries {
                        value = Self::field_graft(subquery, referent_ty, value, t_value, deps)?;
                    }
                    let graft_body =
                        pure_data.prim_to_snap.call()(s_addr, s_metadata, value.downcast_ty());
                    // Reset, the reference has the snapshot its direct predicate gives it.
                    let arbitrary_value = deps
                        .require_dep::<TyImpureEnc>(*task_key)?
                        .expect_mutref()
                        .arbitrary_value;
                    let shallow_body = arbitrary_value(
                        s_addr,
                        s_metadata,
                        params.ty_exprs(),
                        params.const_exprs(),
                    );
                    (
                        Some(vcx.mk_conj(&conjuncts)),
                        Some(extend_body.upcast_ty()),
                        Some(graft_body.upcast_ty()),
                        Some(shallow_body.upcast_ty()),
                    )
                }
                // An array is like a struct with one field per element, all of
                // the element type, which has all of the array's positions.
                // Its length is a parameter, so its predicate holds those of the
                // elements by a quantifier, and `extend`, `graft` and `shallow`
                // are abstract, with their results given element by element.
                TySpecifics::ArrayLike(data) if !data.slice => {
                    let element = element.unwrap();
                    let index_access = data.data.1.index_access;
                    let index =
                        |a: vir::ExprSnap<'vir>, i: vir::ExprCSnap<'vir>| -> vir::ExprSnap<'vir> {
                            index_access.call()(a.downcast_ty(), i).upcast_ty()
                        };
                    let element_ty = data.data.0.decompose(task_key.params);
                    let usize_prim = usize_enc.expect_primitive();
                    let len: vir::ExprInt<'vir> = usize_prim
                        .snap_to_prim(params.const_exprs()[0])
                        .downcast_ty();
                    let subqueries = Self::subqueries(task_key, sel, deps)?;
                    let i_decl = vcx.mk_local_decl("i", usize_snap);
                    let i: vir::ExprCSnap<'vir> = vcx.mk_local_ex(i_decl);
                    let i_value: vir::ExprInt<'vir> = usize_prim.snap_to_prim(i).downcast_ty();
                    let in_range = vir::expr! { vcx; (i_value) < (len) };
                    let element_pred = |v: vir::ExprSnap<'vir>| {
                        vcx.mk_predicate_app_expr(element(
                            i,
                            v.downcast_ty(),
                            params.ty_exprs(),
                            params.const_exprs(),
                            sel,
                        )(None))
                    };
                    let forall = |trigger: vir::ExprSnap<'vir>, body: vir::ExprBool<'vir>| {
                        vcx.mk_forall_expr(
                            vcx.alloc_slice(&[i_decl]),
                            vcx.alloc_slice(&[vcx.mk_trigger(&[trigger])]),
                            vir::expr! { vcx; (in_range) ==> (body) },
                        )
                    };

                    // The element predicate.
                    let v_decl = vcx.mk_local_decl("v", vir::TYPE_PSNAP);
                    let v: vir::ExprSnap<'vir> = vcx.mk_local_ex(v_decl).upcast_ty();
                    let mut conjuncts = Vec::new();
                    for &subquery in &subqueries {
                        let preds = Self::field_preds(&subqueries, subquery, element_ty, v, deps)?;
                        if !preds.is_empty() {
                            let guard = subquery.guard;
                            let preds = vcx.mk_conj(&preds);
                            conjuncts.push(vir::expr! { vcx; (guard) ==> (preds) });
                        }
                    }
                    element_decl = Some(vcx.mk_predicate(
                        element,
                        (
                            i_decl,
                            v_decl,
                            params.ty_decls(),
                            params.const_decls(),
                            sel_decl,
                        ),
                        Some(vcx.mk_conj(&conjuncts)),
                    ));
                    let k_i = index(k, i);
                    let body = forall(k_i, element_pred(k_i));

                    let result = vcx.mk_result(snapshot);
                    let result_i = index(result, i);
                    let mut extended = k_i;
                    for &subquery in &subqueries {
                        extended = Self::field_extend(
                            &subqueries,
                            subquery,
                            element_ty,
                            extended,
                            k_i,
                            deps,
                        )?;
                    }
                    // What `extend` reads at element `i`, through its predicate.
                    let array_pred =
                        predicate(k, params.ty_exprs(), params.const_exprs(), sel)(None);
                    let element_pred = element(
                        i,
                        k_i.downcast_ty(),
                        params.ty_exprs(),
                        params.const_exprs(),
                        sel,
                    )(None);
                    let extended = vcx.mk_unfolding_expr(
                        array_pred,
                        vcx.mk_unfolding_expr(element_pred, extended),
                    );
                    array_posts
                        .0
                        .push(forall(result_i, vcx.mk_eq_expr(result_i, extended)));

                    let (s_i, t_i) = (index(s, i), index(t, i));
                    let mut grafted = s_i;
                    for &subquery in &subqueries {
                        grafted = Self::field_graft(subquery, element_ty, grafted, t_i, deps)?;
                    }
                    array_posts
                        .1
                        .push(forall(result_i, vcx.mk_eq_expr(result_i, grafted)));

                    let collapsed = Self::field_shallow(element_ty, s_i, deps)?;
                    array_posts
                        .2
                        .push(forall(result_i, vcx.mk_eq_expr(result_i, collapsed)));
                    // Making an array shallow makes each element shallow, so
                    // it commutes with replacing an element. The arrays are
                    // equal because they have equal elements, which the
                    // postcondition gives only element by element.
                    let update = data.data.1.update;
                    let a_decl = vcx.mk_local_decl("a", snapshot);
                    let a: vir::ExprSnap<'vir> = vcx.mk_local_ex(a_decl);
                    let updated = update.call()(a.downcast_ty(), i, v.downcast_ty()).upcast_ty();
                    let shallow_updated =
                        shallow.call()(updated, params.ty_exprs(), params.const_exprs());
                    let shallow_a = shallow.call()(a, params.ty_exprs(), params.const_exprs());
                    let shallow_v = Self::field_shallow(element_ty, v, deps)?;
                    let updated_shallow =
                        update.call()(shallow_a.downcast_ty(), i, shallow_v.downcast_ty());
                    array_axioms.push(vcx.mk_domain_axiom(
                        vir::vir_format_identifier!(vcx, "shallow_update_{}", task_key.name()),
                        vir::expr! { vcx;
                            forall [a_decl], [i_decl], [v_decl], ..[params.ty_decls()],
                                ..[params.const_decls()]
                                :: { [shallow_updated] }
                                ([shallow_updated]) == ([updated_shallow.upcast_ty()])
                        },
                    ));
                    (Some(body), None, None, None)
                }
                // A shared reference or raw pointer has no lifetime positions.
                TySpecifics::ImmRef(_) | TySpecifics::Raw(_) => {
                    (Some(vcx.mk_bool::<true>()), Some(k), Some(s), Some(s))
                }
                // Neither does any other type without lifetime or type
                // parameters, e.g. a primitive type. Its predicate holds
                // nothing, so that it can be folded wherever it is required.
                _ if positionless => (Some(vcx.mk_bool::<true>()), Some(k), Some(s), Some(s)),
                // TODO: the remaining types.
                _ => (None, None, None, None),
            };

            debug_assert_eq!(body.is_some(), has_body);
            let predicate_decl = vcx.mk_predicate(
                predicate,
                (k_decl, params.ty_decls(), params.const_decls(), sel_decl),
                body,
            );
            let copy_decl = match copy {
                Some(copy) => {
                    let copy_trait = vcx.tcx().require_lang_item(LangItem::Copy, DUMMY_SP);
                    let is_copy = deps.require_ref::<TraitEnc>(copy_trait)?.impl_fun;
                    let is_copy = is_copy(params.ty_exprs(), params.const_exprs());
                    let pred = vir::expr! { vcx;
                        acc([predicate](
                            k,
                            [..[params.ty_exprs()]],
                            [..[params.const_exprs()]],
                            sel
                        ))
                    };
                    Some(vcx.mk_method(
                        copy,
                        (k_decl, params.ty_decls(), params.const_decls(), sel_decl),
                        &[],
                        vcx.alloc_slice(&[is_copy]),
                        vcx.alloc_slice(&[pred]),
                        None,
                    ))
                }
                None => None,
            };

            let pre = vir::expr! { vcx;
                acc([predicate](
                    k,
                    [..[params.ty_exprs()]],
                    [..[params.const_exprs()]],
                    sel
                ))
            };
            let extend_body = extend_body.map(|extend_body| {
                vir::expr! { vcx;
                    unfolding ([predicate](
                        k,
                        [..[params.ty_exprs()]],
                        [..[params.const_exprs()]],
                        sel
                    )) in (extend_body)
                }
            });
            // At a type parameter, `extend`, `graft` and `shallow` are abstract. What
            // `extend` reads differs from its key only at the extended positions, so
            // grafting it back in gives it again, and making it shallow gives the
            // shallow key; field
            // snapshots of types containing the parameter have these forms. Their
            // results are values of the parameter's type, as the snapshots of its
            // predicate are, so that they can be cast to the instantiated type.
            let param_typ = if ty.specifics.is_param() {
                let typ = deps.require_dep::<ParamTypEnc>(())?.typ;
                let result = vcx.mk_result(vir::TYPE_PSNAP);
                Some(vcx.mk_eq_expr(typ(result), params.ty_exprs()[0]))
            } else {
                None
            };
            let mut extend_posts = array_posts.0;
            extend_posts.extend(param_typ);
            if ty.specifics.is_param() {
                let result = vcx.mk_result(snapshot);
                let grafted = graft.call()(k, params.ty_exprs(), params.const_exprs(), result, sel);
                extend_posts.push(vcx.mk_eq_expr(grafted, result));
                let shallow_result =
                    shallow.call()(result, params.ty_exprs(), params.const_exprs());
                let shallow_key = shallow.call()(k, params.ty_exprs(), params.const_exprs());
                extend_posts.push(vcx.mk_eq_expr(shallow_result, shallow_key));
            }
            let extend_decl = vcx.mk_function(
                extend,
                (k_decl, params.ty_decls(), params.const_decls(), sel_decl),
                vcx.alloc_slice(&[pre]),
                vcx.alloc_slice(&extend_posts),
                None,
                extend_body,
            );

            let graft_shallow = if ty.specifics.is_param() {
                Self::param_graft_shallow(task_key, graft, shallow, deps)?
            } else {
                let graft_decl = vcx.mk_function(
                    graft,
                    (
                        s_decl,
                        params.ty_decls(),
                        params.const_decls(),
                        t_decl,
                        sel_decl,
                    ),
                    &[],
                    vcx.alloc_slice(&array_posts.1),
                    None,
                    graft_body,
                );
                let shallow_decl = vcx.mk_function(
                    shallow,
                    (s_decl, params.ty_decls(), params.const_decls()),
                    &[],
                    vcx.alloc_slice(&array_posts.2),
                    None,
                    shallow_body,
                );
                let axioms = (!array_axioms.is_empty()).then(|| {
                    vcx.mk_domain(
                        vir::vir_format_identifier!(vcx, "Indirect_{}", task_key.name()),
                        &[],
                        vcx.alloc_slice(&array_axioms),
                        &[],
                        None,
                    )
                });
                GraftShallowDecls::Functions(graft_decl, shallow_decl, axioms)
            };

            Ok((
                (
                    predicate_decl,
                    extend_decl,
                    graft_shallow,
                    element_decl,
                    copy_decl,
                ),
                (),
            ))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for (predicate, extend, graft_shallow, element, copy) in Self::all_outputs_local(program) {
            program.add_predicate(predicate);
            if let Some(element) = element {
                program.add_predicate(element);
            }
            program.add_function(extend);
            match graft_shallow {
                GraftShallowDecls::Functions(graft, shallow, axioms) => {
                    program.add_function(graft);
                    program.add_function(shallow);
                    if let Some(axioms) = axioms {
                        program.add_domain(axioms);
                    }
                }
                GraftShallowDecls::Domain(domain) => program.add_domain(domain),
            }
            if let Some(copy) = copy {
                program.add_method(copy);
            }
        }
    }
}

/// Tests that verify Viper code using the indirect predicates with Silicon.
#[cfg(test)]
mod verification_tests {
    use prusti_rustc_interface::span::DUMMY_SP;
    use task_encoder::TaskEncoder;

    use super::IndirectPredEnc;
    use crate::encoders::ty::{
        RustTyDecomposition, generics::IndirectCastersEnc, test_util::verify,
    };

    /// Verifies the Viper code `viper` together with the indirect predicate
    /// of the type alias `X` defined in `src`.
    fn verify_with_pred(src: &str, viper: &str) -> Result<(), String> {
        verify(
            src,
            |ty| {
                IndirectPredEnc::encode(ty.ty, true, DUMMY_SP).unwrap();
            },
            viper,
        )
    }

    /// Like `verify_with_pred`, also with the casts of `X`'s type arguments.
    fn verify_with_casts(src: &str, viper: &str) -> Result<(), String> {
        verify(
            src,
            |ty| {
                IndirectPredEnc::encode(ty.ty, true, DUMMY_SP).unwrap();
                for arg in ty.args.args() {
                    let arg = RustTyDecomposition::from_ty(arg.expect_ty(), ty.args.context());
                    IndirectCastersEnc::encode(
                        (RustTyDecomposition::param(), arg.ty),
                        true,
                        DUMMY_SP,
                    )
                    .unwrap();
                }
            },
            viper,
        )
    }

    fn assert_verifies(src: &str, viper: &str) {
        if let Err(output) = verify_with_pred(src, viper) {
            panic!("verification failed:\n{output}");
        }
    }

    /// Asserts that the verification fails with an error containing `error`.
    fn assert_fails(src: &str, viper: &str, error: &str) {
        match verify_with_pred(src, viper) {
            Ok(()) => panic!("verification succeeded"),
            Err(output) => assert!(
                output.contains(error),
                "verification failed differently:\n{output}"
            ),
        }
    }

    /// `&'a mut U`: `'a` is `p_Ref_mutable_lft_0()`, the referent's positions
    /// are `p_Ref_mutable_ty_0(p_Param_all())`.
    #[test]
    fn mutable_reference() {
        let src = "type X<'a, T> = &'a mut T;";
        assert_verifies(
            src,
            r#"
            // 'a holds the referent.
            method lifetime(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures p_Param(k.s_Ref_mutable_0, U)
            {
              unfold p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
            }

            // The referent's positions hold those of the referent, on the
            // snapshot stored in `k`.
            method referent_positions(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_ty_0(p_Param_all()))
              ensures p_Param_ind(k.s_Ref_mutable_2, U, p_Param_all())
            {
              unfold p_Ref_mutable_ind(k, U, p_Ref_mutable_ty_0(p_Param_all()))
            }

            // With 'a, they are on the referent's current snapshot.
            method all_positions(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_all())
              ensures p_Param(k.s_Ref_mutable_0, U)
              ensures p_Param_ind(p_Param_snap(k.s_Ref_mutable_0, U), U, p_Param_all())
            {
              unfold p_Ref_mutable_ind(k, U, p_Ref_mutable_all())
            }

            // Extending by 'a reads the referent's snapshot and keeps the address.
            method extend(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures extend_Ref_mutable(k, U, p_Ref_mutable_lft_0()).s_Ref_mutable_0 == k.s_Ref_mutable_0
              ensures extend_Ref_mutable(k, U, p_Ref_mutable_lft_0()).s_Ref_mutable_2 ==
                (unfolding p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0()) in p_Param_snap(k.s_Ref_mutable_0, U))
            {
            }

            // Split off 'a, the referent's positions are on the snapshot
            // extended by 'a.
            method split(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_all())
              ensures p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures p_Ref_mutable_ind(extend_Ref_mutable(k, U, p_Ref_mutable_lft_0()), U, p_Ref_mutable_ty_0(p_Param_all()))
            {
              unfold p_Ref_mutable_ind(k, U, p_Ref_mutable_all())
              fold p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              fold p_Ref_mutable_ind(extend_Ref_mutable(k, U, p_Ref_mutable_lft_0()), U, p_Ref_mutable_ty_0(p_Param_all()))
            }
            "#,
        );
        // The referent's positions do not hold the referent.
        assert_fails(
            src,
            r#"
            method test(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_ty_0(p_Param_all()))
              ensures p_Param(k.s_Ref_mutable_0, U)
            {
              unfold p_Ref_mutable_ind(k, U, p_Ref_mutable_ty_0(p_Param_all()))
            }
            "#,
            "insufficient permission to access p_Param",
        );
        // 'a does not hold the referent's positions.
        assert_fails(
            src,
            r#"
            method test(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures p_Param(k.s_Ref_mutable_0, U)
              ensures p_Param_ind(p_Param_snap(k.s_Ref_mutable_0, U), U, p_Param_all())
            {
              unfold p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
            }
            "#,
            "insufficient permission to access p_Param_ind",
        );
    }

    /// `&'a U` has no positions: its predicate holds nothing.
    #[test]
    fn shared_reference() {
        let src = "type X<'a, T> = &'a T;";
        assert_verifies(
            src,
            r#"
            method test(k: s_Ref_immutable, U: Type)
              ensures p_Ref_immutable_ind(k, U, p_Ref_immutable_lft_0())
              ensures extend_Ref_immutable(k, U, p_Ref_immutable_lft_0()) == k
            {
              fold p_Ref_immutable_ind(k, U, p_Ref_immutable_lft_0())
            }
            "#,
        );
        assert_fails(
            src,
            r#"
            method test(k: s_Ref_immutable, U: Type)
              requires p_Ref_immutable_ind(k, U, p_Ref_immutable_lft_0())
            {
              unfold p_Ref_immutable_ind(k, U, p_Ref_immutable_lft_0())
              assert false
            }
            "#,
            "Assert might fail",
        );
    }

    /// `'static` is not a position of a struct: no position holds the
    /// referent of a `&'static mut`.
    #[test]
    fn static_mutable_reference() {
        let src = "struct S<'a> { s: &'static mut i32, t: &'a mut i32 }
                   type X<'a> = S<'a>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_S)
              requires p_S_ind(k, p_S_lft_0())
              ensures p_Ref_mutable_ind(k.s_S_1, s_Int_i32_type(), p_Ref_mutable_lft_0())
            {
              unfold p_S_ind(k, p_S_lft_0())
            }
            "#,
        );
        assert_fails(
            src,
            r#"
            method test(k: s_S)
              requires p_S_ind(k, p_S_lft_0())
              ensures p_Ref_mutable_ind(k.s_S_0, s_Int_i32_type(), p_Ref_mutable_lft_0())
            {
              unfold p_S_ind(k, p_S_lft_0())
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
    }

    /// A struct passes its positions on to its fields: `'a` is `p_S_lft_0()`
    /// and the positions of `T` are `p_S_ty_0(p_Param_all())`.
    #[test]
    fn struct_fields() {
        let src = "struct S<'a, T> { x: &'a mut i32, y: T }
                   type X<'a, T> = S<'a, T>;";
        assert_verifies(
            src,
            r#"
            method lifetime(k: s_S, T: Type)
              requires p_S_ind(k, T, p_S_lft_0())
              ensures p_Ref_mutable_ind(k.s_S_0, s_Int_i32_type(), p_Ref_mutable_lft_0())
            {
              unfold p_S_ind(k, T, p_S_lft_0())
            }

            method type_param(k: s_S, T: Type)
              requires p_S_ind(k, T, p_S_ty_0(p_Param_all()))
              ensures p_Param_ind(k.s_S_1, T, p_Param_all())
            {
              unfold p_S_ind(k, T, p_S_ty_0(p_Param_all()))
            }

            method split(k: s_S, T: Type)
              requires p_S_ind(k, T, p_S_all())
              ensures p_S_ind(k, T, p_S_lft_0()) && p_S_ind(k, T, p_S_ty_0(p_Param_all()))
            {
              unfold p_S_ind(k, T, p_S_all())
              fold p_S_ind(k, T, p_S_lft_0())
              fold p_S_ind(k, T, p_S_ty_0(p_Param_all()))
            }

            method merge(k: s_S, T: Type)
              requires p_S_ind(k, T, p_S_lft_0()) && p_S_ind(k, T, p_S_ty_0(p_Param_all()))
              ensures p_S_ind(k, T, p_S_all())
            {
              unfold p_S_ind(k, T, p_S_lft_0())
              unfold p_S_ind(k, T, p_S_ty_0(p_Param_all()))
              fold p_S_ind(k, T, p_S_all())
            }
            "#,
        );
        // The positions of T do not hold `x`.
        assert_fails(
            src,
            r#"
            method test(k: s_S, T: Type)
              requires p_S_ind(k, T, p_S_ty_0(p_Param_all()))
              ensures p_Ref_mutable_ind(k.s_S_0, s_Int_i32_type(), p_Ref_mutable_lft_0())
            {
              unfold p_S_ind(k, T, p_S_ty_0(p_Param_all()))
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
        // 'a does not hold `y`.
        assert_fails(
            src,
            r#"
            method test(k: s_S, T: Type)
              requires p_S_ind(k, T, p_S_lft_0())
              ensures p_Param_ind(k.s_S_1, T, p_Param_all())
            {
              unfold p_S_ind(k, T, p_S_lft_0())
            }
            "#,
            "insufficient permission to access p_Param_ind",
        );
    }

    /// The positions of a tuple are those of its elements in turn.
    #[test]
    fn tuple() {
        let src = "type X<T, U> = (T, U);";
        assert_verifies(
            src,
            r#"
            method elements(k: s_2_Tuple, T: Type, U: Type)
              requires p_2_Tuple_ind(k, T, U, p_2_Tuple_all())
              ensures p_Param_ind(k.s_2_Tuple_0, T, p_Param_all())
              ensures p_Param_ind(k.s_2_Tuple_1, U, p_Param_all())
            {
              unfold p_2_Tuple_ind(k, T, U, p_2_Tuple_all())
            }

            method split(k: s_2_Tuple, T: Type, U: Type)
              requires p_2_Tuple_ind(k, T, U, p_2_Tuple_all())
              ensures p_2_Tuple_ind(k, T, U, p_2_Tuple_ty_0(p_Param_all()))
              ensures p_2_Tuple_ind(k, T, U, p_2_Tuple_ty_1(p_Param_all()))
            {
              unfold p_2_Tuple_ind(k, T, U, p_2_Tuple_all())
              fold p_2_Tuple_ind(k, T, U, p_2_Tuple_ty_0(p_Param_all()))
              fold p_2_Tuple_ind(k, T, U, p_2_Tuple_ty_1(p_Param_all()))
            }
            "#,
        );
        // The positions of T do not hold those of U.
        assert_fails(
            src,
            r#"
            method test(k: s_2_Tuple, T: Type, U: Type)
              requires p_2_Tuple_ind(k, T, U, p_2_Tuple_ty_0(p_Param_all()))
              ensures p_Param_ind(k.s_2_Tuple_1, U, p_Param_all())
            {
              unfold p_2_Tuple_ind(k, T, U, p_2_Tuple_ty_0(p_Param_all()))
            }
            "#,
            "insufficient permission to access p_Param_ind",
        );
    }

    /// In the field type `Three<'a, 'b, 'a>`, `'a` occurs at positions 0
    /// and 2, with `'b` in between.
    #[test]
    fn duplicate_lifetimes() {
        let src = "struct Three<'x, 'y, 'z>(&'x mut i32, &'y mut i32, &'z mut i32);
                   struct S<'a, 'b> { f: Three<'a, 'b, 'a> }
                   type X<'a, 'b> = S<'a, 'b>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_S)
              requires p_S_ind(k, p_S_lft_0())
              ensures p_Three_ind(k.s_S_0, p_Three_lft_0()) && p_Three_ind(k.s_S_0, p_Three_lft_2())
            {
              unfold p_S_ind(k, p_S_lft_0())
            }
            "#,
        );
        // 'a does not hold the position of 'b.
        assert_fails(
            src,
            r#"
            method test(k: s_S)
              requires p_S_ind(k, p_S_lft_0())
              ensures p_Three_ind(k.s_S_0, p_Three_lft_1())
            {
              unfold p_S_ind(k, p_S_lft_0())
            }
            "#,
            "insufficient permission to access p_Three_ind",
        );
    }

    /// In the field type `(T, U, T)`, `T` occurs twice, with `U` in between.
    #[test]
    fn duplicate_type_params() {
        let src = "struct S<T, U> { f: (T, U, T) }
                   type X<T, U> = S<T, U>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_S, T: Type, U: Type)
              requires p_S_ind(k, T, U, p_S_ty_0(p_Param_all()))
              ensures p_Param_ind(k.s_S_0.s_3_Tuple_0, T, p_Param_all())
              ensures p_Param_ind(k.s_S_0.s_3_Tuple_2, T, p_Param_all())
            {
              unfold p_S_ind(k, T, U, p_S_ty_0(p_Param_all()))
              unfold p_3_Tuple_ind(k.s_S_0, T, U, T, p_3_Tuple_ty_0(p_Param_all()))
              unfold p_3_Tuple_ind(k.s_S_0, T, U, T, p_3_Tuple_ty_2(p_Param_all()))
            }
            "#,
        );
        // The positions of T do not hold those of U.
        assert_fails(
            src,
            r#"
            method test(k: s_S, T: Type, U: Type)
              requires p_S_ind(k, T, U, p_S_ty_0(p_Param_all()))
              ensures p_Param_ind(k.s_S_0.s_3_Tuple_1, U, p_Param_all())
            {
              unfold p_S_ind(k, T, U, p_S_ty_0(p_Param_all()))
              unfold p_3_Tuple_ind(k.s_S_0, T, U, T, p_3_Tuple_ty_0(p_Param_all()))
              unfold p_3_Tuple_ind(k.s_S_0, T, U, T, p_3_Tuple_ty_2(p_Param_all()))
            }
            "#,
            "insufficient permission to access p_Param_ind",
        );
    }

    /// In `&'a mut &'b mut i32`, the position of `'b` lies behind `'a`. Split
    /// off `'a`, it is on the snapshot extended by `'a`.
    #[test]
    fn nested_mutable_references() {
        let src = "struct S<'a, 'b> { r: &'a mut &'b mut i32 }
                   type X<'a, 'b> = S<'a, 'b>;";
        assert_verifies(
            src,
            r#"
            method split(k: s_S)
              requires p_S_ind(k, p_S_all())
              ensures p_S_ind(k, p_S_lft_0()) && p_S_ind(extend_S(k, p_S_lft_0()), p_S_lft_1())
            {
              unfold p_S_ind(k, p_S_all())
              fold p_S_ind(k, p_S_lft_0())
              fold p_S_ind(extend_S(k, p_S_lft_0()), p_S_lft_1())
            }
            "#,
        );
        // Not on the snapshot that is not extended.
        assert_fails(
            src,
            r#"
            method split(k: s_S)
              requires p_S_ind(k, p_S_all())
              ensures p_S_ind(k, p_S_lft_0()) && p_S_ind(k, p_S_lft_1())
            {
              unfold p_S_ind(k, p_S_all())
              fold p_S_ind(k, p_S_lft_0())
              fold p_S_ind(k, p_S_lft_1())
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
    }

    /// In `N<'b, 'a>`, `'b` comes first but lies behind `'a`. Extending by both
    /// positions is extending by `'a`, then by `'b` on the result.
    #[test]
    fn enclosed_lifetime_first() {
        let src = "struct N<'b, 'a>(&'a mut &'b mut i32);
                   type X<'b, 'a> = N<'b, 'a>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_N)
              requires p_N_ind(k, p_N_all())
            {
              var all: s_N := extend_N(k, p_N_all())
              unfold p_N_ind(k, p_N_all())
              fold p_N_ind(k, p_N_lft_1())
              fold p_N_ind(extend_N(k, p_N_lft_1()), p_N_lft_0())
              assert all == extend_N(extend_N(k, p_N_lft_1()), p_N_lft_0())
            }
            "#,
        );
        assert_fails(
            src,
            r#"
            method test(k: s_N)
              requires p_N_ind(k, p_N_all())
            {
              var all: s_N := extend_N(k, p_N_all())
              unfold p_N_ind(k, p_N_all())
              fold p_N_ind(k, p_N_lft_1())
              fold p_N_ind(extend_N(k, p_N_lft_1()), p_N_lft_0())
              assert false
            }
            "#,
            "Assert might fail",
        );
    }

    /// In `&'a mut &'a mut i32`, the predicate puts the inner `'a` on the
    /// snapshot extended by the outer one.
    #[test]
    fn nested_mutable_references_same_lifetime() {
        let src = "struct E<'a>(&'a mut &'a mut i32);
                   type X<'a> = E<'a>;";
        assert_verifies(
            src,
            r#"
            // The position of the inner 'a in the field.
            define RB p_Ref_mutable_ty_0(p_Param_Ref_mutable(p_Ref_mutable_lft_0()))
            method test(k: s_E)
              requires p_E_ind(k, p_E_lft_0())
              ensures p_Ref_mutable_ind(k.s_E_0, s_Ref_mutable_type(s_Int_i32_type()), p_Ref_mutable_lft_0())
              ensures p_Ref_mutable_ind(extend_Ref_mutable(k.s_E_0, s_Ref_mutable_type(s_Int_i32_type()), p_Ref_mutable_lft_0()), s_Ref_mutable_type(s_Int_i32_type()), RB)
            {
              unfold p_E_ind(k, p_E_lft_0())
            }
            "#,
        );
        // Not on the field snapshot that is not extended.
        assert_fails(
            src,
            r#"
            // The position of the inner 'a in the field.
            define RB p_Ref_mutable_ty_0(p_Param_Ref_mutable(p_Ref_mutable_lft_0()))
            method test(k: s_E)
              requires p_E_ind(k, p_E_lft_0())
              ensures p_Ref_mutable_ind(k.s_E_0, s_Ref_mutable_type(s_Int_i32_type()), RB)
            {
              unfold p_E_ind(k, p_E_lft_0())
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
    }

    /// An enum passes its positions on to the fields of the variant it is
    /// in: `'a` to variant `A`, the positions of `T` to variant `B`.
    #[test]
    fn enum_variants() {
        let src = "enum E<'a, T> { A(&'a mut i32), B(T), C }
                   type X<'a, T> = E<'a, T>;";
        assert_verifies(
            src,
            r#"
            method variant_a(k: s_E, T: Type)
              requires p_E_ind(k, T, p_E_lft_0()) && k.iss_E_0_cons
              ensures p_Ref_mutable_ind(k.s_E_0_0, s_Int_i32_type(), p_Ref_mutable_lft_0())
            {
              unfold p_E_ind(k, T, p_E_lft_0())
            }

            method variant_b(k: s_E, T: Type)
              requires p_E_ind(k, T, p_E_ty_0(p_Param_all())) && k.iss_E_1_cons
              ensures p_Param_ind(k.s_E_1_0, T, p_Param_all())
            {
              unfold p_E_ind(k, T, p_E_ty_0(p_Param_all()))
            }

            // Variant `C` holds nothing.
            method variant_c(k: s_E, T: Type)
              requires k.iss_E_2_cons
              ensures p_E_ind(k, T, p_E_all())
            {
              fold p_E_ind(k, T, p_E_all())
            }

            // Extending keeps the variant.
            method extend(k: s_E, T: Type)
              requires p_E_ind(k, T, p_E_lft_0())
              ensures p_E_ind(k, T, p_E_lft_0())
              ensures s_E_discr(extend_E(k, T, p_E_lft_0())) == s_E_discr(k)
            {
            }

            method split(k: s_E, T: Type)
              requires p_E_ind(k, T, p_E_all())
              ensures p_E_ind(k, T, p_E_lft_0()) && p_E_ind(k, T, p_E_ty_0(p_Param_all()))
            {
              unfold p_E_ind(k, T, p_E_all())
              fold p_E_ind(k, T, p_E_lft_0())
              fold p_E_ind(k, T, p_E_ty_0(p_Param_all()))
            }
            "#,
        );
        // Without knowing the variant, the field is not held.
        assert_fails(
            src,
            r#"
            method test(k: s_E, T: Type)
              requires p_E_ind(k, T, p_E_lft_0())
              ensures p_Ref_mutable_ind(k.s_E_0_0, s_Int_i32_type(), p_Ref_mutable_lft_0())
            {
              unfold p_E_ind(k, T, p_E_lft_0())
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
        // In variant `B`, 'a holds nothing.
        assert_fails(
            src,
            r#"
            method test(k: s_E, T: Type)
              requires p_E_ind(k, T, p_E_lft_0()) && k.iss_E_1_cons
              ensures p_Ref_mutable_ind(k.s_E_0_0, s_Int_i32_type(), p_Ref_mutable_lft_0())
            {
              unfold p_E_ind(k, T, p_E_lft_0())
            }
            "#,
            "insufficient permission to access p_Ref_mutable_ind",
        );
    }

    /// The casts let `Option<&'a mut U>`'s predicate be unfolded down to the
    /// referent.
    #[test]
    fn casts() {
        let src = "type X<'a, T> = Option<&'a mut T>;";
        let result = verify_with_casts(
            src,
            r#"
            // The position of 'a in `Option<&'a mut U>`.
            define OPT_A p_Option_ty_0(p_Param_Ref_mutable(p_Ref_mutable_lft_0()))
            // The snapshot of the field of `Some` is that of a reference
            // once its type is known.
            method unfold_option(k: s_Option, U: Type)
              requires p_Option_ind(k, s_Ref_mutable_type(U), OPT_A) && k.iss_Option_1_cons
              requires s_Param_typ(k.s_Option_1_0) == s_Ref_mutable_type(U)
              ensures p_Param(k.s_Option_1_0.make_concrete_s_Ref_mutable.s_Ref_mutable_0, U)
            {
              unfold p_Option_ind(k, s_Ref_mutable_type(U), OPT_A)
              make_concrete_ind_Ref_mutable(k.s_Option_1_0.make_concrete_s_Ref_mutable, U, p_Ref_mutable_lft_0())
              unfold p_Ref_mutable_ind(k.s_Option_1_0.make_concrete_s_Ref_mutable, U, p_Ref_mutable_lft_0())
            }

            // Making a snapshot shallow is the same on both sides of a cast.
            method shallow(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
            {
              make_generic_ind_Ref_mutable(k, U, p_Ref_mutable_lft_0())
              assert shallow_Param(make_generic_s_Ref_mutable(k, U), s_Ref_mutable_type(U)) ==
                make_generic_s_Ref_mutable(shallow_Ref_mutable(k, U), U)
            }

            // Casting back and forth keeps what extending reads.
            method round_trip(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures extend_Ref_mutable(k, U, p_Ref_mutable_lft_0()) == old(extend_Ref_mutable(k, U, p_Ref_mutable_lft_0()))
            {
              make_generic_ind_Ref_mutable(k, U, p_Ref_mutable_lft_0())
              make_concrete_ind_Ref_mutable(k, U, p_Ref_mutable_lft_0())
            }

            // What extending the option reads is what extending the reference reads.
            method extend_option(k: s_Option, U: Type)
              requires p_Option_ind(k, s_Ref_mutable_type(U), OPT_A) && k.iss_Option_1_cons
              requires s_Param_typ(k.s_Option_1_0) == s_Ref_mutable_type(U)
              ensures p_Ref_mutable_ind(k.s_Option_1_0.make_concrete_s_Ref_mutable, U, p_Ref_mutable_lft_0())
              ensures old(extend_Option(k, s_Ref_mutable_type(U), OPT_A)).s_Option_1_0 ==
                make_generic_s_Ref_mutable(extend_Ref_mutable(
                  k.s_Option_1_0.make_concrete_s_Ref_mutable, U, p_Ref_mutable_lft_0()), U)
            {
              unfold p_Option_ind(k, s_Ref_mutable_type(U), OPT_A)
              make_concrete_ind_Ref_mutable(k.s_Option_1_0.make_concrete_s_Ref_mutable, U, p_Ref_mutable_lft_0())
            }
            "#,
        );
        if let Err(output) = result {
            panic!("verification failed:\n{output}");
        }
        // The casts do not make the state inconsistent.
        let result = verify_with_casts(
            src,
            r#"
            // The position of 'a in `Option<&'a mut U>`.
            define OPT_A p_Option_ty_0(p_Param_Ref_mutable(p_Ref_mutable_lft_0()))
            method test(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
            {
              make_generic_ind_Ref_mutable(k, U, p_Ref_mutable_lft_0())
              make_concrete_ind_Ref_mutable(k, U, p_Ref_mutable_lft_0())
              assert false
            }
            "#,
        );
        match result {
            Ok(()) => panic!("verification succeeded"),
            Err(output) => assert!(
                output.contains("Assert might fail"),
                "verification failed differently:\n{output}"
            ),
        }
    }

    /// 'a encloses 'b in `B` but not in `A`, so `A`'s predicate for 'b is on
    /// the shallow field snapshot.
    #[test]
    fn field_with_fewer_enclosing_positions() {
        let src = "struct A<'a, 'b>(&'a mut i32, &'b mut i32);
                   struct B<'a, 'b>(A<'a, 'b>, &'a mut &'b mut i32);
                   type X<'a, 'b> = B<'a, 'b>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_B)
              requires p_B_ind(k, p_B_lft_0())
              requires p_B_ind(extend_B(k, p_B_lft_0()), p_B_lft_1())
              ensures p_A_ind(shallow_A(k.s_B_0), p_A_lft_1())
            {
              unfold p_B_ind(extend_B(k, p_B_lft_0()), p_B_lft_1())
            }
            "#,
        );
        assert_fails(
            src,
            r#"
            method test(k: s_B)
              requires p_B_ind(k, p_B_lft_0())
              requires p_B_ind(extend_B(k, p_B_lft_0()), p_B_lft_1())
            {
              unfold p_B_ind(extend_B(k, p_B_lft_0()), p_B_lft_1())
              assert false
            }
            "#,
            "Assert might fail",
        );
        // The field's snapshot itself may carry what extending by 'a read.
        assert_fails(
            src,
            r#"
            method test(k: s_B)
              requires p_B_ind(k, p_B_lft_0())
              requires p_B_ind(extend_B(k, p_B_lft_0()), p_B_lft_1())
              ensures p_A_ind(k.s_B_0, p_A_lft_1())
            {
              unfold p_B_ind(extend_B(k, p_B_lft_0()), p_B_lft_1())
            }
            "#,
            "insufficient permission to access p_A_ind",
        );
    }

    /// Like `field_with_fewer_enclosing_positions`, with a recursive field
    /// type: the field of an unextended snapshot gets `List`'s predicate for
    /// 'b on its own snapshot, though the predicate is on the shallow field
    /// after extending by 'a.
    #[test]
    fn recursive_field_with_fewer_enclosing_positions() {
        let src = "struct List<'a, 'b> {
                       x: &'a mut i32,
                       y: &'b mut i32,
                       next: Option<Box<List<'a, 'b>>>,
                   }
                   struct Outer<'a, 'b>(List<'a, 'b>, &'a mut &'b mut i32);
                   type X<'a, 'b> = Outer<'a, 'b>;";
        assert_verifies(
            src,
            r#"
            method test(k: s_Outer)
              requires shallow_Outer(k) == k
              requires p_Outer_ind(k, p_Outer_lft_0()) && p_Outer_ind(extend_Outer(k, p_Outer_lft_0()), p_Outer_lft_1())
              ensures p_List_ind(k.s_Outer_0, p_List_lft_1())
            {
              unfold p_Outer_ind(extend_Outer(k, p_Outer_lft_0()), p_Outer_lft_1())
            }
            "#,
        );
        assert_fails(
            src,
            r#"
            method test(k: s_Outer)
              requires shallow_Outer(k) == k
              requires p_Outer_ind(k, p_Outer_lft_0()) && p_Outer_ind(extend_Outer(k, p_Outer_lft_0()), p_Outer_lft_1())
            {
              unfold p_Outer_ind(extend_Outer(k, p_Outer_lft_0()), p_Outer_lft_1())
              assert false
            }
            "#,
            "Assert might fail",
        );
    }

    /// `List`'s predicate for 'b on the shallow field of `Outer` can be followed
    /// into the next element: the snapshot in `Option` is shallow, which the
    /// cast of `Box<List>` turns into the shallow concrete snapshot.
    #[test]
    fn recursive_field_into_next() {
        // The tuple's second element provides the casts of `Box<List>`.
        let src = "struct List<'a, 'b> {
                       x: &'a mut i32,
                       y: &'b mut i32,
                       next: Option<Box<List<'a, 'b>>>,
                   }
                   struct Outer<'a, 'b>(List<'a, 'b>, &'a mut &'b mut i32);
                   type X<'a, 'b> = (Outer<'a, 'b>, Box<List<'a, 'b>>);";
        let result = verify_with_casts(
            src,
            r#"
            // The position of 'b in `Box<List<'a, 'b>>` and in `Option<Box<List<'a, 'b>>>`.
            define BOX_B p_Box_ty_0(p_Param_List(p_List_lft_1()))
            define OPT_B p_Option_ty_0(p_Param_Box(BOX_B))
            method test(k: s_Outer, b: s_Box)
              requires p_Outer_ind(k, p_Outer_lft_0()) && p_Outer_ind(extend_Outer(k, p_Outer_lft_0()), p_Outer_lft_1())
              requires k.s_Outer_0.s_List_2 ==
                s_Option_1_cons(make_generic_s_Box(b, s_List_type(), s_Global_type()))
              ensures p_Box_ind(shallow_Box(b, s_List_type(), s_Global_type()), s_List_type(), s_Global_type(), BOX_B)
            {
              unfold p_Outer_ind(extend_Outer(k, p_Outer_lft_0()), p_Outer_lft_1())
              unfold p_List_ind(shallow_List(extend_Outer(k, p_Outer_lft_0()).s_Outer_0), p_List_lft_1())
              unfold p_Option_ind(shallow_List(extend_Outer(k, p_Outer_lft_0()).s_Outer_0).s_List_2,
                s_Box_type(s_List_type(), s_Global_type()), OPT_B)
              make_concrete_ind_Box(shallow_Box(b, s_List_type(), s_Global_type()),
                s_List_type(), s_Global_type(), BOX_B)
            }
            "#,
        );
        if let Err(output) = result {
            panic!("verification failed:\n{output}");
        }
        let result = verify_with_casts(
            src,
            r#"
            // The position of 'b in `Box<List<'a, 'b>>` and in `Option<Box<List<'a, 'b>>>`.
            define BOX_B p_Box_ty_0(p_Param_List(p_List_lft_1()))
            define OPT_B p_Option_ty_0(p_Param_Box(BOX_B))
            method test(k: s_Outer, b: s_Box)
              requires p_Outer_ind(k, p_Outer_lft_0()) && p_Outer_ind(extend_Outer(k, p_Outer_lft_0()), p_Outer_lft_1())
              requires k.s_Outer_0.s_List_2 ==
                s_Option_1_cons(make_generic_s_Box(b, s_List_type(), s_Global_type()))
            {
              unfold p_Outer_ind(extend_Outer(k, p_Outer_lft_0()), p_Outer_lft_1())
              unfold p_List_ind(shallow_List(extend_Outer(k, p_Outer_lft_0()).s_Outer_0), p_List_lft_1())
              unfold p_Option_ind(shallow_List(extend_Outer(k, p_Outer_lft_0()).s_Outer_0).s_List_2,
                s_Box_type(s_List_type(), s_Global_type()), OPT_B)
              make_concrete_ind_Box(shallow_Box(b, s_List_type(), s_Global_type()),
                s_List_type(), s_Global_type(), BOX_B)
              assert false
            }
            "#,
        );
        match result {
            Ok(()) => panic!("verification succeeded"),
            Err(output) => assert!(
                output.contains("Assert might fail"),
                "verification failed differently:\n{output}"
            ),
        }
    }

    /// 'y encloses 'z in `C` but not in `F`, and lies behind 'x in both: `C`
    /// gives `F`'s predicates for 'y and 'z on one snapshot, the field
    /// extended by 'x, as `F` has them.
    #[test]
    fn field_with_fewer_enclosing_positions_behind_reference() {
        let src = "struct F<'x, 'y, 'z>(&'x mut (&'y mut i32, &'z mut i32));
                   struct C<'x, 'y, 'z>(F<'x, 'y, 'z>, &'y mut &'z mut i32);
                   type X<'x, 'y, 'z> = C<'x, 'y, 'z>;";
        let keys = r#"
            // The keys of 'y and 'z: `k` extended by their ancestors.
            define KY(k) extend_C(k, p_C_lft_0())
            define KZ(k) extend_C(KY(k), p_C_lft_1())
            "#;
        assert_verifies(
            src,
            &format!(
                "{keys}{}",
                r#"
            method test(k: s_C)
              requires p_C_ind(k, p_C_lft_0()) && p_C_ind(KY(k), p_C_lft_1()) && p_C_ind(KZ(k), p_C_lft_2())
              ensures p_C_ind(k, p_C_lft_0())
              ensures p_F_ind(KY(k).s_C_0, p_F_lft_1()) && p_F_ind(KY(k).s_C_0, p_F_lft_2())
            {
              // 'z first: its key reads through the predicate of 'y.
              unfold p_C_ind(KZ(k), p_C_lft_2())
              unfold p_C_ind(KY(k), p_C_lft_1())
            }
            "#
            ),
        );
        assert_fails(
            src,
            &format!(
                "{keys}{}",
                r#"
            method test(k: s_C)
              requires p_C_ind(k, p_C_lft_0()) && p_C_ind(KY(k), p_C_lft_1()) && p_C_ind(KZ(k), p_C_lft_2())
            {
              unfold p_C_ind(KZ(k), p_C_lft_2())
              unfold p_C_ind(KY(k), p_C_lft_1())
              assert false
            }
            "#
            ),
            "Assert might fail",
        );
    }

    /// 'l encloses `T` in `D` but not in the tuple, so `D` resets the tuple's
    /// 'l, but leaves all positions of `T` as its key has them: `D` does not
    /// know which of them enclose the query `q`, the instantiation does. With
    /// `T = S<'a, 'b, 'c>`, where 'b and 'c enclose 'a, the key of 'a is
    /// extended by 'b and 'c.
    #[test]
    fn type_param_field_with_fewer_enclosing_positions() {
        let src = "struct D<'l, T> { c: &'l mut T, d: (&'l mut i32, T) }
                   type X<'l, T> = D<'l, T>;";
        let tuple = r#"
            define R s_Ref_mutable_type(s_Int_i32_type())
            define KEEP_T(k, T) graft_2_Tuple(shallow_2_Tuple(k.s_D_1, R, T), R, T,
              k.s_D_1, p_2_Tuple_ty_1(p_Param_all()))
            "#;
        assert_verifies(
            src,
            &format!(
                "{tuple}{}",
                r#"
            method test(k: s_D, T: Type, q: p_Param_lft_index)
              requires p_D_ind(k, T, p_D_ty_0(q))
              ensures p_2_Tuple_ind(KEEP_T(k, T), R, T, p_2_Tuple_ty_1(q))
            {
              unfold p_D_ind(k, T, p_D_ty_0(q))
            }
            "#
            ),
        );
        assert_fails(
            src,
            &format!(
                "{tuple}{}",
                r#"
            method test(k: s_D, T: Type, q: p_Param_lft_index)
              requires p_D_ind(k, T, p_D_ty_0(q))
            {
              unfold p_D_ind(k, T, p_D_ty_0(q))
              assert false
            }
            "#
            ),
            "Assert might fail",
        );
    }

    /// Resetting a mutable reference gives it the snapshot its direct
    /// predicate gives it.
    #[test]
    fn shallow_mutable_reference() {
        let src = "type X<'a, T> = &'a mut T;";
        assert_verifies(
            src,
            r#"
            method unextended(a: Ref, m: s_Param, U: Type)
              ensures shallow_Ref_mutable(p_Ref_mutable_arbitrary_value(a, m, U), U) ==
                p_Ref_mutable_arbitrary_value(a, m, U)
            {
            }

            method extended(k: s_Ref_mutable, U: Type)
              requires p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures p_Ref_mutable_ind(k, U, p_Ref_mutable_lft_0())
              ensures shallow_Ref_mutable(extend_Ref_mutable(k, U, p_Ref_mutable_lft_0()), U) == shallow_Ref_mutable(k, U)
            {
            }
            "#,
        );
        // An arbitrary snapshot need not be unextended.
        assert_fails(
            src,
            r#"
            method test(k: s_Ref_mutable, U: Type)
              ensures shallow_Ref_mutable(k, U) == k
            {
            }
            "#,
            "Postcondition of test might not hold",
        );
    }
}
