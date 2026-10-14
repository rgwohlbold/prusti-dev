//! The occurrences of lifetimes (and type parameters) in a type expression
//! that indirect pieces cover, and which of them enclose which.
//!
//! An occurrence is a path through the type's arguments: into type argument
//! `j` (`In(j)`), ending at lifetime argument `i` (`Here(i)`) or at a type
//! parameter. Only occurrences reached through `&mut` referents and the fields
//! of structs/enums count (not those behind shared references, arrays or raw
//! pointers). `a` encloses `b` if `a` is the lifetime of a `&mut` whose
//! referent contains `b`, either syntactically or through the definition of a
//! type constructor (e.g. `Node<'a, 'b>`: `'a` encloses `'b`).

use std::collections::{BTreeSet, HashMap};

use prusti_rustc_interface::middle::ty;

use super::{
    RustParamData, RustTy, RustTyDecomposition, TySpecifics, generics::GParams,
    lifetime_query::LifetimeQueryFns,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Step {
    /// The lifetime argument `i`.
    Here(usize),
    /// Into the type argument `j`.
    In(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum End<'tcx> {
    Region(ty::Region<'tcx>),
    /// A type parameter: the occurrence continues inside its instantiation.
    Param(ty::ParamTy),
}

#[derive(Debug, Clone)]
pub struct Occurrence<'tcx> {
    /// `In` steps, then `Here(i)` for a region.
    pub path: Vec<Step>,
    pub end: End<'tcx>,
    /// The (indices of the) occurrences enclosing this one.
    pub parents: Vec<usize>,
}

/// The occurrences of `decomp`, enclosing ones first.
pub fn occurrences<'tcx>(decomp: RustTyDecomposition<'tcx>) -> Result<Vec<Occurrence<'tcx>>, String> {
    let mut walker = Walker::default();
    let mut found = Found::default();
    walker.walk(decomp, &mut Vec::new(), &mut found);
    sort(found).ok_or_else(|| format!("lifetimes enclosing each other are not supported ({decomp:?})"))
}

/// The query selecting the occurrence `path`; for a type parameter
/// occurrence, `rest` selects inside it.
pub fn occurrence_query<'vir>(
    fns: &LifetimeQueryFns<'vir>,
    path: &[Step],
    rest: Option<vir::ExprLifetimeQuery<'vir>>,
) -> vir::ExprLifetimeQuery<'vir> {
    let (init, last) = match path.last() {
        Some(Step::Here(i)) => (&path[..path.len() - 1], fns.here(*i)),
        _ => (path, rest.expect("a type parameter occurrence needs the rest of its query")),
    };
    init.iter().rev().fold(last, |q, step| match step {
        Step::In(j) => fns.in_arg(*j, q),
        Step::Here(_) => unreachable!("`Here` ends a path"),
    })
}

/// The position of the generic parameter `index` of `params` among those of
/// its kind.
pub fn param_position<'tcx>(
    params: GParams<'tcx>,
    index: u32,
    of_kind: impl Fn(ty::GenericArg<'tcx>) -> bool,
) -> usize {
    params.rust_params()[..index as usize]
        .iter()
        .filter(|arg| of_kind(**arg))
        .count()
}

#[derive(Default)]
struct Found<'tcx> {
    list: Vec<(Vec<Step>, End<'tcx>)>,
    /// `(a, b)`: `a` encloses `b`
    pairs: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Target {
    Lifetime(usize),
    Ty(usize),
}

/// What a type constructor's definition covers, in terms of its parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Summary {
    lifetimes: BTreeSet<usize>,
    tys: BTreeSet<usize>,
    /// `(i, t)`: lifetime parameter `i` encloses `t`.
    rel: BTreeSet<(usize, Target)>,
}

#[derive(Default)]
struct Walker<'tcx> {
    done: HashMap<RustTy<'tcx>, Summary>,
    /// Summaries being computed, with their current approximation (for
    /// recursive types; iterated to a fixpoint).
    stack: Vec<(RustTy<'tcx>, Summary)>,
}

impl<'tcx> Walker<'tcx> {
    /// Adds the occurrences of `decomp` (at `prefix`) to `found`; returns the
    /// range of their indices.
    fn walk(
        &mut self,
        decomp: RustTyDecomposition<'tcx>,
        prefix: &mut Vec<Step>,
        found: &mut Found<'tcx>,
    ) -> std::ops::Range<usize> {
        let start = found.list.len();
        match &decomp.ty.specifics {
            TySpecifics::Param(RustParamData::Generic) => {
                // TODO: lifetimes inside associated types are not covered.
                if let ty::TyKind::Param(param) = decomp.args.args()[0].expect_ty().kind() {
                    found.list.push((prefix.clone(), End::Param(*param)));
                }
            }
            TySpecifics::MutRef(_) => {
                let own = found.list.len();
                let region = decomp.args.args()[0].expect_region();
                found.list.push((with(prefix, Step::Here(0)), End::Region(region)));
                let referent = self.descend(decomp, 0, prefix, found);
                found.pairs.extend(referent.map(|b| (own, b)));
            }
            TySpecifics::StructLike(_) | TySpecifics::EnumLike(_) => {
                let summary = self.summary(decomp.ty);
                let mut lifetimes = HashMap::new();
                let mut i = 0;
                for arg in decomp.args.args() {
                    if let Some(region) = arg.as_region() {
                        if summary.lifetimes.contains(&i) {
                            lifetimes.insert(i, found.list.len());
                            found.list.push((with(prefix, Step::Here(i)), End::Region(region)));
                        }
                        i += 1;
                    }
                }
                let mut tys = HashMap::new();
                for j in summary.tys.iter().copied() {
                    tys.insert(j, self.descend(decomp, j, prefix, found));
                }
                for (i, target) in &summary.rel {
                    let Some(a) = lifetimes.get(i) else { continue };
                    match target {
                        Target::Lifetime(i2) => found.pairs.extend(lifetimes.get(i2).map(|b| (*a, *b))),
                        Target::Ty(j) => found.pairs.extend(tys[j].clone().map(|b| (*a, b))),
                    }
                }
            }
            // TODO: shared references, arrays and raw pointers.
            _ => (),
        }
        start..found.list.len()
    }

    /// Walks the type argument `j` of `decomp`.
    fn descend(
        &mut self,
        decomp: RustTyDecomposition<'tcx>,
        j: usize,
        prefix: &mut Vec<Step>,
        found: &mut Found<'tcx>,
    ) -> std::ops::Range<usize> {
        let Some(arg) = decomp.args.args().iter().filter_map(|arg| arg.as_type()).nth(j) else {
            return found.list.len()..found.list.len();
        };
        prefix.push(Step::In(j));
        let range = self.walk(RustTyDecomposition::from_ty(arg, decomp.args.context()), prefix, found);
        prefix.pop();
        range
    }

    fn summary(&mut self, ty: RustTy<'tcx>) -> Summary {
        if let Some(summary) = self.done.get(&ty) {
            return summary.clone();
        }
        if let Some((_, approx)) = self.stack.iter().find(|(t, _)| *t == ty) {
            return approx.clone();
        }
        self.stack.push((ty, Summary::default()));
        let summary = loop {
            let summary = self.summary_from_fields(ty);
            let approx = &mut self.stack.last_mut().unwrap().1;
            if summary == *approx {
                break summary;
            }
            *approx = summary;
        };
        self.stack.pop();
        if self.stack.is_empty() {
            self.done.insert(ty, summary.clone());
        }
        summary
    }

    fn summary_from_fields(&mut self, ty: RustTy<'tcx>) -> Summary {
        let fields = match &ty.specifics {
            TySpecifics::StructLike(data) => data.fields.iter().map(|f| f.ty()).collect::<Vec<_>>(),
            TySpecifics::EnumLike(data) => data
                .variants
                .iter()
                .flat_map(|v| v.inner.fields.iter().map(|f| f.ty()))
                .collect(),
            _ => Vec::new(),
        };
        let mut summary = Summary::default();
        for field in fields {
            let mut found = Found::default();
            self.walk(field.decompose(ty.params), &mut Vec::new(), &mut found);
            let targets = found
                .list
                .iter()
                .map(|(_, end)| param_target(ty.params, *end))
                .collect::<Vec<_>>();
            for target in targets.iter().flatten() {
                match target {
                    Target::Lifetime(i) => summary.lifetimes.insert(*i),
                    Target::Ty(j) => summary.tys.insert(*j),
                };
            }
            for (a, b) in found.pairs {
                if let (Some(Target::Lifetime(i)), Some(target)) = (targets[a], targets[b])
                    && target != Target::Lifetime(i)
                {
                    summary.rel.insert((i, target));
                }
            }
        }
        summary
    }
}

/// The parameter of the constructor (with `params`) that `end` is.
fn param_target<'tcx>(params: GParams<'tcx>, end: End<'tcx>) -> Option<Target> {
    match end {
        End::Region(region) => match region.kind() {
            ty::RegionKind::ReEarlyParam(param) => Some(Target::Lifetime(param_position(
                params,
                param.index,
                |arg| arg.as_region().is_some(),
            ))),
            _ => None,
        },
        End::Param(param) => Some(Target::Ty(param_position(params, param.index, |arg| {
            arg.as_type().is_some()
        }))),
    }
}

fn with(prefix: &[Step], step: Step) -> Vec<Step> {
    let mut path = prefix.to_vec();
    path.push(step);
    path
}

/// Sorts enclosing occurrences first (otherwise in walk order); `None` on a
/// cycle.
fn sort(found: Found<'_>) -> Option<Vec<Occurrence<'_>>> {
    let n = found.list.len();
    let mut order = Vec::with_capacity(n);
    while order.len() < n {
        let next = (0..n).find(|b| {
            !order.contains(b)
                && found
                    .pairs
                    .iter()
                    .all(|(a, b2)| b2 != b || a == b || order.contains(a))
        })?;
        order.push(next);
    }
    let new_index = |old: usize| order.iter().position(|o| *o == old).unwrap();
    Some(
        order
            .iter()
            .map(|old| {
                let (path, end) = found.list[*old].clone();
                let mut parents = found
                    .pairs
                    .iter()
                    .filter(|(a, b)| b == old && a != old)
                    .map(|(a, _)| new_index(*a))
                    .collect::<Vec<_>>();
                parents.sort();
                parents.dedup();
                Occurrence { path, end, parents }
            })
            .collect(),
    )
}
