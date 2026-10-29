//! The occurrences of lifetimes (and type parameters) in a type expression
//! that indirect pieces cover, and which of them enclose which.
//!
//! An occurrence is a path through the type's arguments: into type argument
//! `j` (`In(j)`), ending at lifetime argument `i` (`Here(i)`) or at a type
//! parameter. Only occurrences reached through the fields of structs/enums
//! count, and the own lifetime of a `&mut`: its referent belongs to that
//! lifetime ("outer owns"), so nothing inside it counts (nor behind shared
//! references, arrays or raw pointers). Hence no occurrence encloses another.
//!
//! Every occurrence has a position (see `Pos`): the lifetime positions of a
//! type expression are numbered flat, a constructor's own lifetimes first (in
//! topological order of enclosure, enclosing ones first), then the positions
//! of its type arguments. A type parameter `T` covers `arity(T)` positions.

use rustc_hash::FxHashMap;
use std::collections::BTreeSet;

use prusti_rustc_interface::middle::ty;

use super::{RustParamData, RustTy, RustTyDecomposition, TySpecifics, generics::GParams};

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
    /// The first position the occurrence covers.
    pub pos: Pos,
}

impl<'tcx> Occurrence<'tcx> {
    /// The type parameter whose `arity` positions the occurrence covers;
    /// `None` for a region (one position).
    pub fn width_param(&self) -> Option<ty::ParamTy> {
        match self.end {
            End::Region(_) => None,
            End::Param(param) => Some(param),
        }
    }
}

/// A position: `konst + Σ arity(param)` over `params`, type parameters of the
/// context of the type expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Pos {
    pub konst: usize,
    pub params: Vec<ty::ParamTy>,
}

/// The occurrences of `decomp`, in position order (enclosing ones first).
pub fn occurrences<'tcx>(
    decomp: RustTyDecomposition<'tcx>,
) -> Result<Vec<Occurrence<'tcx>>, String> {
    let mut walker = Walker::default();
    let mut found = Found::default();
    walker.walk(decomp, &mut Vec::new(), &mut found);
    // the own lifetimes of a constructor whose lifetimes enclose each other
    // cannot be ordered
    if found.pairs.iter().any(|(a, b)| a > b) {
        return Err(format!(
            "lifetimes enclosing each other are not supported ({decomp:?})"
        ));
    }
    let mut pos = Pos::default();
    Ok(found
        .list
        .into_iter()
        .enumerate()
        .map(|(idx, (path, end))| {
            let mut parents = found
                .pairs
                .iter()
                .filter(|(a, b)| *b == idx && *a != idx)
                .map(|(a, _)| *a)
                .collect::<Vec<_>>();
            parents.sort();
            parents.dedup();
            let occurrence = Occurrence {
                path,
                end,
                parents,
                pos: pos.clone(),
            };
            match end {
                End::Region(_) => pos.konst += 1,
                End::Param(param) => pos.params.push(param),
            }
            occurrence
        })
        .collect())
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

impl Summary {
    /// The own lifetimes in position order: enclosing ones first, otherwise
    /// in declaration order (on a cycle, the rest in declaration order).
    fn own_order(&self) -> Vec<usize> {
        let mut order = Vec::new();
        while order.len() < self.lifetimes.len() {
            let mut rest = self
                .lifetimes
                .iter()
                .copied()
                .filter(|i| !order.contains(i));
            let next = rest.clone().find(|b| {
                self.rel
                    .iter()
                    .all(|(a, t)| *t != Target::Lifetime(*b) || order.contains(a))
            });
            order.push(next.or_else(|| rest.next()).unwrap());
        }
        order
    }
}

#[derive(Default)]
struct Walker<'tcx> {
    done: FxHashMap<RustTy<'tcx>, Summary>,
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
            // The own lifetime owns everything behind the reference: the
            // referent's lifetimes have no occurrences of their own.
            TySpecifics::MutRef(_) => {
                let region = decomp.args.args()[0].expect_region();
                found
                    .list
                    .push((with(prefix, Step::Here(0)), End::Region(region)));
            }
            TySpecifics::StructLike(_) | TySpecifics::EnumLike(_) => {
                let summary = self.summary(decomp.ty);
                let regions = decomp
                    .args
                    .args()
                    .iter()
                    .filter_map(|arg| arg.as_region())
                    .collect::<Vec<_>>();
                let mut lifetimes = FxHashMap::default();
                for i in summary.own_order() {
                    lifetimes.insert(i, found.list.len());
                    found
                        .list
                        .push((with(prefix, Step::Here(i)), End::Region(regions[i])));
                }
                let mut tys = FxHashMap::default();
                for j in summary.tys.iter().copied() {
                    tys.insert(j, self.descend(decomp, j, prefix, found));
                }
                for (i, target) in &summary.rel {
                    let Some(a) = lifetimes.get(i) else { continue };
                    match target {
                        Target::Lifetime(i2) => {
                            found.pairs.extend(lifetimes.get(i2).map(|b| (*a, *b)))
                        }
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
        let Some(arg) = decomp
            .args
            .args()
            .iter()
            .filter_map(|arg| arg.as_type())
            .nth(j)
        else {
            return found.list.len()..found.list.len();
        };
        prefix.push(Step::In(j));
        let range = self.walk(
            RustTyDecomposition::from_ty(arg, decomp.args.context()),
            prefix,
            found,
        );
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
