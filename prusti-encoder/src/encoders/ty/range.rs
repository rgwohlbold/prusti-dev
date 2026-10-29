//! Integer expressions for the lifetime positions and position ranges of
//! indirect pieces (see `occurrences`).

use prusti_rustc_interface::middle::ty;
use vir::CastType;

use super::{arity::TyArityEnc, occurrences::Pos};

/// The range `lo .. hi` of positions (`hi` exclusive).
#[derive(Debug, Clone, Copy)]
pub struct Range<'vir> {
    pub lo: vir::ExprInt<'vir>,
    pub hi: vir::ExprInt<'vir>,
}

impl<'vir> Range<'vir> {
    /// `lo .. lo + width`
    pub fn starting_at(lo: vir::ExprInt<'vir>, width: vir::ExprInt<'vir>) -> Self {
        Self {
            lo,
            hi: add(lo, width),
        }
    }

    /// `lo <= pos && pos < hi`
    pub fn contains(&self, pos: vir::ExprInt<'vir>) -> vir::ExprBool<'vir> {
        vir::expr! { (([self.lo]) <= ([pos])) && (([pos]) < ([self.hi])) }
    }

    /// `lo < hi`
    pub fn nonempty(&self) -> vir::ExprBool<'vir> {
        vir::expr! { ([self.lo]) < ([self.hi]) }
    }

    /// The range moved up by `offset`.
    pub fn shift(&self, offset: vir::ExprInt<'vir>) -> Self {
        Self {
            lo: add(offset, self.lo),
            hi: add(offset, self.hi),
        }
    }

    /// The part of the range inside the positions `offset .. offset + width`,
    /// numbered from `offset` (so within `0 .. width`; empty if disjoint).
    pub fn restrict(&self, offset: vir::ExprInt<'vir>, width: vir::ExprInt<'vir>) -> Self {
        let local = |x| {
            if let (Some(x), Some(offset), Some(width)) =
                (as_const(x), as_const(offset), as_const(width))
            {
                return int(x.saturating_sub(offset).min(width));
            }
            clamp(sub(x, offset), int(0), width)
        };
        Self {
            lo: local(self.lo),
            hi: local(self.hi),
        }
    }

    /// `lo` and `hi` in any expression context (they are fully reified).
    pub fn args<Curr, Next>(
        &self,
    ) -> (
        vir::ExprGenInt<'vir, Curr, Next>,
        vir::ExprGenInt<'vir, Curr, Next>,
    ) {
        // SAFETY: see `GArgsTy::get_ty`.
        let any_ctx = |e: vir::ExprInt<'vir>| unsafe {
            &*(e as *const vir::ExprData<'vir, vir::Int>
                as *const vir::ExprGenData<'vir, Curr, Next, vir::Int>)
        };
        (any_ctx(self.lo), any_ctx(self.hi))
    }
}

/// The integer constant `value`.
pub fn int<'vir>(value: usize) -> vir::ExprInt<'vir> {
    vir::with_vcx(|vcx| {
        vcx.mk_const_expr(vir::ConstData::Int(value as u128))
            .downcast_ty()
    })
}

fn as_const(expr: vir::ExprInt<'_>) -> Option<usize> {
    match expr.kind {
        vir::ExprKindGenData::Const(vir::ConstData::Int(value)) => Some(*value as usize),
        _ => None,
    }
}

/// `a + b`
pub fn add<'vir>(a: vir::ExprInt<'vir>, b: vir::ExprInt<'vir>) -> vir::ExprInt<'vir> {
    match (as_const(a), as_const(b)) {
        (Some(a), Some(b)) => int(a + b),
        (Some(0), _) => b,
        (_, Some(0)) => a,
        _ => vir::expr! { ([a]) + ([b]) }.downcast_ty(),
    }
}

/// `a - b`
fn sub<'vir>(a: vir::ExprInt<'vir>, b: vir::ExprInt<'vir>) -> vir::ExprInt<'vir> {
    match (as_const(a), as_const(b)) {
        (Some(a), Some(b)) if a >= b => int(a - b),
        (_, Some(0)) => a,
        _ => vir::expr! { ([a]) - ([b]) }.downcast_ty(),
    }
}

/// `x` clamped to `lo .. hi` (both inclusive): `clamp(x, lo, hi)` (see
/// `clamp_axiom`), a function so that Silicon does not branch on it.
fn clamp<'vir>(
    x: vir::ExprInt<'vir>,
    lo: vir::ExprInt<'vir>,
    hi: vir::ExprInt<'vir>,
) -> vir::ExprInt<'vir> {
    clamp_fn().call()(x, lo, hi)
}

pub fn clamp_fn<'vir>() -> vir::FunctionIdn<'vir, (vir::Int, vir::Int, vir::Int), vir::Int> {
    vir::FunctionIdn::new(
        vir::ViperIdent::new("clamp"),
        (vir::TYPE_INT, vir::TYPE_INT, vir::TYPE_INT),
        vir::TYPE_INT,
    )
}

/// The definition of `clamp`, i.e. `max(lo, min(hi, x))` for `lo <= hi`:
/// `clamp(x, lo, hi) == (x <= lo ? lo : (hi <= x ? hi : x))`.
pub fn clamp_axiom<'vir>(vcx: &'vir vir::VirCtxt<'vir>) -> vir::DomainAxiom<'vir> {
    let [x, lo, hi] = ["x", "lo", "hi"].map(|name| vcx.mk_local_decl(name, vir::TYPE_INT));
    let (xe, loe, hie) = (vcx.mk_local_ex(x), vcx.mk_local_ex(lo), vcx.mk_local_ex(hi));
    let app = clamp_fn().call()(xe, loe, hie);
    vcx.mk_domain_axiom(
        vir::ViperIdent::new("clamp_def"),
        vir::expr! { vcx;
            forall [x], [lo], [hi] :: {[app]}
                ([app]) == ((([xe]) <= ([loe])) ? ([loe]) : ((([hie]) <= ([xe])) ? ([hie]) : ([xe])))
        },
    )
}

/// The position `pos`, with `ty_of` the type value of a type parameter.
pub fn pos_expr<'vir>(
    pos: &Pos,
    ty_of: impl Fn(ty::ParamTy) -> vir::ExprTyVal<'vir>,
) -> vir::ExprInt<'vir> {
    pos.params.iter().fold(int(pos.konst), |sum, param| {
        add(sum, TyArityEnc::arity(ty_of(*param)))
    })
}
