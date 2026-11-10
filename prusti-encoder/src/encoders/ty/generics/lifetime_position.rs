use prusti_rustc_interface::middle::ty;
use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::CastType;

use crate::encoders::ty::{
    RustParamData, RustTyDecomposition, TySpecifics,
    generics::{
        TyExprEnc,
        arity::{ArityEnc, add},
    },
};

/// Encodes the lifetime positions of a type as an integer expression. Lifetime positions can either
/// be a lifetime parameter on the type itself, or a type parameter involving an `arity` expression
pub struct LifetimePositionEnc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifetimePosition<'tcx> {
    Lifetime(ty::Region<'tcx>),
    Param(RustTyDecomposition<'tcx>),
}

impl TaskEncoder for LifetimePositionEnc {
    task_encoder::encoder_cache!(LifetimePositionEnc);
    const ENCODER_NAME: &'static str = "lifetime position encoder";

    type TaskDescription<'vir> = RustTyDecomposition<'vir>;
    type OutputFullDependency<'vir> = Vec<(LifetimePosition<'vir>, vir::ExprInt<'vir>)>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;
        let positions = match task_key.ty.specifics {
            TySpecifics::ImmRef(_) => Vec::new(),
            TySpecifics::Param(RustParamData::Generic) => vec![(
                LifetimePosition::Param(*task_key),
                vir::with_vcx(|vcx| vcx.mk_uint::<0>()),
            )],
            _ => {
                let args = task_key.args.args();
                let mut positions = args
                    .iter()
                    .filter_map(|arg| arg.as_region())
                    .enumerate()
                    .map(|(j, region)| (LifetimePosition::Lifetime(region), int(j)))
                    .collect::<Vec<_>>();
                let mut offset = int(positions.len());
                let arity = deps.require_ref::<ArityEnc>(task_key.ty)?.arity;
                for arg_ty in args.iter().filter_map(|arg| arg.as_type()) {
                    let arg_ty = RustTyDecomposition::from_ty(arg_ty, task_key.args.context());
                    for (occupant, at) in deps.require_dep::<Self>(arg_ty)? {
                        positions.push((occupant, add(offset, at)));
                    }
                    // The arity of a type constructor is given by an axiom,
                    // which a type parameter does not have.
                    if !matches!(
                        arg_ty.ty.specifics,
                        TySpecifics::Param(RustParamData::Generic)
                    ) {
                        deps.require_ref::<ArityEnc>(arg_ty.ty)?;
                    }
                    let arg_ty = deps.require_dep::<TyExprEnc>(arg_ty)?;
                    offset = add(offset, arity.call()(arg_ty));
                }
                positions
            }
        };
        Ok(((), positions))
    }
}

fn int<'vir>(n: usize) -> vir::ExprInt<'vir> {
    vir::with_vcx(|vcx| {
        vcx.mk_const_expr(vir::ConstData::Int(n as u128))
            .downcast_ty()
    })
}

#[cfg(test)]
mod tests {
    use prusti_rustc_interface::span::DUMMY_SP;
    use task_encoder::TaskEncoder;

    use super::{LifetimePosition, LifetimePositionEnc};
    use crate::encoders::ty::test_util::{simplified, with_type_alias};

    /// The lifetime positions of the type alias `X` defined in `src`, as
    /// `occupant @ position`.
    fn positions(src: &str) -> Vec<String> {
        with_type_alias(src, |ty| {
            let (_, _, positions) = LifetimePositionEnc::encode(ty, true, DUMMY_SP)
                .unwrap()
                .unwrap();
            positions
                .into_iter()
                .map(|(occupant, position)| {
                    let position = simplified(position);
                    match occupant {
                        LifetimePosition::Lifetime(region) => format!("{region} @ {position:?}"),
                        LifetimePosition::Param(param) => {
                            format!("{} @ {position:?}", param.args.args()[0])
                        }
                    }
                })
                .collect()
        })
    }

    #[test]
    fn lifetime_after_type_param() {
        assert_eq!(
            positions("type X<'a, T> = (T, &'a mut i32);"),
            ["T @ 0", "'a @ arity(T$1)"],
        );
    }

    #[test]
    fn lifetimes_of_struct() {
        assert_eq!(
            positions(
                "struct Pair<'c, 'd>(&'c mut i32, &'d mut i32);
                 type X<'a, 'b> = Pair<'b, 'a>;"
            ),
            ["'b @ 0", "'a @ 1"],
        );
    }

    #[test]
    fn shared_reference() {
        assert_eq!(positions("type X<'a, 'b> = &'a &'b mut i32;"), [""; 0]);
    }

    #[test]
    fn repeated_lifetime() {
        assert_eq!(
            positions(
                "struct Pair<'c, 'd>(&'c mut i32, &'d mut i32);
                 type X<'a, 'b> = (&'a mut i32, Pair<'a, 'b>);"
            ),
            [
                "'a @ 0",
                "'a @ arity(s_Ref_mutable_type(s_Int_i32_type()))",
                "'b @ (arity(s_Ref_mutable_type(s_Int_i32_type()))) + (1)",
            ],
        );
    }

    #[test]
    fn same_reference_twice() {
        assert_eq!(
            positions("type X<'a, T> = (&'a mut T, &'a mut T);"),
            [
                "'a @ 0",
                "T @ 1",
                "'a @ arity(s_Ref_mutable_type(T$1))",
                "T @ (arity(s_Ref_mutable_type(T$1))) + (1)",
            ],
        );
    }

    #[test]
    fn nested_mutable_references() {
        assert_eq!(
            positions("type X<'a, 'b> = &'a mut &'b mut i32;"),
            ["'a @ 0", "'b @ 1"],
        );
    }

    #[test]
    fn static_lifetime() {
        assert_eq!(
            positions("type X<T> = &'static mut T;"),
            ["'static @ 0", "T @ 1"],
        );
    }

    #[test]
    fn array() {
        assert_eq!(
            positions("type X<'a, T, const N: usize> = [&'a mut T; N];"),
            ["'a @ 0", "T @ 1"],
        );
    }
}
