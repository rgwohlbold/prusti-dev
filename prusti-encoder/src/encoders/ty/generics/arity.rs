use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::CastType;

use crate::encoders::ty::{
    RustTy, TySpecifics, generics::GenericParamsEnc, lifted::TyConstructorEnc,
};

pub struct ArityEnc;

#[derive(Debug, Clone, Copy)]
pub struct ArityEncOutputRef<'vir> {
    pub arity: vir::FunctionIdn<'vir, vir::TyVal, vir::Int>,
}

impl<'vir> task_encoder::OutputRefAny for ArityEncOutputRef<'vir> {}

/// The sum `a + b`.
pub(crate) fn add<'vir>(a: vir::ExprInt<'vir>, b: vir::ExprInt<'vir>) -> vir::ExprInt<'vir> {
    vir::with_vcx(|vcx| vcx.mk_bin_op_expr(vir::BinOpKind::Add, a, b).downcast_ty())
}

fn arity_fn_ident<'vir>() -> vir::FunctionIdn<'vir, vir::TyVal, vir::Int> {
    vir::FunctionIdn::new(vir::ViperIdent::new("arity"), vir::TYPE_TYVAL, vir::TYPE_INT)
}

impl TaskEncoder for ArityEnc {
    task_encoder::encoder_cache!(ArityEnc);

    const ENCODER_NAME: &'static str = "arity encoder";

    type TaskDescription<'vir> = RustTy<'vir>;

    type OutputRef<'vir> = ArityEncOutputRef<'vir>;

    type OutputFullLocal<'vir> = vir::DomainAxiom<'vir>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(
            *task_key,
            ArityEncOutputRef {
                arity: arity_fn_ident(),
            },
        )?;
        let num_lifetimes = task_key.params.rust_params().regions().count();
        let ty_constructor = deps.require_ref::<TyConstructorEnc>(*task_key)?;
        let params = deps.require_dep::<GenericParamsEnc>(task_key.params)?;
        vir::with_vcx(|vcx| {
            let arity_fn_ident = arity_fn_ident();
            let num_lifetimes = vcx.mk_const_expr(vir::ConstData::Int(num_lifetimes as u128)).downcast_ty();
            let instantiated_type = ty_constructor.ty_constructor.call()(params.ty_exprs(), params.const_exprs());
            // shared references have no lifetime positions, they work with snapshots instead
            let rhs = if matches!(task_key.specifics, TySpecifics::ImmRef(_)) {
                vcx.mk_uint::<0>()
            } else {
                // L + arity(T_0) + ... + arity(T_{M-1})
                params.ty_exprs().iter().fold(num_lifetimes, |acc, ty_arg| {
                    add(acc, arity_fn_ident.call()(*ty_arg))
                })
            };
            let expr = vir::expr! { vcx;
                 forall ..[params.ty_decls()], ..[params.const_decls()] :: {[arity_fn_ident](instantiated_type)}
                    ([arity_fn_ident](instantiated_type)) == (rhs)
            };
            let axiom = vcx.mk_domain_axiom(
                vir::vir_format_identifier!(vcx, "arity_{}", task_key.name())
            , expr);
            Ok((axiom, ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        let axioms = Self::all_outputs_local_no_errors(program);
        let domain = vir::with_vcx(|vcx| {
            let arity_fn = vcx.mk_domain_function(arity_fn_ident(), false, None);
            vcx.mk_domain(
                vir::ViperIdent::new("Arity"),
                &[],
                vcx.alloc_slice(&axioms),
                vcx.alloc_slice(&[arity_fn]),
                None,
            )
        });
        program.add_domain(domain);
    }
}

#[cfg(test)]
mod tests {
    use prusti_rustc_interface::span::DUMMY_SP;
    use task_encoder::TaskEncoder;

    use super::ArityEnc;
    use crate::encoders::ty::test_util::{dedent, simplified, with_type_alias};

    /// The expression of the arity axiom of the type constructor of the type alias `X` defined in `src`.
    /// A quantified axiom shows its body on its own line.
    fn arity_axiom(src: &str) -> String {
        with_type_alias(src, |ty| {
            let (_, axiom, _) = ArityEnc::encode(ty.ty, true, DUMMY_SP).unwrap().unwrap();
            let expr = simplified(axiom.expr);
            let vir::ExprKindGenData::Forall(forall) = expr.kind else {
                return format!("{expr:?}");
            };
            let body = format!("{:?}", forall.body);
            let quantifier = format!("{expr:?}");
            let quantifier = quantifier.strip_suffix(&body).unwrap().trim_end();
            format!("{quantifier}\n  {body}")
        })
    }

    #[test]
    fn tuple() {
        assert_eq!(
            arity_axiom("type X<T, U> = (T, U);"),
            dedent(
                r#"
                forall T$0: Type, U$1: Type :: {arity(s_2_Tuple_type(T$0, U$1))}
                  (arity(s_2_Tuple_type(T$0, U$1))) == ((arity(T$0)) + (arity(U$1)))
                "#
            ),
        );
    }

    #[test]
    fn struct_with_lifetimes() {
        assert_eq!(
            arity_axiom(
                "struct S<'a, 'b, T>(&'a mut T, &'b mut T);
                 type X<'a, 'b, T> = S<'a, 'b, T>;"
            ),
            dedent(
                r#"
                forall T$2: Type :: {arity(s_S_type(T$2))}
                  (arity(s_S_type(T$2))) == ((2) + (arity(T$2)))
                "#
            ),
        );
    }

    #[test]
    fn shared_reference() {
        assert_eq!(
            arity_axiom("type X<'a, T> = &'a T;"),
            dedent(
                r#"
                forall U$1: Type :: {arity(s_Ref_immutable_type(U$1))}
                  (arity(s_Ref_immutable_type(U$1))) == (0)
                "#
            ),
        );
    }

    #[test]
    fn mutable_reference() {
        assert_eq!(
            arity_axiom("type X<'a, T> = &'a mut T;"),
            dedent(
                r#"
                forall U$1: Type :: {arity(s_Ref_mutable_type(U$1))}
                  (arity(s_Ref_mutable_type(U$1))) == ((1) + (arity(U$1)))
                "#
            ),
        );
    }

    #[test]
    fn primitive() {
        assert_eq!(
            arity_axiom("type X = i32;"),
            "(arity(s_Int_i32_type())) == (0)",
        );
    }

    #[test]
    fn struct_without_type_params() {
        assert_eq!(
            arity_axiom(
                "struct S<'a>(&'a mut i32);
                 type X<'a> = S<'a>;"
            ),
            "(arity(s_S_type())) == (1)",
        );
    }

    #[test]
    fn array() {
        assert_eq!(
            arity_axiom("type X<T, const N: usize> = [T; N];"),
            dedent(
                r#"
                forall U$1: Type, M$0: s_UInt_usize :: {arity(s_Array_type(U$1, M$0))}
                  (arity(s_Array_type(U$1, M$0))) == (arity(U$1))
                "#
            ),
        );
    }

    #[test]
    fn raw_pointer() {
        assert_eq!(
            arity_axiom("type X<T> = *mut T;"),
            dedent(
                r#"
                forall T$0: Type :: {arity(s_RawPtr_mutable_type(T$0))}
                  (arity(s_RawPtr_mutable_type(T$0))) == (arity(T$0))
                "#
            ),
        );
    }

    #[test]
    fn dyn_trait() {
        assert_eq!(
            arity_axiom("type X = dyn Copy;"),
            "(arity(s_Dyn_type())) == (0)",
        );
    }

    #[test]
    fn boxed() {
        assert_eq!(
            arity_axiom("type X<T> = Box<T>;"),
            dedent(
                r#"
                forall T$0: Type, A$1: Type :: {arity(s_Box_type(T$0, A$1))}
                  (arity(s_Box_type(T$0, A$1))) == ((arity(T$0)) + (arity(A$1)))
                "#
            ),
        );
    }
}
