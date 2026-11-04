use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, ViperIdent, vir_format};

use crate::encoders::ty::{RustTy, generics::GenericParamsEnc, lifted::TyConstructorEnc};

pub struct ArityEnc;

fn arity_fn_ident<'vir>() -> vir::FunctionIdn<'vir, vir::TyVal, vir::Int> {
    vir::FunctionIdn::new(vir::ViperIdent::new("arity"), vir::TYPE_TYVAL, vir::TYPE_INT)
}

impl TaskEncoder for ArityEnc {
    task_encoder::encoder_cache!(ArityEnc);

    const ENCODER_NAME: &'static str = "arity encoder";

    type TaskDescription<'vir> = RustTy<'vir>;

    type OutputFullLocal<'vir> = vir::DomainAxiom<'vir>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;
        let num_lifetimes = task_key.params.rust_params().regions().count();
        let ty_constructor = deps.require_ref::<TyConstructorEnc>(*task_key)?;
        let params = deps.require_dep::<GenericParamsEnc>(task_key.params)?;
        vir::with_vcx(|vcx| {
            let arity_fn_ident = arity_fn_ident();
            let num_lifetimes = vcx.mk_const_expr(vir::ConstData::Int(num_lifetimes as u128)).downcast_ty();
            let instantiated_type = ty_constructor.ty_constructor.call()(params.ty_exprs(), params.const_exprs());
            // L + arity(T_0) + ... + arity(T_{M-1})
            let rhs = params.ty_exprs().iter().fold(num_lifetimes, |acc, ty_arg| {
                vcx.mk_bin_op_expr(vir::BinOpKind::Add, acc, arity_fn_ident.call()(*ty_arg))
                    .downcast_ty()
            });
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