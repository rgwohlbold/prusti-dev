use task_encoder::{EncodeFullResult, TaskEncoder};

use super::{
    RustParamData, RustTy, RustTyDecomposition, TySpecifics,
    generics::{GParams, GenericParamsEnc},
    lifted::TyConstructorEnc,
    occurrences::{End, occurrences, param_position},
    range::{add, clamp_axiom, clamp_fn, int},
};

/// Encodes `arity(T)`: the number of lifetime positions of a value of type
/// `T` that indirect pieces cover (see `occurrences`). One axiom per type
/// constructor: its own tracked lifetimes plus the arities of its tracked
/// type parameters. A type with arity 0 has no pieces. The domain also holds
/// `clamp` (see `range`).
pub struct TyArityEnc;

impl TaskEncoder for TyArityEnc {
    task_encoder::encoder_cache!(TyArityEnc);
    const ENCODER_NAME: &'static str = "arity encoder";

    type TaskDescription<'vir> = RustTy<'vir>;
    type OutputFullLocal<'vir> = Option<vir::DomainAxiom<'vir>>;
    type EncodingError = ();

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut task_encoder::TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;
        if matches!(
            task_key.specifics,
            TySpecifics::Param(RustParamData::Generic)
        ) {
            return Ok((None, ()));
        }
        let constructor = deps
            .require_ref::<TyConstructorEnc>(*task_key)?
            .ty_constructor;
        vir::with_vcx(|vcx| {
            let params = deps.require_dep::<GenericParamsEnc>(task_key.params)?;
            let lhs = Self::arity(constructor(params.ty_exprs(), params.const_exprs()));
            let rhs = arity_rhs(task_key, task_key.params, params.ty_exprs());
            let body = vir::expr! { vcx; ([lhs]) == ([rhs]) };
            let axiom_expr = if params.ty_decls().is_empty() && params.const_decls().is_empty() {
                body
            } else {
                vir::expr! {
                    forall ..[params.ty_decls()], ..[params.const_decls()] :: {[lhs]}
                        [body]
                }
            };
            let axiom = vcx.mk_domain_axiom(
                vir::vir_format_identifier!(vcx, "arity_{}", task_key.name()),
                axiom_expr,
            );
            Ok((Some(axiom), ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        let mut axioms = Self::all_outputs_local_no_errors(program)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        vir::with_vcx(|vcx| {
            let t_decl = vcx.mk_local_decl("T", vir::TYPE_TYVAL);
            let arity_t = Self::arity(vcx.mk_local_ex(t_decl));
            let zero = int(0);
            axioms.push(vcx.mk_domain_axiom(
                vir::ViperIdent::new("arity_nonnegative"),
                vir::expr! { vcx; forall [t_decl] :: {[arity_t]} ([zero]) <= ([arity_t]) },
            ));
            axioms.push(clamp_axiom(vcx));
            let arity = vcx.mk_domain_function(Self::arity_fn(), false, None);
            let clamp = vcx.mk_domain_function(clamp_fn(), false, None);
            let domain = vcx.mk_domain(
                vir::ViperIdent::new("Arity"),
                &[],
                vcx.alloc_slice(&axioms),
                vcx.alloc_slice(&[arity, clamp]),
                None,
            );
            program.add_domain(domain);
        })
    }
}

impl TyArityEnc {
    pub fn arity_fn<'vir>() -> vir::FunctionIdn<'vir, vir::TyVal, vir::Int> {
        vir::FunctionIdn::new(
            vir::ViperIdent::new("arity"),
            vir::TYPE_TYVAL,
            vir::TYPE_INT,
        )
    }

    /// `arity(ty)`
    pub fn arity<'vir, Curr, Next>(
        ty: vir::ExprGenTyVal<'vir, Curr, Next>,
    ) -> vir::ExprGenInt<'vir, Curr, Next> {
        Self::arity_fn().call()(ty)
    }
}

/// The definition of `arity` for the constructor `ty`: its own tracked
/// lifetimes plus `arity` of its tracked type parameters.
fn arity_rhs<'vir>(
    ty: RustTy<'vir>,
    params: GParams<'vir>,
    ty_exprs: &[vir::ExprTyVal<'vir>],
) -> vir::ExprInt<'vir> {
    let occurrences = occurrences(RustTyDecomposition::identity(ty)).unwrap_or_default();
    {
        let own = occurrences
            .iter()
            .filter(|o| matches!(o.end, End::Region(_)))
            .count();
        let arities = occurrences.iter().filter_map(|o| match o.end {
            End::Param(param) => Some(TyArityEnc::arity(
                ty_exprs[param_position(params, param.index, |arg| arg.as_type().is_some())],
            )),
            End::Region(_) => None,
        });
        arities.fold(int(own), add)
    }
}
