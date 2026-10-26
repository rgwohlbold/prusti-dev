use task_encoder::{EncodeFullResult, TaskEncoder};

use super::{
    RustParamData, RustTy, RustTyDecomposition, TySpecifics,
    generics::{GArgsTy, GParams, GenericParamsEnc},
    lifted::TyConstructorEnc,
    occurrences::{End, occurrences, param_position},
};

#[derive(Debug, Clone, Copy)]
enum HasIndTyRef<'vir> {
    GenericParam,
    Constructor(vir::FunctionIdn<'vir, (vir::ManyTyVal, vir::ManyCSnap), vir::TyVal>),
}

#[derive(Debug, Clone, Copy)]
pub struct TyHasIndRef<'vir> {
    ty: HasIndTyRef<'vir>,
}

impl<'vir> task_encoder::OutputRefAny for TyHasIndRef<'vir> {}

impl<'vir> TyHasIndRef<'vir> {
    /// `has_ind(T)` for the type with these arguments.
    pub fn apply<Curr, Next>(&self, args: GArgsTy<'vir>) -> vir::ExprGenBool<'vir, Curr, Next> {
        let ty: vir::ExprGenTyVal<'vir, Curr, Next> = match self.ty {
            HasIndTyRef::GenericParam => args.get_ty::<Curr, Next>()[0],
            HasIndTyRef::Constructor(constructor) => {
                constructor.call()(args.get_ty::<Curr, Next>(), args.get_const::<Curr, Next>())
            }
        };
        TyHasIndEnc::has_ind_fn().call()(ty)
    }
}

/// Encodes `has_ind(T)`: can a value of type `T` reach anything through
/// mutable references, i.e. can its indirect pieces hold anything? One axiom
/// per type constructor: true if it has a tracked lifetime, otherwise iff one
/// of its tracked type parameters has (see `occurrences`). Pieces at generic
/// positions are guarded by it, so that types without mutable references
/// have no pieces at all.
pub struct TyHasIndEnc;

impl TaskEncoder for TyHasIndEnc {
    task_encoder::encoder_cache!(TyHasIndEnc);
    const ENCODER_NAME: &'static str = "has_ind encoder";

    type TaskDescription<'vir> = RustTy<'vir>;
    type OutputRef<'vir> = TyHasIndRef<'vir>;
    type OutputFullLocal<'vir> = Option<vir::DomainAxiom<'vir>>;
    type EncodingError = ();

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut task_encoder::TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        vir::with_vcx(|vcx| {
            let ty = if matches!(task_key.specifics, TySpecifics::Param(RustParamData::Generic)) {
                HasIndTyRef::GenericParam
            } else {
                HasIndTyRef::Constructor(deps.require_ref::<TyConstructorEnc>(*task_key)?.ty_constructor)
            };
            let output_ref = TyHasIndRef { ty };
            deps.emit_output_ref(*task_key, output_ref)?;
            if matches!(ty, HasIndTyRef::GenericParam) {
                return Ok((None, ()));
            }

            let params = deps.require_dep::<GenericParamsEnc>(task_key.params)?;
            let args = GArgsTy::new(
                vcx.alloc_slice(params.ty_exprs()),
                vcx.alloc_slice(params.const_exprs()),
            );
            let lhs = output_ref.apply(args);
            let rhs = has_ind_rhs(*task_key, task_key.params, params.ty_exprs());
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
                vir::vir_format_identifier!(vcx, "has_ind_{}", task_key.name()),
                axiom_expr,
            );
            Ok((Some(axiom), ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        let axioms = Self::all_outputs_local_no_errors(program)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        vir::with_vcx(|vcx| {
            let has_ind = vcx.mk_domain_function(Self::has_ind_fn(), false, None);
            let domain = vcx.mk_domain(
                vir::ViperIdent::new("HasInd"),
                &[],
                vcx.alloc_slice(&axioms),
                vcx.alloc_slice(&[has_ind]),
                None,
            );
            program.add_domain(domain);
        })
    }
}

impl TyHasIndEnc {
    pub fn has_ind_fn<'vir>() -> vir::FunctionIdn<'vir, vir::TyVal, vir::Bool> {
        vir::FunctionIdn::new(vir::ViperIdent::new("has_ind"), vir::TYPE_TYVAL, vir::TYPE_BOOL)
    }
}

/// The definition of `has_ind` for the constructor `ty`: a tracked lifetime,
/// or `has_ind` of a tracked type parameter.
fn has_ind_rhs<'vir>(
    ty: RustTy<'vir>,
    params: GParams<'vir>,
    ty_exprs: &[vir::ExprTyVal<'vir>],
) -> vir::ExprBool<'vir> {
    let occurrences = occurrences(RustTyDecomposition::identity(ty)).unwrap_or_default();
    vir::with_vcx(|vcx| {
        if occurrences.iter().any(|o| matches!(o.end, End::Region(_))) {
            return vcx.mk_bool::<true>();
        }
        let params = occurrences
            .iter()
            .filter_map(|o| match o.end {
                End::Param(param) => Some(param_position(params, param.index, |arg| {
                    arg.as_type().is_some()
                })),
                End::Region(_) => None,
            })
            .map(|j| TyHasIndEnc::has_ind_fn().call()(ty_exprs[j]))
            .collect::<Vec<_>>();
        vcx.mk_disj(&params)
    })
}
