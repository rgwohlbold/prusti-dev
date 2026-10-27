use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, MethodIdn};

use crate::encoders::{
    Pure,
    ty::{
        RustTy,
        indirect_pred::TyIndirectEnc,
        lifted::ty_constructor::TyConstructorEnc,
        pure::TyPureEnc,
    },
};

use super::{GenericParamsEnc, casters::CastersEnc};

/// The trusted casts of an indirect predicate between a type `X` and `Param`,
/// for the same pairs as [`CastersEnc`]:
/// ```text
/// method make_generic_ind_X(k: s_X, T.., lo: Int, hi: Int)
///   requires p_X_ind(k, T.., lo, hi)
///   ensures  p_Param_ind(make_generic_s_X(k, T..), X_type(T..), lo, hi)
///   ensures  ext_Param(make_generic_s_X(k, T..), X_type(T..), lo, hi)
///              == make_generic_s_X(old(ext_X(k, T.., lo, hi)), T..)
/// method make_concrete_ind_X(p: s_Param, T.., lo: Int, hi: Int)
///   requires p_Param_ind(p, X_type(T..), lo, hi)
///   ensures  p_X_ind(make_concrete_s_X(p), T.., lo, hi)
///   ensures  make_generic_s_X(make_concrete_s_X(p), T..) == p
///   ensures  old(ext_Param(p, X_type(T..), lo, hi))
///              == make_generic_s_X(ext_X(make_concrete_s_X(p), T.., lo, hi), T..)
/// ```
/// The equations are the only link between `ext_Param` and `ext_X`.
pub(crate) struct IndCastersEnc;

// TODO: remove once the impure encoder casts indirect predicates.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct IndCasters<'vir> {
    pub(crate) make_generic: IndCaster<'vir>,
    pub(crate) make_concrete: IndCaster<'vir>,
    /// The pure cast of the keys (`make_generic_s_X`).
    pub(crate) make_generic_s:
        vir::FunctionIdn<'vir, (vir::CSnap, vir::ManyTyVal, vir::ManyCSnap), vir::PSnap>,
    /// The pure cast back (`make_concrete_s_X`).
    pub(crate) make_concrete_s: vir::AdtDestructor<'vir, vir::PSnap, vir::CSnap>,
}

pub(crate) type IndCaster<'vir> =
    MethodIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::Int, vir::Int)>;

impl<'vir> task_encoder::OutputRefAny for IndCasters<'vir> {}

impl TaskEncoder for IndCastersEnc {
    task_encoder::encoder_cache!(IndCastersEnc);
    const ENCODER_NAME: &'static str = "indirect casters encoder";

    type TaskDescription<'vir> = (RustTy<'vir>, RustTy<'vir>);
    type OutputRef<'vir> = IndCasters<'vir>;
    type OutputFullLocal<'vir> = Vec<vir::Method<'vir>>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        let (param, concrete) = *task_key;
        assert!(param.specifics.is_param() && !concrete.specifics.is_param());
        vir::with_vcx(|vcx| {
            let base_name = concrete.name();
            let generics = deps.require_dep::<GenericParamsEnc>(concrete.params)?;
            let concrete_snap = deps.require_ref::<TyPureEnc>(concrete)?.snapshot;
            let args = (
                concrete_snap,
                generics.ty_args(),
                generics.const_args(),
                vir::TYPE_INT,
                vir::TYPE_INT,
            );
            let make_generic_ident = MethodIdn::new(
                vir::vir_format_identifier!(vcx, "make_generic_ind_{base_name}"),
                args,
            );
            let make_concrete_ident = MethodIdn::new(
                vir::vir_format_identifier!(vcx, "make_concrete_ind_{base_name}"),
                (vir::TYPE_PSNAP.upcast_ty(), args.1, args.2, args.3, args.4),
            );
            let casters = deps.require_ref::<CastersEnc<Pure>>(*task_key)?;
            deps.emit_output_ref(
                *task_key,
                IndCasters {
                    make_generic: make_generic_ident,
                    make_concrete: make_concrete_ident,
                    make_generic_s: casters.make_generic,
                    make_concrete_s: casters.make_concrete,
                },
            )?;

            let ty_constructor = deps.require_ref::<TyConstructorEnc>(concrete)?.ty_constructor;
            let concrete_ind = deps.require_ref::<TyIndirectEnc>(concrete)?;
            let param_ind = deps.require_ref::<TyIndirectEnc>(param)?;

            let tys = generics.ty_exprs();
            let consts = generics.const_exprs();
            let lifted_ty = ty_constructor(tys, consts);
            let lo_decl = vcx.mk_local_decl("lo", vir::TYPE_INT);
            let hi_decl = vcx.mk_local_decl("hi", vir::TYPE_INT);
            let (lo, hi) = (vcx.mk_local_ex(lo_decl), vcx.mk_local_ex(hi_decl));

            let make_generic =
                |s: vir::ExprSnap<'vir>| (casters.make_generic)(s.downcast_ty(), tys, consts);
            let make_concrete =
                |p: vir::ExprSnap<'vir>| casters.make_concrete.call()(p.downcast_ty()).upcast_ty();
            let concrete_pred = |k: vir::ExprSnap<'vir>| {
                vir::expr! { vcx; acc([concrete_ind.pred](k, [..[tys]], [..[consts]], lo, hi)) }
            };
            let no_consts: [vir::ExprCSnap<'vir>; 0] = [];
            let param_pred = |p: vir::ExprSnap<'vir>| {
                vir::expr! { vcx; acc([param_ind.pred](p, [..[[lifted_ty]]], [..[no_consts]], lo, hi)) }
            };
            let concrete_ext = |k| (concrete_ind.ext)(k, tys, consts, lo, hi);
            let param_ext = |p| (param_ind.ext)(p, &[lifted_ty], &[], lo, hi);

            // make_generic_ind_X
            let k_decl = vcx.mk_local_decl("k", concrete_snap);
            let k = vcx.mk_local_ex(k_decl);
            let generic_ext = param_ext(make_generic(k).upcast_ty()).downcast_ty::<vir::PSnap>();
            let generic_ext_old = make_generic(vir::expr! { vcx; old([concrete_ext(k)]) });
            let make_generic_method = vcx.mk_method(
                make_generic_ident,
                (k_decl, generics.ty_decls(), generics.const_decls(), lo_decl, hi_decl),
                &[],
                vcx.alloc_slice(&[concrete_pred(k)]),
                vcx.alloc_slice(&[
                    param_pred(make_generic(k).upcast_ty()),
                    vir::expr! { vcx; ([generic_ext]) == ([generic_ext_old]) },
                ]),
                None,
            );

            // make_concrete_ind_X
            let p_decl = vcx.mk_local_decl("p", vir::TYPE_PSNAP.upcast_ty());
            let p = vcx.mk_local_ex(p_decl);
            let param_ext_old = vir::expr! { vcx; old([param_ext(p)]) }.downcast_ty::<vir::PSnap>();
            let concrete_ext_value = make_generic(concrete_ext(make_concrete(p)));
            let round_trip = make_generic(make_concrete(p));
            let make_concrete_method = vcx.mk_method(
                make_concrete_ident,
                (p_decl, generics.ty_decls(), generics.const_decls(), lo_decl, hi_decl),
                &[],
                vcx.alloc_slice(&[param_pred(p)]),
                vcx.alloc_slice(&[
                    concrete_pred(make_concrete(p)),
                    vir::expr! { vcx; ([round_trip]) == ([p.downcast_ty::<vir::PSnap>()]) },
                    vir::expr! { vcx; ([param_ext_old]) == ([concrete_ext_value]) },
                ]),
                None,
            );
            Ok((vec![make_generic_method, make_concrete_method], ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for output in Self::all_outputs_local_no_errors(program) {
            for method in output {
                program.add_method(method);
            }
        }
    }
}
