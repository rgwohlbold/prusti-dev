use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, MethodIdn, Reify};

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
/// method make_generic_ind_X(k: s_X, T.., q: LifetimeQuery)
///   requires p_X_ind(k, T.., q)
///   ensures  p_Param_ind(make_generic_s_X(k, T..), X_type(T..), q)
///   ensures  pre(q) ==> forall K: s_X :: { ext_Param(make_generic_s_X(K, T..), X_type(T..), make_generic_s_X(k, T..), q) }
///              agree(K, k) ==> ext_Param(..) == make_generic_s_X(old(ext_X(K, T.., k, q)), T..)
/// method make_concrete_ind_X(p: s_Param, T.., q: LifetimeQuery)
///   requires p_Param_ind(p, X_type(T..), q)
///   ensures  p_X_ind(make_concrete_s_X(p), T.., q)
///   ensures  make_generic_s_X(make_concrete_s_X(p), T..) == p
///   ensures  pre(q) ==> forall K: s_Param :: { ext_X(make_concrete_s_X(K), T.., make_concrete_s_X(p), q) }
///              agree(make_concrete_s_X(K), make_concrete_s_X(p)) ==>
///                old(ext_Param(K, X_type(T..), p, q)) == make_generic_s_X(ext_X(..), T..)
/// ```
/// where `pre(q)` and `agree` are the preconditions of `ext_X`. The equations
/// are the only link between `ext_Param` and `ext_X`; the concrete one
/// triggers on an `ext_X(make_concrete_s_X(K), ..)` term.
pub(crate) struct IndCastersEnc;

// TODO: remove once the impure encoder casts indirect predicates.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct IndCasters<'vir> {
    pub(crate) make_generic: IndCaster<'vir>,
    pub(crate) make_concrete: IndCaster<'vir>,
}

pub(crate) type IndCaster<'vir> =
    MethodIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::LifetimeQuery)>;

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
                vir::TYPE_LIFETIME_QUERY,
            );
            let make_generic_ident = MethodIdn::new(
                vir::vir_format_identifier!(vcx, "make_generic_ind_{base_name}"),
                args,
            );
            let make_concrete_ident = MethodIdn::new(
                vir::vir_format_identifier!(vcx, "make_concrete_ind_{base_name}"),
                (vir::TYPE_PSNAP.upcast_ty(), args.1, args.2, args.3),
            );
            deps.emit_output_ref(
                *task_key,
                IndCasters {
                    make_generic: make_generic_ident,
                    make_concrete: make_concrete_ident,
                },
            )?;

            let casters = deps.require_ref::<CastersEnc<Pure>>(*task_key)?;
            let ty_constructor = deps.require_ref::<TyConstructorEnc>(concrete)?.ty_constructor;
            let concrete_ind = deps.require_ref::<TyIndirectEnc>(concrete)?;
            let concrete_dep = deps.require_dep::<TyIndirectEnc>(concrete)?;
            let param_ind = deps.require_ref::<TyIndirectEnc>(param)?;

            let tys = generics.ty_exprs();
            let consts = generics.const_exprs();
            let lifted_ty = ty_constructor(tys, consts);
            let q_decl = vcx.mk_local_decl("q", vir::TYPE_LIFETIME_QUERY);
            let q = vcx.mk_local_ex(q_decl);

            let make_generic =
                |s: vir::ExprSnap<'vir>| (casters.make_generic)(s.downcast_ty(), tys, consts);
            let make_concrete =
                |p: vir::ExprSnap<'vir>| casters.make_concrete.call()(p.downcast_ty()).upcast_ty();
            let concrete_pred = |k| {
                vcx.mk_predicate_app_expr((concrete_ind.pred)(k, tys, consts, q)(None))
            };
            let param_pred = |p| {
                vcx.mk_predicate_app_expr((param_ind.pred)(p, &[lifted_ty], &[], q)(None))
            };
            let concrete_ext = |k, kp| (concrete_ind.ext)(k, tys, consts, kp, q);
            let param_ext = |p, pp| (param_ind.ext)(p, &[lifted_ty], &[], pp, q);
            // `pre(q) ==> forall K :: { trigger } agree ==> body`
            let equation = |k_decl: vir::LocalDeclSnap<'vir>,
                             trigger: vir::ExprSnap<'vir>,
                             agree: Option<vir::ExprBool<'vir>>,
                             body: vir::ExprBool<'vir>| {
                let body = match agree {
                    Some(agree) => vcx
                        .mk_bin_op_expr(vir::BinOpKind::Implies, agree, body)
                        .downcast_ty(),
                    None => body,
                };
                let forall = vcx.mk_forall_expr(
                    vcx.alloc_slice(&[k_decl]),
                    vcx.alloc_slice(&[vcx.mk_trigger(&[trigger])]),
                    body,
                );
                match concrete_dep.query_pre {
                    Some(pre) => vcx
                        .mk_bin_op_expr(vir::BinOpKind::Implies, pre.reify(vcx, q), forall)
                        .downcast_ty(),
                    None => forall,
                }
            };

            // make_generic_ind_X
            let k_decl = vcx.mk_local_decl("k", concrete_snap);
            let k = vcx.mk_local_ex(k_decl);
            let big_k_decl = vcx.mk_local_decl("K", concrete_snap);
            let big_k = vcx.mk_local_ex(big_k_decl);
            let generic_ext = param_ext(make_generic(big_k).upcast_ty(), make_generic(k).upcast_ty());
            let generic_equation = equation(
                big_k_decl,
                generic_ext,
                concrete_dep.agree.map(|agree| agree.reify(vcx, (big_k, k))),
                vcx.mk_eq_expr(
                    generic_ext.downcast_ty::<vir::PSnap>(),
                    make_generic(vcx.mk_old_expr(concrete_ext(big_k, k))),
                ),
            );
            let make_generic_method = vcx.mk_method(
                make_generic_ident,
                (k_decl, generics.ty_decls(), generics.const_decls(), q_decl),
                &[],
                vcx.alloc_slice(&[concrete_pred(k)]),
                vcx.alloc_slice(&[param_pred(make_generic(k).upcast_ty()), generic_equation]),
                None,
            );

            // make_concrete_ind_X
            let p_decl = vcx.mk_local_decl("p", vir::TYPE_PSNAP.upcast_ty());
            let p = vcx.mk_local_ex(p_decl);
            let big_p_decl = vcx.mk_local_decl("K", vir::TYPE_PSNAP.upcast_ty());
            let big_p = vcx.mk_local_ex(big_p_decl);
            let concrete_ext_app = concrete_ext(make_concrete(big_p), make_concrete(p));
            let concrete_equation = equation(
                big_p_decl,
                concrete_ext_app,
                concrete_dep
                    .agree
                    .map(|agree| agree.reify(vcx, (make_concrete(big_p), make_concrete(p)))),
                vcx.mk_eq_expr(
                    vcx.mk_old_expr(param_ext(big_p, p)).downcast_ty::<vir::PSnap>(),
                    make_generic(concrete_ext_app),
                ),
            );
            let make_concrete_method = vcx.mk_method(
                make_concrete_ident,
                (p_decl, generics.ty_decls(), generics.const_decls(), q_decl),
                &[],
                vcx.alloc_slice(&[param_pred(p)]),
                vcx.alloc_slice(&[
                    concrete_pred(make_concrete(p)),
                    vcx.mk_eq_expr(make_generic(make_concrete(p)), p.downcast_ty()),
                    concrete_equation,
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
