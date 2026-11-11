use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, FunctionIdn, ViperIdent};

/// Encodes helper functions on sets of integers: `interval(lo, hi)` is the
/// set `[lo, hi)`, and `shift(s, x)` contains the `i` with `i + x` in `s`.
pub struct SetHelperEnc;

#[derive(Debug, Clone, Copy)]
pub struct SetHelperRef<'vir> {
    pub interval: FunctionIdn<'vir, (vir::Int, vir::Int), vir::Set>,
    pub shift: FunctionIdn<'vir, (vir::Set, vir::Int), vir::Set>,
}

impl<'vir> task_encoder::OutputRefAny for SetHelperRef<'vir> {}

impl TaskEncoder for SetHelperEnc {
    task_encoder::encoder_cache!(SetHelperEnc);
    const ENCODER_NAME: &'static str = "set helper encoder";

    type TaskDescription<'vir> = ();
    type OutputRef<'vir> = SetHelperRef<'vir>;
    type OutputFullLocal<'vir> = vir::Domain<'vir>;

    fn task_to_key<'vir>(_task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {}

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        vir::with_vcx(|vcx| {
            let set = vcx.mk_ty_set(vir::TYPE_INT);
            let interval = FunctionIdn::new(
                ViperIdent::new("interval"),
                (vir::TYPE_INT, vir::TYPE_INT),
                set,
            );
            let shift = FunctionIdn::new(ViperIdent::new("shift"), (set, vir::TYPE_INT), set);
            deps.emit_output_ref(*task_key, SetHelperRef { interval, shift })?;

            let i_decl = vcx.mk_local_decl("i", vir::TYPE_INT);
            let lo_decl = vcx.mk_local_decl("lo", vir::TYPE_INT);
            let hi_decl = vcx.mk_local_decl("hi", vir::TYPE_INT);
            let s_decl = vcx.mk_local_decl("s", set);
            let x_decl = vcx.mk_local_decl("x", vir::TYPE_INT);
            let i = vcx.mk_local_ex(i_decl);
            let lo = vcx.mk_local_ex(lo_decl);
            let hi = vcx.mk_local_ex(hi_decl);
            let s = vcx.mk_local_ex(s_decl);
            let x = vcx.mk_local_ex(x_decl);
            let membership = |set: vir::ExprSet<'vir>, body: vir::ExprBool<'vir>| {
                vcx.mk_eq_expr(vcx.mk_set_in_expr(i, set), body)
            };

            let in_interval = vcx.mk_set_in_expr(i, interval.call()(lo, hi));
            let interval_def = vcx.mk_forall_expr(
                vcx.alloc_slice(&[lo_decl, hi_decl, i_decl]),
                vcx.alloc_slice(&[vcx.mk_trigger(&[in_interval])]),
                membership(
                    interval.call()(lo, hi),
                    vir::expr! { vcx; ((lo) <= (i)) && ((i) < (hi)) },
                ),
            );
            let in_shift = vcx.mk_set_in_expr(i, shift.call()(s, x));
            let shifted: vir::ExprInt<'vir> =
                vcx.mk_bin_op_expr(vir::BinOpKind::Add, i, x).downcast_ty();
            let shift_def = vcx.mk_forall_expr(
                vcx.alloc_slice(&[s_decl.as_dyn(), x_decl.as_dyn(), i_decl.as_dyn()]),
                vcx.alloc_slice(&[vcx.mk_trigger(&[in_shift])]),
                membership(shift.call()(s, x), vcx.mk_set_in_expr(shifted, s)),
            );

            let domain = vcx.mk_domain(
                ViperIdent::new("SetHelper"),
                &[],
                vcx.alloc_slice(&[
                    vcx.mk_domain_axiom(ViperIdent::new("interval_def"), interval_def),
                    vcx.mk_domain_axiom(ViperIdent::new("shift_def"), shift_def),
                ]),
                vcx.alloc_slice(&[
                    vcx.mk_domain_function(interval, false, None),
                    vcx.mk_domain_function(shift, false, None),
                ]),
                None,
            );
            Ok((domain, ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for domain in Self::all_outputs_local_no_errors(program) {
            program.add_domain(domain);
        }
    }
}
