use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, FunctionIdn};

/// Encodes the `LifetimeQuery` adt, which selects the lifetimes of a type that
/// an indirect predicate covers, and the domain `LifetimeQueryOps` reading it:
/// ```text
/// adt LifetimeQuery { q_none()  q_all()  q_sel(q_own: Seq[Bool], q_args: Seq[LifetimeQuery]) }
/// ```
/// `q_own[i]` selects the type's own lifetime parameter `i`, `q_args[j]` what
/// is selected inside its type parameter `j`.
pub struct LifetimeQueryEnc;

#[derive(Debug, Clone, Copy)]
pub struct LifetimeQueryFns<'vir> {
    q_none: FunctionIdn<'vir, (), vir::LifetimeQuery>,
    selects_lifetime: FunctionIdn<'vir, (vir::LifetimeQuery, vir::Int), vir::Bool>,
    selection_in_arg: FunctionIdn<'vir, (vir::LifetimeQuery, vir::Int), vir::LifetimeQuery>,
    selects_in_arg: FunctionIdn<'vir, (vir::LifetimeQuery, vir::Int), vir::Bool>,
}

impl<'vir> LifetimeQueryFns<'vir> {
    pub fn q_none(&self) -> vir::ExprLifetimeQuery<'vir> {
        (self.q_none)()
    }

    /// Does `query` select the type's own lifetime parameter `index`?
    pub fn selects_lifetime(
        &self,
        query: vir::ExprLifetimeQuery<'vir>,
        index: usize,
    ) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| (self.selects_lifetime)(query, mk_int(vcx, index)))
    }

    /// The part of `query` that selects something inside type argument `index`.
    pub fn selection_in_arg(
        &self,
        query: vir::ExprLifetimeQuery<'vir>,
        index: usize,
    ) -> vir::ExprLifetimeQuery<'vir> {
        vir::with_vcx(|vcx| (self.selection_in_arg)(query, mk_int(vcx, index)))
    }

    /// Does `query` select anything inside type argument `index`?
    pub fn selects_in_arg(
        &self,
        query: vir::ExprLifetimeQuery<'vir>,
        index: usize,
    ) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| (self.selects_in_arg)(query, mk_int(vcx, index)))
    }
}

fn mk_int<'vir>(vcx: &'vir vir::VirCtxt<'_>, value: usize) -> vir::ExprInt<'vir> {
    vcx.mk_const_expr(vir::ConstData::Int(value as u128))
        .downcast_ty()
}

fn mk_bin<'vir, T: vir::CompType>(
    vcx: &'vir vir::VirCtxt<'_>,
    kind: vir::BinOpKind,
    lhs: vir::Expr<'vir, T>,
    rhs: vir::Expr<'vir, T>,
) -> vir::ExprBool<'vir> {
    vcx.mk_bin_op_expr(kind, lhs, rhs).downcast_ty()
}

fn mk_not<'vir>(vcx: &'vir vir::VirCtxt<'_>, expr: vir::ExprBool<'vir>) -> vir::ExprBool<'vir> {
    vcx.mk_unary_op_expr(vir::UnOpKind::Not, expr.upcast_ty())
        .downcast_ty()
}

/// `forall vars :: { trigger } trigger == body`
fn mk_defining_axiom<'vir, T: vir::CompType>(
    vcx: &'vir vir::VirCtxt<'_>,
    name: &'vir str,
    vars: &[vir::LocalDeclDyn<'vir>],
    trigger: vir::Expr<'vir, T>,
    body: vir::Expr<'vir, T>,
) -> vir::DomainAxiom<'vir> {
    let forall = vcx.mk_forall_expr(
        vcx.alloc_slice(vars),
        vcx.alloc_slice(&[vcx.mk_trigger(&[trigger])]),
        vcx.mk_eq_expr(trigger, body),
    );
    vcx.mk_domain_axiom(vir::ViperIdent::new(name), forall)
}

#[derive(Clone)]
pub struct LifetimeQueryEncOutput<'vir> {
    adt: vir::Adt<'vir>,
    domain: vir::Domain<'vir>,
}

impl TaskEncoder for LifetimeQueryEnc {
    task_encoder::encoder_cache!(LifetimeQueryEnc);
    const ENCODER_NAME: &'static str = "lifetime query encoder";

    type TaskDescription<'vir> = ();
    type OutputFullLocal<'vir> = LifetimeQueryEncOutput<'vir>;
    type OutputFullDependency<'vir> = LifetimeQueryFns<'vir>;

    fn task_to_key<'vir>(_task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {}

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;
        vir::with_vcx(|vcx| {
            let query_ty = vir::TYPE_LIFETIME_QUERY;
            let own_ty = vcx.mk_ty_seq(vir::TYPE_BOOL);
            let args_ty = vcx.mk_ty_seq(query_ty);

            // the adt
            let own_decl = vcx.mk_local_decl("q_own", own_ty);
            let args_decl = vcx.mk_local_decl("q_args", args_ty);
            let adt = vcx.mk_adt(
                vir::ViperIdent::new("LifetimeQuery"),
                &[],
                vcx.alloc_slice(&[
                    vcx.mk_adt_constructor::<(), !, vir::Dyn>("q_none", &[]),
                    vcx.mk_adt_constructor::<(), !, vir::Dyn>("q_all", &[]),
                    vcx.mk_adt_constructor(
                        "q_sel",
                        vcx.alloc_slice(&[own_decl.as_dyn(), args_decl.as_dyn()]),
                    ),
                ]),
            );
            let q_none = FunctionIdn::new(vir::ViperIdent::new("q_none"), (), query_ty);
            let q_own = vcx.mk_adt_destructor("q_own", query_ty, own_ty);
            let q_args = vcx.mk_adt_destructor("q_args", query_ty, args_ty);

            // the operations
            let ops_args = (query_ty, vir::TYPE_INT);
            let selects_lifetime =
                FunctionIdn::new(vir::ViperIdent::new("selects_lifetime"), ops_args, vir::TYPE_BOOL);
            let selection_in_arg =
                FunctionIdn::new(vir::ViperIdent::new("selection_in_arg"), ops_args, query_ty);
            let selects_in_arg =
                FunctionIdn::new(vir::ViperIdent::new("selects_in_arg"), ops_args, vir::TYPE_BOOL);

            let query_decl = vcx.mk_local_decl("query", query_ty);
            let index_decl = vcx.mk_local_decl("index", vir::TYPE_INT);
            let query = vcx.mk_local_ex(query_decl);
            let index = vcx.mk_local_ex(index_decl);
            let vars = [query_decl.as_dyn(), index_decl.as_dyn()];
            let is_all = vcx.mk_adt_discriminator_expr(query, "q_all");
            let is_sel = vcx.mk_adt_discriminator_expr(query, "q_sel");
            // `0 <= index && index < |seq|`
            let in_range = |seq: vir::ExprSeq<'vir>| {
                vcx.mk_conj(&[
                    mk_bin(vcx, vir::BinOpKind::CmpLe, vcx.mk_int::<0>(), index),
                    mk_bin(
                        vcx,
                        vir::BinOpKind::CmpLt,
                        index,
                        vcx.mk_collection_len_expr(seq),
                    ),
                ])
            };

            // selects_lifetime(query, index) ==
            //   query.isq_all || (query.isq_sel && 0 <= index && index < |query.q_own| && query.q_own[index])
            let own = q_own.call()(query);
            let selects_lifetime_ax = mk_defining_axiom(
                vcx,
                "ax_selects_lifetime",
                &vars,
                selects_lifetime(query, index),
                vcx.mk_disj(&[
                    is_all,
                    vcx.mk_conj(&[
                        is_sel,
                        in_range(own),
                        vcx.mk_seq_index_expr(own, index).downcast_ty(),
                    ]),
                ]),
            );

            // selection_in_arg(query, index) ==
            //   query.isq_all ? q_all()
            //   : (query.isq_sel && 0 <= index && index < |query.q_args|) ? query.q_args[index] : q_none()
            let q_all: FunctionIdn<'vir, (), vir::LifetimeQuery> = FunctionIdn::new(vir::ViperIdent::new("q_all"), (), query_ty);
            let args = q_args.call()(query);
            let selection_in_arg_ax = mk_defining_axiom(
                vcx,
                "ax_selection_in_arg",
                &vars,
                selection_in_arg(query, index),
                vcx.mk_ternary_expr(
                    is_all,
                    q_all(),
                    vcx.mk_ternary_expr(
                        vcx.mk_conj(&[is_sel, in_range(args)]),
                        vcx.mk_seq_index_expr(args, index).downcast_ty(),
                        q_none(),
                    ),
                ),
            );

            // selects_in_arg(query, index) == (selection_in_arg(query, index) != q_none())
            let selects_in_arg_ax = mk_defining_axiom(
                vcx,
                "ax_selects_in_arg",
                &vars,
                selects_in_arg(query, index),
                mk_bin(
                    vcx,
                    vir::BinOpKind::CmpNe,
                    selection_in_arg(query, index),
                    q_none(),
                ),
            );

            let domain = vcx.mk_domain(
                vir::ViperIdent::new("LifetimeQueryOps"),
                &[],
                vcx.alloc_slice(&[selects_lifetime_ax, selection_in_arg_ax, selects_in_arg_ax]),
                vcx.alloc_slice(&[
                    vcx.mk_domain_function(selects_lifetime, false, None),
                    vcx.mk_domain_function(selection_in_arg, false, None),
                    vcx.mk_domain_function(selects_in_arg, false, None),
                ]),
                None,
            );
            Ok((
                LifetimeQueryEncOutput { adt, domain },
                LifetimeQueryFns {
                    q_none,
                    selects_lifetime,
                    selection_in_arg,
                    selects_in_arg,
                },
            ))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for output in Self::all_outputs_local_no_errors(program) {
            program.add_adt(output.adt);
            program.add_domain(output.domain);
        }
    }
}

/// Encodes the constructor of canonical queries for a type with `n` lifetime
/// and `m` type parameters (the task):
/// ```text
/// function query_n_m(b_0: Bool, .., s_0: LifetimeQuery, ..): LifetimeQuery
/// axiom { forall .. :: { query_n_m(..) } query_n_m(..) ==
///   ((!b_0 && .. && s_0 == q_none() && ..) ? q_none() : q_sel(Seq(b_0, ..), Seq(s_0, ..))) }
/// ```
/// An empty selection is always `q_none()`, so queries can be compared by `==`.
pub struct LifetimeQueryShapeEnc;

pub type LifetimeQueryShape<'vir> =
    FunctionIdn<'vir, (vir::Many<vir::Bool>, vir::Many<vir::LifetimeQuery>), vir::LifetimeQuery>;

impl TaskEncoder for LifetimeQueryShapeEnc {
    task_encoder::encoder_cache!(LifetimeQueryShapeEnc);
    const ENCODER_NAME: &'static str = "lifetime query shape encoder";

    type TaskDescription<'vir> = (usize, usize);
    type OutputFullLocal<'vir> = vir::Domain<'vir>;
    type OutputFullDependency<'vir> = LifetimeQueryShape<'vir>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;
        let (n, m) = *task_key;
        assert!(n + m > 0, "a type without parameters has no query constructor");
        let query = deps.require_dep::<LifetimeQueryEnc>(())?;
        vir::with_vcx(|vcx| {
            let query_ty = vir::TYPE_LIFETIME_QUERY;
            let lifetime_decls = (0..n)
                .map(|i| vcx.mk_local_decl(vir::vir_format!(vcx, "b_{i}"), vir::TYPE_BOOL))
                .collect::<Vec<_>>();
            let arg_decls = (0..m)
                .map(|j| vcx.mk_local_decl(vir::vir_format!(vcx, "s_{j}"), query_ty))
                .collect::<Vec<_>>();
            let shape: LifetimeQueryShape<'vir> = FunctionIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "query_{n}_{m}")),
                (
                    vcx.alloc_slice(&vec![vir::TYPE_BOOL; n]),
                    vcx.alloc_slice(&vec![query_ty; m]),
                ),
                query_ty,
            );

            let lifetimes = lifetime_decls
                .iter()
                .map(|d| vcx.mk_local_ex(*d))
                .collect::<Vec<_>>();
            let args = arg_decls
                .iter()
                .map(|d| vcx.mk_local_ex(*d))
                .collect::<Vec<_>>();
            let empty = vcx.mk_conj(
                &lifetimes
                    .iter()
                    .map(|b| mk_not(vcx, *b))
                    .chain(args.iter().map(|s| vcx.mk_eq_expr(*s, query.q_none())))
                    .collect::<Vec<_>>(),
            );
            let own_ty = vcx.mk_ty_seq(vir::TYPE_BOOL);
            let args_ty = vcx.mk_ty_seq(query_ty);
            let q_sel: FunctionIdn<'vir, (vir::Seq, vir::Seq), vir::LifetimeQuery> = FunctionIdn::new(vir::ViperIdent::new("q_sel"), (own_ty, args_ty), query_ty);
            let selection = q_sel(
                vcx.mk_seq_literal_expr(vcx.alloc_slice(&lifetimes), vir::TYPE_BOOL),
                vcx.mk_seq_literal_expr(vcx.alloc_slice(&args), query_ty),
            );

            let vars = lifetime_decls
                .iter()
                .map(|d| d.as_dyn())
                .chain(arg_decls.iter().map(|d| d.as_dyn()))
                .collect::<Vec<_>>();
            let axiom = mk_defining_axiom(
                vcx,
                vir::vir_format!(vcx, "ax_query_{n}_{m}"),
                &vars,
                shape(lifetimes.as_slice(), args.as_slice()),
                vcx.mk_ternary_expr(empty, query.q_none(), selection),
            );
            let domain = vcx.mk_domain(
                vir::ViperIdent::new(vir::vir_format!(vcx, "LifetimeQueryShape_{n}_{m}")),
                &[],
                vcx.alloc_slice(&[axiom]),
                vcx.alloc_slice(&[vcx.mk_domain_function(shape, false, None)]),
                None,
            );
            Ok((domain, shape))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for domain in Self::all_outputs_local_no_errors(program) {
            program.add_domain(domain);
        }
    }
}
