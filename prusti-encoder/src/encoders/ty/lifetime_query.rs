use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{AdtDestructor, CastType, FunctionIdn};

/// Encodes the `LifetimeQuery` adt, which selects the one occurrence of a
/// lifetime in a type that an indirect predicate covers:
/// ```text
/// adt LifetimeQuery { q_here(q_idx: Int)  q_in(q_arg: Int, q_rest: LifetimeQuery)  q_all() }
/// ```
/// `q_here(i)` selects the type's own lifetime parameter `i`, `q_in(j, rest)`
/// the occurrence `rest` inside its type parameter `j`, `q_all()` every
/// occurrence (opaque for a type parameter: `RegionsIn(T)`).
pub struct LifetimeQueryEnc;

#[derive(Debug, Clone, Copy)]
pub struct LifetimeQueryFns<'vir> {
    q_here: FunctionIdn<'vir, vir::Int, vir::LifetimeQuery>,
    q_in: FunctionIdn<'vir, (vir::Int, vir::LifetimeQuery), vir::LifetimeQuery>,
    q_all: FunctionIdn<'vir, (), vir::LifetimeQuery>,
    q_arg: AdtDestructor<'vir, vir::LifetimeQuery, vir::Int>,
    q_rest: AdtDestructor<'vir, vir::LifetimeQuery, vir::LifetimeQuery>,
}

impl<'vir> LifetimeQueryFns<'vir> {
    /// `q_here(index)`
    pub fn here(&self, index: usize) -> vir::ExprLifetimeQuery<'vir> {
        vir::with_vcx(|vcx| (self.q_here)(mk_int(vcx, index)))
    }

    /// `q_in(index, rest)`
    pub fn in_arg(
        &self,
        index: usize,
        rest: vir::ExprLifetimeQuery<'vir>,
    ) -> vir::ExprLifetimeQuery<'vir> {
        vir::with_vcx(|vcx| (self.q_in)(mk_int(vcx, index), rest))
    }

    /// `q_all()`
    pub fn all(&self) -> vir::ExprLifetimeQuery<'vir> {
        (self.q_all)()
    }

    /// `query == q_all()`
    pub fn is_all(&self, query: vir::ExprLifetimeQuery<'vir>) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| vcx.mk_eq_expr(query, self.all()))
    }

    /// `query == q_here(index)`
    pub fn is_here(&self, query: vir::ExprLifetimeQuery<'vir>, index: usize) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| vcx.mk_eq_expr(query, self.here(index)))
    }

    /// `query.isq_in && query.q_arg == index`
    pub fn is_in(&self, query: vir::ExprLifetimeQuery<'vir>, index: usize) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| {
            vcx.mk_conj(&[
                vcx.mk_adt_discriminator_expr(query, "q_in"),
                vcx.mk_eq_expr(self.q_arg.call()(query), mk_int(vcx, index)),
            ])
        })
    }

    /// `query.q_rest`
    pub fn rest(&self, query: vir::ExprLifetimeQuery<'vir>) -> vir::ExprLifetimeQuery<'vir> {
        self.q_rest.call()(query)
    }
}

fn mk_int<'vir>(vcx: &'vir vir::VirCtxt<'_>, value: usize) -> vir::ExprInt<'vir> {
    vcx.mk_const_expr(vir::ConstData::Int(value as u128))
        .downcast_ty()
}

impl TaskEncoder for LifetimeQueryEnc {
    task_encoder::encoder_cache!(LifetimeQueryEnc);
    const ENCODER_NAME: &'static str = "lifetime query encoder";

    type TaskDescription<'vir> = ();
    type OutputFullLocal<'vir> = vir::Adt<'vir>;
    type OutputFullDependency<'vir> = LifetimeQueryFns<'vir>;

    fn task_to_key<'vir>(_task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {}

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;
        vir::with_vcx(|vcx| {
            let query_ty = vir::TYPE_LIFETIME_QUERY;
            let idx_decl = vcx.mk_local_decl("q_idx", vir::TYPE_INT);
            let arg_decl = vcx.mk_local_decl("q_arg", vir::TYPE_INT);
            let rest_decl = vcx.mk_local_decl("q_rest", query_ty);
            let adt = vcx.mk_adt(
                vir::ViperIdent::new("LifetimeQuery"),
                &[],
                vcx.alloc_slice(&[
                    vcx.mk_adt_constructor("q_here", vcx.alloc_slice(&[idx_decl.as_dyn()])),
                    vcx.mk_adt_constructor(
                        "q_in",
                        vcx.alloc_slice(&[arg_decl.as_dyn(), rest_decl.as_dyn()]),
                    ),
                    vcx.mk_adt_constructor::<(), !, vir::Dyn>("q_all", &[]),
                ]),
            );
            Ok((
                adt,
                LifetimeQueryFns {
                    q_here: FunctionIdn::new(vir::ViperIdent::new("q_here"), vir::TYPE_INT, query_ty),
                    q_in: FunctionIdn::new(
                        vir::ViperIdent::new("q_in"),
                        (vir::TYPE_INT, query_ty),
                        query_ty,
                    ),
                    q_all: FunctionIdn::new(vir::ViperIdent::new("q_all"), (), query_ty),
                    q_arg: vcx.mk_adt_destructor("q_arg", query_ty, vir::TYPE_INT),
                    q_rest: vcx.mk_adt_destructor("q_rest", query_ty, query_ty),
                },
            ))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for adt in Self::all_outputs_local_no_errors(program) {
            program.add_adt(adt);
        }
    }
}
