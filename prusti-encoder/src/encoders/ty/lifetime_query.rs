use task_encoder::{EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{AdtDestructor, CastType, FunctionIdn};

/// Encodes the `LifetimeQuery` adt, which selects the one occurrence of a
/// lifetime in a type that an indirect predicate covers:
/// ```text
/// adt LifetimeQuery { q_one(i: Int)  q_arg(k: Int, q_sub: LifetimeQuery)  q_all() }
/// ```
/// `q_one(i)` selects the type's own lifetime parameter `i`, `q_arg(k, q_sub)`
/// the occurrence `q_sub` inside its type parameter `k`, `q_all()` every
/// occurrence (opaque for a type parameter: `RegionsIn(T)`).
pub struct LifetimeQueryEnc;

#[derive(Debug, Clone, Copy)]
pub struct LifetimeQueryFns<'vir> {
    q_one: FunctionIdn<'vir, vir::Int, vir::LifetimeQuery>,
    q_arg: FunctionIdn<'vir, (vir::Int, vir::LifetimeQuery), vir::LifetimeQuery>,
    q_all: FunctionIdn<'vir, (), vir::LifetimeQuery>,
    k: AdtDestructor<'vir, vir::LifetimeQuery, vir::Int>,
    q_sub: AdtDestructor<'vir, vir::LifetimeQuery, vir::LifetimeQuery>,
}

impl<'vir> LifetimeQueryFns<'vir> {
    /// `q_one(index)`
    pub fn one(&self, index: usize) -> vir::ExprLifetimeQuery<'vir> {
        vir::with_vcx(|vcx| (self.q_one)(mk_int(vcx, index)))
    }

    /// `q_arg(index, sub)`
    pub fn arg(
        &self,
        index: usize,
        sub: vir::ExprLifetimeQuery<'vir>,
    ) -> vir::ExprLifetimeQuery<'vir> {
        vir::with_vcx(|vcx| (self.q_arg)(mk_int(vcx, index), sub))
    }

    /// `q_all()`
    pub fn all(&self) -> vir::ExprLifetimeQuery<'vir> {
        (self.q_all)()
    }

    /// `query == q_all()`
    pub fn is_all(&self, query: vir::ExprLifetimeQuery<'vir>) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| vcx.mk_eq_expr(query, self.all()))
    }

    /// `query == q_one(index)`
    pub fn is_one(&self, query: vir::ExprLifetimeQuery<'vir>, index: usize) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| vcx.mk_eq_expr(query, self.one(index)))
    }

    /// `query.isq_arg && query.k == index`
    pub fn is_arg(&self, query: vir::ExprLifetimeQuery<'vir>, index: usize) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| {
            vcx.mk_conj(&[
                vcx.mk_adt_discriminator_expr(query, "q_arg"),
                vcx.mk_eq_expr(self.k.call()(query), mk_int(vcx, index)),
            ])
        })
    }

    /// `query.q_sub`
    pub fn sub(&self, query: vir::ExprLifetimeQuery<'vir>) -> vir::ExprLifetimeQuery<'vir> {
        self.q_sub.call()(query)
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
            let i_decl = vcx.mk_local_decl("i", vir::TYPE_INT);
            let k_decl = vcx.mk_local_decl("k", vir::TYPE_INT);
            let sub_decl = vcx.mk_local_decl("q_sub", query_ty);
            let adt = vcx.mk_adt(
                vir::ViperIdent::new("LifetimeQuery"),
                &[],
                vcx.alloc_slice(&[
                    vcx.mk_adt_constructor("q_one", vcx.alloc_slice(&[i_decl.as_dyn()])),
                    vcx.mk_adt_constructor(
                        "q_arg",
                        vcx.alloc_slice(&[k_decl.as_dyn(), sub_decl.as_dyn()]),
                    ),
                    vcx.mk_adt_constructor::<(), !, vir::Dyn>("q_all", &[]),
                ]),
            );
            Ok((
                adt,
                LifetimeQueryFns {
                    q_one: FunctionIdn::new(vir::ViperIdent::new("q_one"), vir::TYPE_INT, query_ty),
                    q_arg: FunctionIdn::new(
                        vir::ViperIdent::new("q_arg"),
                        (vir::TYPE_INT, query_ty),
                        query_ty,
                    ),
                    q_all: FunctionIdn::new(vir::ViperIdent::new("q_all"), (), query_ty),
                    k: vcx.mk_adt_destructor("k", query_ty, vir::TYPE_INT),
                    q_sub: vcx.mk_adt_destructor("q_sub", query_ty, query_ty),
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
