use rustc_hash::FxHashMap;
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, PredicateIdn};

use super::{
    RustTy, RustTyDecomposition,
    data::TySpecifics,
    generics::{
        GArgsTyEnc, GenericParamsEnc, TyExprEnc,
        arity::{ArityEnc, add},
        clamp::ClampEnc,
        lifetime_position::{LifetimePosition, LifetimePositionEnc},
    },
    pure::TyPureEnc,
};

pub struct IndirectPredEnc;

#[derive(Debug, Clone, Copy)]
pub struct IndirectPredRef<'vir> {
    pub predicate: PredicateIdn<
        'vir,
        (
            vir::Snap,
            vir::ManyTyVal,
            vir::ManyCSnap,
            vir::Int,
            vir::Int,
        ),
    >,
}

impl<'vir> task_encoder::OutputRefAny for IndirectPredRef<'vir> {}

/// The part of the query `[lo, hi)` of an indirect predicate that falls into
/// the positions of one own lifetime or type parameter of the type.
#[derive(Clone, Copy)]
struct Subquery<'vir> {
    /// Whether the subquery is non-empty.
    guard: vir::ExprBool<'vir>,
    /// The subquery is `[start, end)`, counted from the first position of the
    /// parameter.
    start: vir::ExprInt<'vir>,
    end: vir::ExprInt<'vir>,
}

impl IndirectPredEnc {
    /// The subqueries of the query `[lo, hi)` for own lifetimes and type parameters of `ty`.
    ///
    fn subqueries<'vir>(
        ty: RustTy<'vir>,
        lo: vir::ExprInt<'vir>,
        hi: vir::ExprInt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> Result<FxHashMap<LifetimePosition<'vir>, Subquery<'vir>>, EncodeFullError<'vir, Self>>
    {
        let arity = deps.require_ref::<ArityEnc>(ty)?.arity;
        let clamp = deps.require_dep::<ClampEnc>(())?;
        let own_positions =
            deps.require_dep::<LifetimePositionEnc>(RustTyDecomposition::identity(ty))?;
        vir::with_vcx(|vcx| {
            let sub = |a: vir::ExprInt<'vir>, b: vir::ExprInt<'vir>| -> vir::ExprInt<'vir> {
                vcx.mk_bin_op_expr(vir::BinOpKind::Sub, a, b).downcast_ty()
            };
            own_positions
                .into_iter()
                .map(|(param, x)| {
                    let subquery = match param {
                        // An own lifetime has the single position `x`, which the query either contains or not.
                        LifetimePosition::Lifetime(_) => Subquery {
                            guard: vir::expr! { vcx; ((lo) <= (x)) && ((x) < (hi)) },
                            start: vcx.mk_uint::<0>(),
                            end: vcx.mk_uint::<1>(),
                        },
                        // The subquery of an own type parameter at the positions `[x, x + w)` is `[clamp(lo - x, 0, w), clamp(hi - x, 0, w))`.
                        LifetimePosition::Param(ty) => {
                            let w = arity.call()(deps.require_dep::<TyExprEnc>(ty)?);
                            let zero = vcx.mk_uint::<0>();
                            let start = clamp.call()(sub(lo, x), zero, w);
                            let end = clamp.call()(sub(hi, x), zero, w);
                            Subquery {
                                guard: vir::expr! { vcx; (start) < (end) },
                                start,
                                end,
                            }
                        }
                    };
                    Ok((param, subquery))
                })
                .collect()
        })
    }
}

impl TaskEncoder for IndirectPredEnc {
    task_encoder::encoder_cache!(IndirectPredEnc);
    const ENCODER_NAME: &'static str = "indirect predicate encoder";

    type TaskDescription<'vir> = RustTy<'vir>;

    type OutputRef<'vir> = IndirectPredRef<'vir>;
    type OutputFullLocal<'vir> = vir::Predicate<'vir>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        let pure = deps.require_dep::<TyPureEnc>(*task_key)?;
        let params = deps.require_dep::<GenericParamsEnc>(task_key.params)?;
        let ty = task_key.zip(pure);
        vir::with_vcx(|vcx| {
            let snapshot = pure.data.snapshot;
            let predicate = PredicateIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "p_{}_ind", task_key.name())),
                (
                    snapshot,
                    params.ty_args(),
                    params.const_args(),
                    vir::TYPE_INT,
                    vir::TYPE_INT,
                ),
            );
            deps.emit_output_ref(*task_key, IndirectPredRef { predicate })?;

            let k_decl = vcx.mk_local_decl("k", snapshot);
            let lo_decl = vcx.mk_local_decl("lo", vir::TYPE_INT);
            let hi_decl = vcx.mk_local_decl("hi", vir::TYPE_INT);
            let k: vir::ExprSnap<'vir> = vcx.mk_local_ex(k_decl);
            let lo = vcx.mk_local_ex(lo_decl);
            let hi = vcx.mk_local_ex(hi_decl);

            let body = match &ty.specifics {
                TySpecifics::StructLike(data) => {
                    let subqueries = Self::subqueries(task_key, lo, hi, deps)?;
                    let mut conjuncts = Vec::new();
                    for (field, pure_field) in &data.fields {
                        let field_ty = field.decompose(task_key.params);
                        let field_positions = deps.require_dep::<LifetimePositionEnc>(field_ty)?;
                        // A field without lifetime positions holds no permissions.
                        if field_positions.is_empty() {
                            continue;
                        }
                        let field_pred =
                            deps.require_ref::<IndirectPredEnc>(field_ty.ty)?.predicate;
                        let field_args = deps.require_dep::<GArgsTyEnc>(field_ty.args)?;
                        let field_snap = pure_field.read.call()(k.downcast_ty());
                        for (param, y) in field_positions {
                            // Other lifetimes, such as `'static`, and alias types
                            // are not parameters of the struct.
                            let Some(subquery) = subqueries.get(&param) else {
                                continue;
                            };
                            let guard = subquery.guard;
                            let start = add(y, subquery.start);
                            let end = add(y, subquery.end);
                            conjuncts.push(vir::expr! { vcx;
                                (guard) ==> (acc([field_pred](
                                    field_snap,
                                    [..[field_args.get_ty()]],
                                    [..[field_args.get_const()]],
                                    start,
                                    end
                                )))
                            });
                        }
                    }
                    Some(vcx.mk_conj(&conjuncts))
                }
                // TODO: references, enums and the remaining types.
                _ => None,
            };

            Ok((
                vcx.mk_predicate(
                    predicate,
                    (
                        k_decl,
                        params.ty_decls(),
                        params.const_decls(),
                        lo_decl,
                        hi_decl,
                    ),
                    body,
                ),
                (),
            ))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for predicate in Self::all_outputs_local_no_errors(program) {
            program.add_predicate(predicate);
        }
    }
}

#[cfg(test)]
mod tests {
    use prusti_rustc_interface::span::DUMMY_SP;
    use task_encoder::TaskEncoder;
    use vir::CastType;

    use super::IndirectPredEnc;
    use crate::encoders::ty::test_util::{dedent, simplified, with_type_alias};

    /// The indirect predicate of the type constructor of the type alias `X`
    /// defined in `src`, with one conjunct of its body per line.
    fn indirect_pred(src: &str) -> String {
        fn conjuncts<'vir>(e: vir::ExprDyn<'vir>, out: &mut Vec<String>) {
            match e.kind {
                vir::ExprKindGenData::BinOp(b) if b.kind == vir::BinOpKind::And => {
                    conjuncts(b.lhs, out);
                    conjuncts(b.rhs, out);
                }
                _ => out.push(format!("  ({e:?})")),
            }
        }

        with_type_alias(src, |ty| {
            let (_, predicate, _) = IndirectPredEnc::encode(ty.ty, true, DUMMY_SP)
                .unwrap()
                .unwrap();
            let predicate = simplified(predicate);
            let debug = format!("{predicate:?}");
            let signature = debug.lines().next().unwrap().trim_end_matches('{');
            let Some(body) = predicate.expr else {
                return signature.to_string();
            };
            let mut body_conjuncts = Vec::new();
            conjuncts(body.as_dyn(), &mut body_conjuncts);
            format!("{signature} {{\n{}\n}}", body_conjuncts.join(" &&\n"))
        })
    }

    #[test]
    fn lifetimes() {
        assert_eq!(
            indirect_pred(
                "struct Pair<'a, 'b>(&'a mut i32, &'b mut i32);
                 type X<'a, 'b> = Pair<'a, 'b>;"
            ),
            dedent(
                r#"
                predicate p_Pair_ind(k: s_Pair, lo: Int, hi: Int) {
                  ((((lo) <= (0)) && ((0) < (hi))) ==> (p_Ref_mutable_ind(k.s_Pair_0, s_Int_i32_type(), 0, 1))) &&
                  ((((lo) <= (1)) && ((1) < (hi))) ==> (p_Ref_mutable_ind(k.s_Pair_1, s_Int_i32_type(), 0, 1)))
                }
                "#
            ),
        );
    }

    #[test]
    fn nested() {
        assert_eq!(
            indirect_pred(
                "struct Pair<'a, 'b>(&'a mut i32, &'b mut i32);
                 struct S<'c, 'd> {
                     p: Pair<'d, 'c>,
                     q: &'c mut i32,
                     r: &'d i32,
                     s: &'static mut i32,
                 }
                 type X<'c, 'd> = S<'c, 'd>;"
            ),
            dedent(
                r#"
                predicate p_S_ind(k: s_S, lo: Int, hi: Int) {
                  ((((lo) <= (1)) && ((1) < (hi))) ==> (p_Pair_ind(k.s_S_0, 0, 1))) &&
                  ((((lo) <= (0)) && ((0) < (hi))) ==> (p_Pair_ind(k.s_S_0, 1, 2))) &&
                  ((((lo) <= (0)) && ((0) < (hi))) ==> (p_Ref_mutable_ind(k.s_S_1, s_Int_i32_type(), 0, 1)))
                }
                "#
            ),
        );
    }

    #[test]
    fn after_type_param() {
        assert_eq!(
            indirect_pred(
                "struct S<'a, T> { f: (T, &'a mut i32) }
                 type X<'a, T> = S<'a, T>;"
            ),
            dedent(
                r#"
                predicate p_S_ind(k: s_S, T$1: Type, lo: Int, hi: Int) {
                  (((clamp((lo) - (1), 0, arity(T$1))) < (clamp((hi) - (1), 0, arity(T$1)))) ==> (p_2_Tuple_ind(k.s_S_0, T$1, s_Ref_mutable_type(s_Int_i32_type()), clamp((lo) - (1), 0, arity(T$1)), clamp((hi) - (1), 0, arity(T$1))))) &&
                  ((((lo) <= (0)) && ((0) < (hi))) ==> (p_2_Tuple_ind(k.s_S_0, T$1, s_Ref_mutable_type(s_Int_i32_type()), arity(T$1), (arity(T$1)) + (1))))
                }
                "#
            ),
        );
    }

    #[test]
    fn tuple() {
        assert_eq!(
            indirect_pred("type X<T, U> = (T, U);"),
            dedent(
                r#"
                predicate p_2_Tuple_ind(k: s_2_Tuple, T$0: Type, U$1: Type, lo: Int, hi: Int) {
                  (((clamp(lo, 0, arity(T$0))) < (clamp(hi, 0, arity(T$0)))) ==> (p_Param_ind(k.s_2_Tuple_0, T$0, clamp(lo, 0, arity(T$0)), clamp(hi, 0, arity(T$0))))) &&
                  (((clamp((lo) - (arity(T$0)), 0, arity(U$1))) < (clamp((hi) - (arity(T$0)), 0, arity(U$1)))) ==> (p_Param_ind(k.s_2_Tuple_1, U$1, clamp((lo) - (arity(T$0)), 0, arity(U$1)), clamp((hi) - (arity(T$0)), 0, arity(U$1)))))
                }
                "#
            ),
        );
    }

    /// Immutable references have no body yet.
    #[test]
    fn immutable_reference() {
        assert_eq!(
            indirect_pred("type X<'a, T> = &'a T;"),
            "predicate p_Ref_immutable_ind(k: s_Ref_immutable, U$1: Type, lo: Int, hi: Int)",
        );
    }

    /// Mutable references have no body yet.
    #[test]
    fn mutable_reference() {
        assert_eq!(
            indirect_pred("type X<'a, T> = &'a mut T;"),
            "predicate p_Ref_mutable_ind(k: s_Ref_mutable, U$1: Type, lo: Int, hi: Int)",
        );
    }
}
