use prusti_rustc_interface::index::Idx;
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, MethodIdn};

use crate::encoders::ty::{
    RustTyDecomposition,
    indirect_pred::{IndirectPredEnc, LifetimePositionBlock},
    pure::TyPureEnc,
};

use super::{
    GArgsTyEnc, GenericParamsEnc,
    lifetime_position::{LifetimePosition, LifetimePositionEnc},
};

/// Trusted methods that merge the indirect predicates of the lifetime
/// positions `[first, end)` of an instantiated type into one predicate over
/// their intervals, and split it again. A generic call site merges the
/// positions a type parameter of the callee covers before casting the result
/// to the parameter's predicate, and splits them after the call. Both state
/// that fetching through the merged predicate gives what fetching through
/// each position's predicate on its key gives, plugged together.
pub struct MergeSplitEnc;

#[derive(Debug, Clone, Copy)]
pub struct MergeSplit<'vir> {
    pub merge: MergeSplitIdn<'vir>,
    pub split: MergeSplitIdn<'vir>,
}

pub type MergeSplitIdn<'vir> = MethodIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap)>;

impl<'vir> task_encoder::OutputRefAny for MergeSplit<'vir> {}

/// The instantiated type has a const parameter, which canonical keys do not
/// support yet.
#[derive(Debug, Clone)]
pub struct ConstParamInBlock;

impl TaskEncoder for MergeSplitEnc {
    task_encoder::encoder_cache!(MergeSplitEnc);
    const ENCODER_NAME: &'static str = "merge/split encoder";

    type TaskDescription<'vir> = (
        RustTyDecomposition<'vir>,
        LifetimePosition,
        LifetimePosition,
    );
    type OutputRef<'vir> = MergeSplit<'vir>;
    type OutputFullLocal<'vir> = [vir::Method<'vir>; 2];
    type EncodingError = ConstParamInBlock;

    // Instantiations that differ only in their context or lifetimes share the
    // methods.
    fn task_to_key<'vir>((ty, first, end): &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        (ty.canonical().unwrap_or(*ty), *first, *end)
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        let (ty, first, end) = *task_key;
        if ty.canonical().is_none() {
            return Err(EncodeFullError::EncodingError(ConstParamInBlock, None));
        }
        let graph = deps.require_dep::<LifetimePositionEnc>(ty)?;
        assert!(
            first < end && end.index() <= graph.positions().count(),
            "block out of range"
        );
        let snapshot = deps.require_ref::<TyPureEnc>(ty.ty)?.snapshot;
        let ind = deps.require_ref::<IndirectPredEnc>(ty.ty)?;
        let params = deps.require_dep::<GenericParamsEnc>(ty.args.context())?;
        let args = deps.require_dep::<GArgsTyEnc>(ty.args)?;
        vir::with_vcx(|vcx| {
            let name = ty.instantiated_name();
            let (f, e) = (first.index(), end.index());
            let arg_tys = (snapshot, params.ty_args(), params.const_args());
            let merge = MethodIdn::new(
                vir::vir_format_identifier!(vcx, "merge_ind_{name}_{f}_{e}"),
                arg_tys,
            );
            let split = MethodIdn::new(
                vir::vir_format_identifier!(vcx, "split_ind_{name}_{f}_{e}"),
                arg_tys,
            );
            deps.emit_output_ref(*task_key, MergeSplit { merge, split })?;

            let k_decl = vcx.mk_local_decl("k", snapshot);
            // The contracts name each written-out type argument once, in a `let`.
            let mut bindings = Vec::new();
            let ty_args: &[vir::ExprTyVal<'vir>] = args.get_ty();
            let tys = ty_args
                .iter()
                .enumerate()
                .map(|(i, &ty_arg)| {
                    if matches!(ty_arg.kind, vir::ExprKindGenData::Local(_)) {
                        return ty_arg;
                    }
                    let name = if ty_args.len() == 1 {
                        "ty".to_string()
                    } else {
                        format!("ty{i}")
                    };
                    let decl = vcx.mk_local_decl(vcx.alloc_str(&name), vir::TYPE_TYVAL);
                    bindings.push((decl, ty_arg));
                    vcx.mk_local_ex(decl)
                })
                .collect::<Vec<_>>();
            let with_bindings = |e: vir::ExprBool<'vir>| {
                bindings
                    .iter()
                    .rev()
                    .fold(e, |e, &(decl, ty_arg)| vcx.mk_let_expr(decl, ty_arg, e))
            };
            let block = LifetimePositionBlock {
                graph: &graph,
                first,
                end,
                ind,
                tys: vcx.alloc_slice(&tys),
                consts: args.get_const(),
                k: vcx.mk_local_ex(k_decl),
            };
            let positions = block.positions();
            // Outermost first, so that each key's fetches are framed by the
            // predicates before it.
            let preds = positions
                .iter()
                .map(|&p| {
                    let (lo, hi) = graph.interval(p);
                    block.predicate(block.key(p), lo, hi)
                })
                .collect::<Vec<_>>();
            let (lo, _) = graph.interval(first);
            let (_, hi) = graph.interval(LifetimePosition::new(end.index() - 1));
            let merged = block.predicate(block.k, lo, hi);
            let fetched = ind.fetch.call()(block.k, block.tys, block.consts, lo, hi);
            let plugged = block.fold(&positions);
            let decls = (k_decl, params.ty_decls(), params.const_decls());
            let merge_post =
                conjunction(&[merged, vcx.mk_eq_expr(fetched, vcx.mk_old_expr(plugged))]);
            let merge = vcx.mk_method(
                merge,
                decls,
                &[],
                vcx.alloc_slice(&[with_bindings(conjunction(&preds))]),
                vcx.alloc_slice(&[with_bindings(merge_post)]),
                None,
            );
            let mut split_posts = preds;
            split_posts.push(vcx.mk_eq_expr(plugged, vcx.mk_old_expr(fetched)));
            let split = vcx.mk_method(
                split,
                decls,
                &[],
                vcx.alloc_slice(&[with_bindings(merged)]),
                vcx.alloc_slice(&[with_bindings(conjunction(&split_posts))]),
                None,
            );
            Ok(([merge, split], ()))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for methods in Self::all_outputs_local_no_errors(program) {
            for method in methods {
                program.add_method(method);
            }
        }
    }
}

/// `parts` joined by `&&` from the left, which Viper prints as one flat
/// conjunction, in the order the parts frame each other.
// TODO: `vcx.mk_conj` builds a balanced tree, which keeps long conjunctions shallow but
// prints with parentheses; once it joins short conjunctions from the left, use it here.
fn conjunction<'vir>(parts: &[vir::ExprBool<'vir>]) -> vir::ExprBool<'vir> {
    vir::with_vcx(|vcx| {
        parts
            .iter()
            .copied()
            .reduce(|conj, part| {
                vcx.mk_bin_op_expr(vir::BinOpKind::And, conj, part)
                    .downcast_ty()
            })
            .unwrap_or_else(|| vcx.mk_bool::<true>())
    })
}

/// Tests that verify `merge`/`split` with Silicon: merging the predicates of
/// the positions of a block, and casting the result to the predicate of a type
/// parameter where the block is one, keeps what is fetched through them;
/// likewise casting back and splitting.
#[cfg(test)]
mod verification_tests {
    use prusti_rustc_interface::{index::Idx, span::DUMMY_SP};
    use task_encoder::TaskEncoder;

    use super::{LifetimePosition, MergeSplitEnc};
    use crate::encoders::ty::{
        RustTyDecomposition, generics::IndirectCastersEnc, indirect_pred::IndirectPredEnc,
        test_util::verify,
    };

    /// Asserts that the methods in `viper` verify together with the indirect
    /// predicate of the type alias `X` defined in `src`, its casts, and
    /// `merge`/`split` of `X` for each of `blocks`, given by its first and end
    /// position; and that each of them does not with `assert false` at the end
    /// of its body, the probe that its state is consistent.
    fn assert_merge_split(src: &str, blocks: &[(usize, usize)], viper: &str) {
        let verify_with_merge = |viper: &str| {
            verify(
                src,
                |ty| {
                    IndirectPredEnc::encode(ty.ty, true, DUMMY_SP).unwrap();
                    for &(first, end) in blocks {
                        let block = (ty, LifetimePosition::new(first), LifetimePosition::new(end));
                        MergeSplitEnc::encode(block, true, DUMMY_SP).unwrap();
                    }
                    let casts = (RustTyDecomposition::param(), ty.ty);
                    IndirectCastersEnc::encode(casts, true, DUMMY_SP).unwrap();
                },
                viper,
            )
        };
        if let Err(output) = verify_with_merge(viper) {
            panic!("verification failed:\n{output}");
        }
        // The bodies end in a line of just `}`.
        let ends = viper
            .match_indices("\n            }")
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        assert!(!ends.is_empty(), "no method body to probe");
        for end in ends {
            let probe = format!(
                "{}\n              assert false{}",
                &viper[..end],
                &viper[end..]
            );
            match verify_with_merge(&probe) {
                Ok(()) => panic!("probe verified:\n{probe}"),
                Err(output) => assert!(
                    output.contains("Assert might fail"),
                    "probe failed differently:\n{output}"
                ),
            }
        }
    }

    /// `&'a mut i32` as a type parameter, merge/split are identities.
    #[test]
    fn reference_as_param() {
        assert_merge_split(
            "type X<'a> = &'a mut i32;",
            &[(0, 1)],
            r#"
            define I32 s_Int_i32_type()
            define RI s_Ref_mutable_type(I32)
            define G(k) make_generic_s_Ref_mutable(k, I32)

            method merge(k: s_Ref_mutable)
              requires p_Ref_mutable_ind(k, I32, 0, 1)
              ensures p_Param_ind(G(k), RI, 0, 1)
              ensures fetch_Param(G(k), RI, 0, 1) == old(G(fetch_Ref_mutable(k, I32, 0, 1)))
            {
              merge_ind_Ref_mutable$lt$Int_i32$gt$_0_1(k)
              make_generic_ind_Ref_mutable(k, I32, 0, 1)
            }

            method split(k: s_Ref_mutable)
              requires p_Param_ind(G(k), RI, 0, 1)
              ensures p_Ref_mutable_ind(k, I32, 0, 1)
              ensures G(fetch_Ref_mutable(k, I32, 0, 1)) == old(fetch_Param(G(k), RI, 0, 1))
            {
              make_concrete_ind_Ref_mutable(k, I32, 0, 1)
              split_ind_Ref_mutable$lt$Int_i32$gt$_0_1(k)
            }
            "#,
        );
    }

    /// `N<'b, 'a>` as a type parameter, with `'b` behind `'a`.
    #[test]
    fn nested_reference_as_param() {
        assert_merge_split(
            "struct N<'b, 'a>(&'a mut &'b mut i32);
             type X<'b, 'a> = N<'b, 'a>;",
            &[(0, 2)],
            r#"
            define NT s_N_type()
            define G(k) make_generic_s_N(k)
            // The key of the predicate of 'b: `k` with what 'a reads plugged in.
            define KB(k) plug_N(k, fetch_N(k, 1, 2), 1, 2)

            method merge(k: s_N)
              requires p_N_ind(k, 1, 2) && p_N_ind(KB(k), 0, 1)
              ensures p_Param_ind(G(k), NT, 0, 2)
              ensures fetch_Param(G(k), NT, 0, 2) == old(G(fetch_N(KB(k), 0, 1)))
            {
              merge_ind_N_0_2(k)
              make_generic_ind_N(k, 0, 2)
            }

            method split(k: s_N)
              requires p_Param_ind(G(k), NT, 0, 2)
              ensures p_N_ind(k, 1, 2) && p_N_ind(KB(k), 0, 1)
              ensures G(fetch_N(KB(k), 0, 1)) == old(fetch_Param(G(k), NT, 0, 2))
            {
              make_concrete_ind_N(k, 0, 2)
              split_ind_N_0_2(k)
            }
            "#,
        );
    }

    /// `A<'a, 'b>` as a type parameter, with unrelated `'a` and `'b`.
    #[test]
    fn unrelated_positions_as_param() {
        assert_merge_split(
            "struct A<'a, 'b>(&'a mut i32, &'b mut i32);
             type X<'a, 'b> = A<'a, 'b>;",
            &[(0, 2)],
            r#"
            define AT s_A_type()
            define G(k) make_generic_s_A(k)

            method merge(k: s_A)
              requires p_A_ind(k, 0, 1) && p_A_ind(k, 1, 2)
              ensures p_Param_ind(G(k), AT, 0, 2)
              ensures fetch_Param(G(k), AT, 0, 2).make_concrete_s_A.s_A_0 == old(fetch_A(k, 0, 1).s_A_0)
              ensures fetch_Param(G(k), AT, 0, 2).make_concrete_s_A.s_A_1 == old(fetch_A(k, 1, 2).s_A_1)
            {
              merge_ind_A_0_2(k)
              make_generic_ind_A(k, 0, 2)
            }

            method split(k: s_A)
              requires p_Param_ind(G(k), AT, 0, 2)
              ensures p_A_ind(k, 0, 1) && p_A_ind(k, 1, 2)
              ensures fetch_A(k, 0, 1).s_A_0 == old(fetch_Param(G(k), AT, 0, 2)).make_concrete_s_A.s_A_0
              ensures fetch_A(k, 1, 2).s_A_1 == old(fetch_Param(G(k), AT, 0, 2)).make_concrete_s_A.s_A_1
            {
              make_concrete_ind_A(k, 0, 2)
              split_ind_A_0_2(k)
            }
            "#,
        );
    }

    /// `&'a mut T` as a type parameter, in a context generic in `T`.
    #[test]
    fn generic_reference_as_param() {
        assert_merge_split(
            "type X<'a, T> = &'a mut T;",
            &[(0, 2)],
            r#"
            define RT(T) s_Ref_mutable_type(T)
            define G(k, T) make_generic_s_Ref_mutable(k, T)
            // The key of the predicate of the positions of `T`: `k` with what 'a
            // reads plugged in.
            define KT(k, T) plug_Ref_mutable(k, T, fetch_Ref_mutable(k, T, 0, 1), 0, 1)

            method merge(k: s_Ref_mutable, T: Type)
              requires p_Ref_mutable_ind(k, T, 0, 1)
              requires p_Ref_mutable_ind(KT(k, T), T, 1, 1 + arity(T))
              ensures p_Param_ind(G(k, T), RT(T), 0, 1 + arity(T))
              ensures fetch_Param(G(k, T), RT(T), 0, 1 + arity(T)) ==
                old(G(fetch_Ref_mutable(KT(k, T), T, 1, 1 + arity(T)), T))
            {
              merge_ind_Ref_mutable$lt$P0$gt$_0_2(k, T)
              make_generic_ind_Ref_mutable(k, T, 0, 1 + arity(T))
            }

            method split(k: s_Ref_mutable, T: Type)
              requires p_Param_ind(G(k, T), RT(T), 0, 1 + arity(T))
              ensures p_Ref_mutable_ind(k, T, 0, 1)
              ensures p_Ref_mutable_ind(KT(k, T), T, 1, 1 + arity(T))
              ensures G(fetch_Ref_mutable(KT(k, T), T, 1, 1 + arity(T)), T) ==
                old(fetch_Param(G(k, T), RT(T), 0, 1 + arity(T)))
            {
              make_concrete_ind_Ref_mutable(k, T, 0, 1 + arity(T))
              split_ind_Ref_mutable$lt$P0$gt$_0_2(k, T)
            }
            "#,
        );
    }

    /// `&'a mut &'b mut T` as `&'c mut U`: merges `'b` and `T`.
    #[test]
    fn generic_nested_reference_as_reference_to_param() {
        assert_merge_split(
            "type X<'a, 'b, T> = &'a mut &'b mut T;",
            &[(1, 3)],
            r#"
            define RT(T) s_Ref_mutable_type(T)
            // The keys of the predicates of 'b and of the positions of `T`.
            define KB(k, T) plug_Ref_mutable(k, RT(T), fetch_Ref_mutable(k, RT(T), 0, 1), 0, 1)
            define KT(k, T) plug_Ref_mutable(KB(k, T), RT(T), fetch_Ref_mutable(KB(k, T), RT(T), 1, 2), 1, 2)

            method merge(k: s_Ref_mutable, T: Type)
              requires p_Ref_mutable_ind(k, RT(T), 0, 1)
              requires p_Ref_mutable_ind(KB(k, T), RT(T), 1, 2)
              requires p_Ref_mutable_ind(KT(k, T), RT(T), 2, 2 + arity(T))
              ensures p_Ref_mutable_ind(k, RT(T), 0, 1)
              ensures p_Ref_mutable_ind(KB(k, T), RT(T), 1, 2 + arity(T))
              ensures fetch_Ref_mutable(KB(k, T), RT(T), 1, 2 + arity(T)) ==
                old(fetch_Ref_mutable(KT(k, T), RT(T), 2, 2 + arity(T)))
            {
              merge_ind_Ref_mutable$lt$Ref_mutable$lt$P0$gt$$gt$_1_3(KB(k, T), T)
            }

            method split(k: s_Ref_mutable, T: Type)
              requires p_Ref_mutable_ind(k, RT(T), 0, 1)
              requires p_Ref_mutable_ind(KB(k, T), RT(T), 1, 2 + arity(T))
              ensures p_Ref_mutable_ind(k, RT(T), 0, 1)
              ensures p_Ref_mutable_ind(KB(k, T), RT(T), 1, 2)
              ensures p_Ref_mutable_ind(KT(k, T), RT(T), 2, 2 + arity(T))
              ensures fetch_Ref_mutable(KT(k, T), RT(T), 2, 2 + arity(T)) ==
                old(fetch_Ref_mutable(KB(k, T), RT(T), 1, 2 + arity(T)))
            {
              split_ind_Ref_mutable$lt$Ref_mutable$lt$P0$gt$$gt$_1_3(KB(k, T), T)
            }
            "#,
        );
    }

    /// `(T, &'a mut &'b mut i32)`: merges `'a` and `'b`, after `T`.
    #[test]
    fn block_after_type_param() {
        assert_merge_split(
            "type X<'a, 'b, T> = (T, &'a mut &'b mut i32);",
            &[(1, 3)],
            r#"
            define RRI s_Ref_mutable_type(s_Ref_mutable_type(s_Int_i32_type()))
            // The key of the predicate of 'b: `k` with what 'a reads plugged in.
            define KB(k, T) plug_2_Tuple(k, T, RRI, fetch_2_Tuple(k, T, RRI, arity(T), arity(T) + 1), arity(T), arity(T) + 1)

            method merge(k: s_2_Tuple, T: Type)
              requires p_2_Tuple_ind(k, T, RRI, arity(T), arity(T) + 1)
              requires p_2_Tuple_ind(KB(k, T), T, RRI, arity(T) + 1, arity(T) + 2)
              ensures p_2_Tuple_ind(k, T, RRI, arity(T), arity(T) + 2)
              ensures fetch_2_Tuple(k, T, RRI, arity(T), arity(T) + 2) ==
                old(fetch_2_Tuple(KB(k, T), T, RRI, arity(T) + 1, arity(T) + 2))
            {
              merge_ind_2_Tuple$lt$P0$com$$sp$Ref_mutable$lt$Ref_mutable$lt$Int_i32$gt$$gt$$gt$_1_3(k, T)
            }

            method split(k: s_2_Tuple, T: Type)
              requires p_2_Tuple_ind(k, T, RRI, arity(T), arity(T) + 2)
              ensures p_2_Tuple_ind(k, T, RRI, arity(T), arity(T) + 1)
              ensures p_2_Tuple_ind(KB(k, T), T, RRI, arity(T) + 1, arity(T) + 2)
              ensures fetch_2_Tuple(KB(k, T), T, RRI, arity(T) + 1, arity(T) + 2) ==
                old(fetch_2_Tuple(k, T, RRI, arity(T), arity(T) + 2))
            {
              split_ind_2_Tuple$lt$P0$com$$sp$Ref_mutable$lt$Ref_mutable$lt$Int_i32$gt$$gt$$gt$_1_3(k, T)
            }
            "#,
        );
    }

    /// `Option<&'a mut Box<List<'a, 'b>>>` as `Option<T>`.
    #[test]
    fn recursive_type_in_option() {
        assert_merge_split(
            "struct List<'a, 'b> {
                 x: &'a mut i32,
                 y: &'b mut i32,
                 next: Option<Box<List<'a, 'b>>>,
             }
             type X<'a, 'b> = Option<&'a mut Box<List<'a, 'b>>>;",
            &[(0, 3)],
            r#"
            define RBL s_Ref_mutable_type(s_Box_type(s_List_type(), s_Global_type()))
            // The key of the predicates of `List`'s positions: `k` with what the
            // reference's 'a reads plugged in.
            define KA(k) plug_Option(k, RBL, fetch_Option(k, RBL, 0, 1), 0, 1)
            // What the positions of `List` read, plugged together.
            define LIST(k) plug_Option(fetch_Option(KA(k), RBL, 1, 2), RBL, fetch_Option(KA(k), RBL, 2, 3), 2, 3)

            method merge(k: s_Option)
              requires p_Option_ind(k, RBL, 0, 1)
              requires p_Option_ind(KA(k), RBL, 1, 2) && p_Option_ind(KA(k), RBL, 2, 3)
              ensures p_Option_ind(k, RBL, 0, 3)
              ensures fetch_Option(k, RBL, 0, 3) == old(LIST(k))
            {
              merge_ind_Option$lt$Ref_mutable$lt$Box$lt$List$com$$sp$Global$gt$$gt$$gt$_0_3(k)
            }

            method split(k: s_Option)
              requires p_Option_ind(k, RBL, 0, 3)
              ensures p_Option_ind(k, RBL, 0, 1)
              ensures p_Option_ind(KA(k), RBL, 1, 2) && p_Option_ind(KA(k), RBL, 2, 3)
              ensures LIST(k) == old(fetch_Option(k, RBL, 0, 3))
            {
              split_ind_Option$lt$Ref_mutable$lt$Box$lt$List$com$$sp$Global$gt$$gt$$gt$_0_3(k)
            }
            "#,
        );
    }
}
