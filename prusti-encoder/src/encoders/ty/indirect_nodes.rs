use pcg::borrow_pcg::{GeneralizedLifetime as PcgLifetime, region_projection::PcgRegion};
use prusti_rustc_interface::index::Idx;
use task_encoder::{EncodeFullError, TaskEncoder, TaskEncoderDependencies};

use super::{
    RustTyDecomposition,
    generics::{
        GArgsTyEnc,
        lifetime_position::{
            EnclosureGraph, GeneralizedLifetime, LifetimePosition, LifetimePositionEnc,
        },
    },
    indirect_pred::{IndirectPredEnc, IndirectPredRef, LifetimePositionBlock},
};

/// The predicates of the nodes on places of a type: one `p_T_ind` per
/// occurrence of the node's generalized lifetime in the type.
pub struct NodePredicates<'vir> {
    graph: EnclosureGraph<'vir>,
    ind: IndirectPredRef<'vir>,
    tys: &'vir [vir::ExprTyVal<'vir>],
    consts: &'vir [vir::ExprCSnap<'vir>],
}

impl<'vir> NodePredicates<'vir> {
    pub fn new<E: TaskEncoder>(
        ty: RustTyDecomposition<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, E>,
    ) -> Result<Self, EncodeFullError<'vir, E>> {
        let args = deps.require_dep::<GArgsTyEnc>(ty.args)?;
        Ok(Self {
            graph: deps.require_dep::<LifetimePositionEnc>(ty)?,
            ind: deps.require_ref::<IndirectPredEnc>(ty.ty)?,
            tys: args.get_ty(),
            consts: args.get_const(),
        })
    }

    /// All positions of the type, merged on the snapshot `snap`.
    fn block(&self, snap: vir::ExprSnap<'vir>) -> LifetimePositionBlock<'vir, '_> {
        LifetimePositionBlock {
            graph: &self.graph,
            first: LifetimePosition::new(0),
            end: LifetimePosition::new(self.graph.positions().count()),
            ind: self.ind,
            tys: self.tys,
            consts: self.consts,
            k: snap,
        }
    }

    /// The predicates of the node of `lifetime` on a place with snapshot
    /// `snap`, outermost first. `in_state` puts each key, which extends
    /// `snap`, in the state the node refers to.
    pub fn of(
        &self,
        lifetime: PcgLifetime<'vir>,
        snap: vir::ExprSnap<'vir>,
        mut in_state: impl FnMut(vir::ExprSnap<'vir>) -> vir::ExprSnap<'vir>,
    ) -> Vec<vir::ExprBool<'vir>> {
        let block = self.block(snap);
        block
            .positions()
            .into_iter()
            .filter(|&p| is_position_of(self.graph.lifetime(p), lifetime))
            .map(|p| {
                let (lo, hi) = self.graph.interval(p);
                block.predicate(in_state(block.key(p)), lo, hi)
            })
            .collect()
    }

    /// Whether the node of `lifetime` has any predicates.
    pub fn has_positions(&self, lifetime: PcgLifetime<'vir>) -> bool {
        self.graph
            .positions()
            .any(|p| is_position_of(self.graph.lifetime(p), lifetime))
    }

    pub fn deep_snap(&self, snap: vir::ExprSnap<'vir>) -> vir::ExprSnap<'vir> {
        let block = self.block(snap);
        block.fold(&block.positions())
    }
}

/// Whether a position of `position`'s lifetime belongs to the node of the PCG
/// lifetime `lifetime`. Positions are matched by lifetime, not by the PCG's
/// lifetime index, which counts each lifetime once.
fn is_position_of<'tcx>(position: GeneralizedLifetime<'tcx>, lifetime: PcgLifetime<'tcx>) -> bool {
    match (position, lifetime) {
        (GeneralizedLifetime::Region(r), PcgLifetime::Region(l)) => PcgRegion::from(r) == l,
        // The decomposition of a type parameter or alias has it as its only
        // argument.
        (GeneralizedLifetime::RegionsIn(param), PcgLifetime::RegionsIn(opaque)) => {
            vir::with_vcx(|vcx| param.args.args()[0].expect_ty() == opaque.ty(vcx.tcx()))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use pcg::borrow_pcg::GeneralizedLifetime as PcgLifetime;
    use prusti_rustc_interface::{middle::ty, span::DUMMY_SP};
    use task_encoder::TaskEncoder;

    use super::NodePredicates;
    use crate::encoders::ty::{
        RustTyDecomposition,
        generics::{GArgsTyEnc, lifetime_position::LifetimePositionEnc},
        indirect_pred::IndirectPredEnc,
        pure::TyPureEnc,
        test_util::{simplified, with_type_alias},
    };

    /// Calls `f` with the node predicates of the type alias `X` defined in
    /// `src`, the generalized lifetime named `name` in `X`, which is a region
    /// or a type parameter, and the snapshot `x` of a place of type `X`.
    fn with_node<R: Send>(
        src: &str,
        name: &str,
        f: impl for<'vir> FnOnce(&NodePredicates<'vir>, PcgLifetime<'vir>, vir::ExprSnap<'vir>) -> R
        + Send,
    ) -> R {
        let name = name.to_string();
        with_type_alias(src, move |ty: RustTyDecomposition<'_>| {
            let (_, _, graph) = LifetimePositionEnc::encode(ty, true, DUMMY_SP)
                .unwrap()
                .unwrap();
            let (ind, _, _) = IndirectPredEnc::encode(ty.ty, true, DUMMY_SP)
                .unwrap()
                .unwrap();
            let (_, _, args) = GArgsTyEnc::encode(ty.args, true, DUMMY_SP)
                .unwrap()
                .unwrap();
            let (pure, _, _) = TyPureEnc::encode(ty.ty, true, DUMMY_SP).unwrap().unwrap();
            let nodes = NodePredicates {
                graph,
                ind,
                tys: args.get_ty(),
                consts: args.get_const(),
            };
            let lifetime = ty
                .args
                .args()
                .iter()
                .flat_map(|arg| arg.walk())
                .find_map(|arg| match arg.kind() {
                    ty::GenericArgKind::Lifetime(region) if region.to_string() == name => {
                        Some(PcgLifetime::Region(region.into()))
                    }
                    ty::GenericArgKind::Type(arg_ty) if arg_ty.to_string() == name => {
                        Some(PcgLifetime::RegionsIn(arg_ty.try_into().unwrap()))
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("`X` has no lifetime or parameter `{name}`"));
            let x = vir::with_vcx(|vcx| vcx.mk_local_ex(vcx.mk_local_decl("x", pure.snapshot)));
            f(&nodes, lifetime, x)
        })
    }

    /// The predicates of the node of `name` on a place `x` of type `X`.
    fn node(src: &str, name: &str) -> Vec<String> {
        with_node(src, name, |nodes, lifetime, x| {
            nodes
                .of(lifetime, x, |key| key)
                .into_iter()
                .map(|pred| format!("{:?}", simplified(pred)))
                .collect()
        })
    }

    #[test]
    fn nested_mutable_references() {
        let src = "type X<'a, 'b> = &'a mut &'b mut i32;";
        assert_eq!(
            node(src, "'a"),
            ["p_Ref_mutable_ind(x, s_Ref_mutable_type(s_Int_i32_type()), 0, 1)"],
        );
        assert_eq!(
            node(src, "'b"),
            [
                "p_Ref_mutable_ind(extend_Ref_mutable(x, s_Ref_mutable_type(s_Int_i32_type()), 0, 1), s_Ref_mutable_type(s_Int_i32_type()), 1, 2)"
            ],
        );
    }

    #[test]
    fn repeated_lifetime() {
        let src = "struct Pair<'c, 'd>(&'c mut i32, &'d mut i32);
                   type X<'a, 'b> = (&'a mut i32, Pair<'a, 'b>);";
        assert_eq!(
            node(src, "'a"),
            [
                "p_2_Tuple_ind(x, s_Ref_mutable_type(s_Int_i32_type()), s_Pair_type(), 0, 1)",
                "p_2_Tuple_ind(x, s_Ref_mutable_type(s_Int_i32_type()), s_Pair_type(), 1, 2)",
            ],
        );
    }

    #[test]
    fn regions_in_type_parameter() {
        assert_eq!(
            node("type X<'a, T> = &'a mut T;", "T"),
            ["p_Ref_mutable_ind(extend_Ref_mutable(x, T$1, 0, 1), T$1, 1, (1) + (arity(T$1)))"],
        );
    }

    #[test]
    fn shared_reference() {
        assert_eq!(node("type X<'a, 'b> = &'a &'b mut i32;", "'a"), [""; 0]);
    }

    #[test]
    fn deep_snap() {
        let deep_snap = with_node(
            "type X<'a, 'b> = &'a mut &'b mut i32;",
            "'a",
            |nodes, _, x| format!("{:?}", simplified(nodes.deep_snap(x))),
        );
        assert_eq!(
            deep_snap,
            "extend_Ref_mutable(extend_Ref_mutable(x, s_Ref_mutable_type(s_Int_i32_type()), 0, 1), s_Ref_mutable_type(s_Int_i32_type()), 1, 2)",
        );
    }
}
