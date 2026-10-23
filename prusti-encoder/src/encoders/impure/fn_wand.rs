use crate::encoders::{
    EncodeResult, ImpureEncVisitor, MirLocalDefEncOutput, MirSpecEnc,
    pure::spec::{EncodedPledge, MirSpecEncMode, PledgeArgs, PledgeExpr},
    ty::{
        RustTyDecomposition,
        generics::{GArgs, GArgsTyEnc, GParams, GenericParamsEnc, ind_casters::IndCastersEnc},
        indirect_node::{IndirectNodes, IndirectNodesEnc, RegionIdx, node_for_generalized_idx},
    },
};
use pcg::borrow_pcg::{
    FunctionData, FunctionShape, FunctionShapeInput, FunctionShapeNode, FunctionShapeOutput,
    MakeFunctionShapeError, region_projection::Generalized, state::BorrowsState,
    unblock_graph::UnblockGraph,
};
use prusti_interface::PrustiError;
use prusti_rustc_interface::{
    data_structures::fx::{FxHashMap, FxHashSet},
    middle::{mir, ty},
    span::def_id::DefId,
};
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, HasType};

/// Encodes the magic wands given a function signature.
pub struct WandEnc;

/// A `RegionsIn(T)` node of a callee's signature (see `generic_nodes`).
#[derive(Clone)]
pub(crate) struct GenericNode<'vir> {
    /// The argument (or the result) the node belongs to.
    pub local: mir::Local,
    pub node: RegionIdx,
    /// The callee's view of the local's type, with the caller's type arguments.
    pub nodes: IndirectNodes<'vir>,
    /// For an argument: whether the result blocks it (its pieces come back
    /// through a wand); `None` for the result.
    pub blocked: Option<bool>,
}

#[derive(Clone, Debug)]
pub enum WandEncError {
    Unsupported(String),
}

impl<'vir, E: TaskEncoder> ImpureEncVisitor<'vir, '_, E> {
    pub fn package_wands(
        &mut self,
        final_borrow_state: &BorrowsState<'_, 'vir>,
    ) -> EncodeResult<'vir, Vec<vir::Stmt<'vir>>, E> {
        let mut wand_packages = Vec::new();
        let label = self.new_label("package_post");
        let result = self.local_defs[mir::RETURN_PLACE].impure_snap;
        let result = self.vcx.mk_local_labelled_old_expr(result, label);
        let args = self
            .local_defs
            .args()
            .map(|a| self.vcx.mk_old_expr(a.impure_snap));
        let args = PledgeExpr::pledge_args(result, args);

        for wand_data in self.wands.viper_wands() {
            let Some(wand) = self
                .wands
                .mk_wand(&wand_data, args, None, None, self.vcx, self.deps)
            else {
                continue;
            };
            let mut package_script = Vec::new();
            // each package starts from the state at the end of the body
            let mut locals = wand_data.rhs.iter().map(|g| g.mir_local()).collect::<Vec<_>>();
            locals.dedup();
            for local in locals {
                let ug = UnblockGraph::for_node(
                    mir::Place::from(local),
                    final_borrow_state,
                    self.pcg_ctxt(),
                );
                let actions = ug.actions(self.pcg_ctxt()).unwrap();
                let unblock = self.block(|visitor| {
                    visitor.pcs_unblock_actions(final_borrow_state, &actions, Some(label))
                })?;
                package_script.extend(unblock);
            }

            if !wand_data.pledges.is_empty() {
                // Statements in the package script only see resources already
                // in the package state. A resource that is not obtained from
                // the LHS (e.g. an argument the result does not borrow from)
                // is only moved there when consumed, which for the RHS happens
                // after the script. Asserting the RHS resources moves them in
                // early, so that the pledge exhales below can read them.
                let resources = wand_data
                    .rhs
                    .iter()
                    .filter_map(|g| {
                        self.wands.encode_predicates_for_function_shape_node(
                            self.vcx,
                            self.deps,
                            *g,
                            None,
                            |i| args[i],
                        )
                    })
                    .collect::<Vec<_>>();
                package_script.push(self.vcx.mk_assert_stmt(self.vcx.mk_conj(&resources)));
            }

            for EncodedPledge {
                expiry_postcondition,
                ..
            } in &wand_data.pledges
            {
                let span = expiry_postcondition.span();
                self.vcx.with_span(span, |vcx| {
                    vcx.handle_error("exhale.failed:assertion.false", move |_| {
                        Some(vec![PrustiError::verification(
                            "pledge postcondition might not hold",
                            span.into(),
                        )])
                    });
                    package_script.push(vcx.mk_exhale_stmt(expiry_postcondition.expr(args)));
                });
            }
            wand_packages.push(
                self.vcx
                    .mk_package_stmt(wand, self.vcx.alloc_slice(&package_script)),
            );
        }
        Ok(wand_packages)
    }
}

type EncodedPledges<'vir> = Vec<EncodedPledge<'vir>>;

/// Not tied to a caller or callee context. `indirect_pres`, `indirect_posts`,
/// `wand_posts`, and `package_wands` are identity-substituted and intended for
/// use in the callee's own contract; `apply_wands` is for caller use and
/// re-substitutes via a [`WandCallContext`].
#[derive(Clone)]
pub struct WandEncOutput<'vir> {
    /// Information about the corresponding function.
    function_data: FunctionData<'vir>,

    /// The lifetime projections of all arguments to the function.
    inputs: Vec<FunctionShapeInput<Generalized>>,

    /// The lifetime projections of all function outputs (according to the
    /// corresponding [`FunctionShape`]). This *includes* lifetime projections
    /// of nested lifetimes in the function arguments.
    outputs: Vec<FunctionShapeOutput<Generalized>>,

    /// Encoded VIR expressions for the magic wands.
    wands: Vec<WandData<'vir>>,

    /// `RegionsIn(T)` nodes of by-value arguments whose pieces move into the
    /// result's (or a reference's) values: neither given back nor blocked.
    consumed: Vec<FunctionShapeInput<Generalized>>,
}

/// Substitution context for instantiating a wand at a call site. When `None`,
/// the wand is encoded using the callee's identity substitution (appropriate
/// when emitting wands inside the function being defined). When `Some`, the
/// wand is re-encoded with the call-site substitutions and the caller's
/// generic parameters, so that placeholders like `Self` or other callee
/// generics are replaced by concrete types from the caller's perspective.
pub type WandCallContext<'vir> = Option<GArgs<'vir>>;

impl<'vir> WandEncOutput<'vir> {
    pub(crate) fn fn_sig(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        call_ctx: WandCallContext<'vir>,
    ) -> ty::FnSig<'vir> {
        match call_ctx {
            // TODO: change pcg's `fn_sig` to take `&[GenericArg]` instead of `GenericArgsRef`
            Some(ctx) => self
                .function_data
                .fn_sig(vcx.tcx(), vcx.tcx().mk_args(ctx.args())),
            None => self.function_data.identity_fn_sig(vcx.tcx()),
        }
    }

    pub(crate) fn g_params(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        call_ctx: WandCallContext<'vir>,
    ) -> GParams<'vir> {
        match call_ctx {
            Some(ctx) => ctx.context(),
            None => GParams::new(
                self.function_data.identity_substs(vcx.tcx()),
                self.function_data.param_env(vcx.tcx()),
                false,
            ),
        }
    }

    /// The indirect nodes of the type of the given node's base (the callee's
    /// view: of its identity signature; at a call site instantiated with the
    /// caller's type arguments), and the node among them; `None` for a node
    /// that holds nothing we can name.
    fn function_shape_node(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: FunctionShapeNode<Generalized>,
        call_ctx: WandCallContext<'vir>,
    ) -> Result<Option<(IndirectNodes<'vir>, RegionIdx)>, String> {
        let arg_ty = g.ty(self.fn_sig(vcx, None));
        let decomp = RustTyDecomposition::from_ty(arg_ty, self.g_params(vcx, None));
        let Some(node) = node_for_generalized_idx(arg_ty, g.region_idx(), decomp, vcx.tcx()) else {
            return Ok(None);
        };
        let nodes = deps
            .require_dep::<IndirectNodesEnc>(decomp)
            .map_err(|err| format!("indirect predicates of `{arg_ty}`: {err:?}"))?;
        let nodes = match call_ctx {
            None => nodes,
            Some(ctx) => {
                let args = deps
                    .require_dep::<GArgsTyEnc>(decomp.args.substitute(ctx))
                    .map_err(|err| format!("type arguments of `{arg_ty}`: {err:?}"))?;
                nodes.with_args(args)
            }
        };
        Ok(Some((nodes, node)))
    }

    /// The caller's snapshot of the local of `g` in the callee's view: for a
    /// local whose type is a type parameter of the callee, the generic
    /// snapshot of the caller's concrete one.
    fn callee_snap(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: FunctionShapeNode<Generalized>,
        call_ctx: WandCallContext<'vir>,
        snap: vir::ExprSnap<'vir>,
    ) -> vir::ExprSnap<'vir> {
        if call_ctx.is_none() || !matches!(g.ty(self.fn_sig(vcx, None)).kind(), ty::TyKind::Param(_)) {
            return snap;
        }
        let concrete = RustTyDecomposition::from_ty(g.ty(self.fn_sig(vcx, call_ctx)), self.g_params(vcx, call_ctx));
        if concrete.ty.specifics.is_param() {
            return snap;
        }
        let (Ok(casters), Ok(args)) = (
            deps.require_ref::<IndCastersEnc>((RustTyDecomposition::param(), concrete.ty)),
            deps.require_dep::<GArgsTyEnc>(concrete.args),
        ) else {
            return snap;
        };
        (casters.make_generic_s)(snap.downcast_ty(), args.get_ty(), args.get_const()).upcast_ty()
    }

    /// Does the given node hold any resources?
    fn has_resources(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: FunctionShapeNode<Generalized>,
        call_ctx: WandCallContext<'vir>,
    ) -> bool {
        // Unsupported nodes were rejected when encoding the wands.
        self.function_shape_node(vcx, deps, g, call_ctx)
            .unwrap()
            .is_some_and(|(nodes, node)| nodes.holds(node))
    }

    /// The position of the given node among those of its base: enclosing
    /// regions first, so that the keys of later nodes are well-defined.
    fn node_rank(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: FunctionShapeNode<Generalized>,
    ) -> (mir::Local, usize) {
        let position = self.function_shape_node(vcx, deps, g, None).unwrap().map_or(0, |(nodes, node)| {
            nodes.pieces_of(node).first().copied().unwrap_or(0)
        });
        (g.mir_local(), position)
    }

    fn encode_predicates_for_function_shape_node(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: impl Into<FunctionShapeNode<Generalized>>,
        call_ctx: WandCallContext<'vir>,
        mut snap: impl FnMut(mir::Local) -> vir::ExprSnap<'vir>,
    ) -> Option<vir::ExprBool<'vir>> {
        let g = g.into();
        let (nodes, node) = self.function_shape_node(vcx, deps, g, call_ctx).unwrap()?;
        let snap = self.callee_snap(vcx, deps, g, call_ctx, snap(g.mir_local()));
        nodes.pred(node, snap)
    }

    pub fn indirect_pres<'a, E: TaskEncoder>(
        &'a self,
        vcx: &'vir vir::VirCtxt<'vir>,
        local_defs: &'a MirLocalDefEncOutput<'vir>,
        deps: &'a mut TaskEncoderDependencies<'vir, E>,
    ) -> impl Iterator<Item = vir::ExprBool<'vir>> + 'a {
        let mut inputs = self.inputs().collect::<Vec<_>>();
        inputs.sort_by_cached_key(|g| self.node_rank(vcx, deps, (*g).into()));
        inputs.into_iter().filter_map(|g| {
            self.encode_predicates_for_function_shape_node(vcx, deps, g, None, |i| {
                local_defs[i].impure_snap
            })
        })
    }

    pub fn indirect_posts<'a, E: TaskEncoder>(
        &'a self,
        vcx: &'vir vir::VirCtxt<'vir>,
        local_defs: &'a MirLocalDefEncOutput<'vir>,
        deps: &'a mut TaskEncoderDependencies<'vir, E>,
    ) -> impl Iterator<Item = vir::ExprBool<'vir>> + 'a {
        // The encoded predicates for the input lifetime projections that are
        // not blocked by any of the result lifetime projections. These will be
        // encoded as part of the postcondition of the function (in contrast,
        // the predicates for the blocked inputs will appear on the right-hand
        // side of a magic wand in the postcondition).
        let mut unblocked_inputs = self
            .inputs()
            .filter(|i| !self.blocked_inputs().contains(i))
            .collect::<Vec<_>>();
        unblocked_inputs.sort_by_cached_key(|g| self.node_rank(vcx, deps, (*g).into()));
        let unblocked_input_posts = unblocked_inputs
            .into_iter()
            .filter_map(|lp| {
                self.encode_predicates_for_function_shape_node(vcx, deps, lp, None, |i| {
                    vcx.mk_old_expr(local_defs[i].impure_snap)
                })
            })
            .collect::<Vec<_>>()
            .into_iter();

        let mut outputs = self.outputs().collect::<Vec<_>>();
        outputs.sort_by_cached_key(|g| self.node_rank(vcx, deps, (*g).into()));
        let output_posts = outputs.into_iter().filter_map(|g| {
            self.encode_predicates_for_function_shape_node(vcx, deps, g, None, |i| {
                local_defs[i].impure_snap
            })
        });
        unblocked_input_posts.chain(output_posts)
    }

    pub fn wand_posts<'a, E: TaskEncoder>(
        &'a self,
        vcx: &'vir vir::VirCtxt<'vir>,
        local_defs: &'a MirLocalDefEncOutput<'vir>,
        deps: &'a mut TaskEncoderDependencies<'vir, E>,
    ) -> impl Iterator<Item = vir::ExprBool<'vir>> + 'a {
        let wand_result =
            vcx.mk_local_decl("wand_result", local_defs[mir::RETURN_PLACE].local_snap.ty());
        let wand_result_expr = vcx.mk_local_ex(wand_result);
        let args = local_defs
            .args()
            .map(|arg| vcx.mk_old_expr(arg.impure_snap));
        let args = PledgeExpr::pledge_args(wand_result_expr, args);

        // TODO: wands for late-bound regions
        self.viper_wands().into_iter().filter_map(move |wand_data| {
            let wand = self.mk_wand(&wand_data, args, None, None, vcx, deps)?;
            Some(vcx.mk_let_expr(
                wand_result,
                local_defs[mir::RETURN_PLACE].impure_snap,
                vcx.mk_wand_expr(wand),
            ))
        })
    }

    pub fn apply_wands<E: TaskEncoder>(
        &self,
        arguments: &[vir::ExprSnap<'vir>],
        label_pre: &'vir str,
        label_post: &'vir str,
        call_ctx: GArgs<'vir>,
        visitor: &mut ImpureEncVisitor<'vir, '_, E>,
    ) {
        let result = visitor
            .vcx
            .mk_local_labelled_old_expr(arguments[mir::RETURN_PLACE.as_usize()], label_post);
        let args = (1..arguments.len()).map(|l| {
            visitor
                .vcx
                .mk_local_labelled_old_expr(arguments[l], label_pre)
        });
        let args = PledgeExpr::pledge_args(result, args);
        for wand_data in self.viper_wands() {
            let Some(wand) = self.mk_wand(
                &wand_data,
                args,
                Some(label_pre),
                Some(call_ctx),
                visitor.vcx,
                visitor.deps,
            ) else {
                continue;
            };
            visitor.stmt(visitor.vcx.mk_apply_stmt(wand));
        }
    }

    fn mk_wand<E: TaskEncoder>(
        &self,
        wand_data: &WandData<'vir>,
        pledge_args: PledgeArgs<'vir>,
        pledge_old_label: Option<&'vir str>,
        call_ctx: WandCallContext<'vir>,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, E>,
    ) -> Option<vir::Wand<'vir>> {
        debug_assert!(!wand_data.lhs.is_empty());
        let generics = self.generics_subst(call_ctx, vcx, deps);
        let pledge_expr = |pledge: &PledgeExpr<'vir>| match pledge_old_label {
            Some(label) => {
                vcx.with_local_subst(generics, || pledge.expr_at_label(pledge_args, label))
            }
            None => pledge.expr(pledge_args),
        };
        let rhs = wand_data.rhs.iter().filter_map(|g| {
            self.encode_predicates_for_function_shape_node(vcx, deps, *g, call_ctx, |i| {
                pledge_args[i]
            })
        });
        let rhs = rhs
            .chain(
                wand_data
                    .pledges
                    .iter()
                    .map(|pledge| pledge_expr(&pledge.expiry_postcondition)),
            )
            .collect::<Vec<_>>();
        if rhs.is_empty() {
            // We skip emitting the wand when there is nothing on the RHS, i.e.,
            // nothing would be unblocked by applying this wand, nor are there
            // any pledge postconditions.
            return None;
        }
        let rhs = vcx.mk_conj(&rhs);
        let lhs = wand_data.lhs.iter().filter_map(|g| {
            self.encode_predicates_for_function_shape_node(vcx, deps, *g, call_ctx, |i| {
                pledge_args[i]
            })
        });
        let lhs = lhs
            .chain(
                wand_data
                    .pledges
                    .iter()
                    .filter_map(|pledge| pledge.expiry_obligation.as_ref().map(pledge_expr)),
            )
            .collect::<Vec<_>>();
        let lhs = vcx.mk_conj(&lhs);
        Some(vcx.mk_wand(lhs, rhs))
    }

    /// The pledges are encoded once, at the callee's identity substitution,
    /// so they refer to the callee's generic parameters. At a call site, these
    /// are replaced by the call's generic arguments, as Viper does for the
    /// callee's postcondition (from which the caller holds the wand).
    fn generics_subst<E: TaskEncoder>(
        &self,
        call_ctx: WandCallContext<'vir>,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, E>,
    ) -> &'vir FxHashMap<&'vir str, vir::ExprDyn<'vir>> {
        let Some(call_args) = call_ctx else {
            return vcx.alloc(FxHashMap::default());
        };
        let params = deps
            .require_dep::<GenericParamsEnc>(self.g_params(vcx, None))
            .unwrap();
        let args = deps.require_dep::<GArgsTyEnc>(call_args).unwrap();
        let tys = params
            .ty_decls()
            .iter()
            .zip(args.get_ty::<(), !>())
            .map(|(decl, arg)| (decl.name, arg.as_dyn()));
        let consts = params
            .const_decls()
            .iter()
            .zip(args.get_const::<(), !>())
            .map(|(decl, arg)| (decl.name, arg.as_dyn()));
        vcx.alloc(tys.chain(consts).collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WandEncTask<'tcx> {
    pub data: FunctionData<'tcx>,
}

impl<'tcx> WandEncTask<'tcx> {
    pub fn def_id(&self) -> DefId {
        self.data.def_id()
    }

    pub fn function_shape(
        &self,
        vcx: &vir::VirCtxt<'tcx>,
    ) -> Result<FunctionShape<Generalized>, MakeFunctionShapeError> {
        self.data.shape(vcx.tcx())
    }
}

pub type WandRhsKey = FunctionShapeInput<Generalized>;
pub type WandLhsKey = FunctionShapeNode<Generalized>;

#[derive(Clone, Debug)]
pub struct WandData<'vir> {
    /// Lifetime projections on the right-hand side of the wand. Guaranteed to be
    /// non-empty.
    rhs: Vec<WandRhsKey>,
    /// Lifetime projections on the left-hand side of the wand. Guaranteed to be
    /// non-empty.
    lhs: Vec<WandLhsKey>,
    pledges: EncodedPledges<'vir>,
}

impl<'vir> WandData<'vir> {
    pub fn new(lhs: Vec<WandLhsKey>, rhs: Vec<WandRhsKey>, pledges: EncodedPledges<'vir>) -> Self {
        debug_assert!(!lhs.is_empty());
        debug_assert!(!rhs.is_empty());
        Self { rhs, lhs, pledges }
    }
}

impl TaskEncoder for WandEnc {
    task_encoder::encoder_cache!(WandEnc);

    type TaskDescription<'vir> = WandEncTask<'vir>;

    type TaskKey<'vir> = WandEncTask<'vir>;

    type OutputFullDependency<'vir> = WandEncOutput<'vir>;

    type EncodingError = WandEncError;

    const ENCODER_NAME: &'static str = "wand encoder";

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        task.clone()
    }

    fn describe_error(error: Self::EncodingError) -> String {
        match error {
            WandEncError::Unsupported(message) => message,
        }
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(task_key.clone(), ())?;
        vir::with_vcx(|vcx| {
            let def_id = task_key.def_id();

            let shape = task_key.function_shape(vcx).map_err(|e| {
                EncodeFullError::EncodingError(
                    WandEncError::Unsupported(format!("function shape: {e:?}")),
                    None,
                )
            })?;

            let coupled_edges = shape.coupled_edges();
            let edges: FxHashSet<_> = shape.edges().map(|e| (e.input(), e.output())).collect();
            // An input that flows into the post-state of another argument is
            // moved there: the caller gets its pieces back through that
            // argument, not on its own.
            let mut moved = edges
                .iter()
                .filter(|(input, output)| {
                    output.mir_local() != mir::RETURN_PLACE && output.mir_local() != input.mir_local()
                })
                .map(|(input, _)| *input)
                .collect::<Vec<_>>();
            moved.sort();
            moved.dedup();

            let (inputs, outputs) = shape.take_inputs_and_outputs();
            let nodes_output = WandEncOutput {
                function_data: task_key.data,
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                wands: Vec::new(),
                consumed: Vec::new(),
            };
            for g in nodes_output
                .inputs()
                .map(FunctionShapeNode::from)
                .chain(nodes_output.outputs().map(FunctionShapeNode::from))
            {
                nodes_output
                    .function_shape_node(vcx, deps, g, None)
                    .map_err(|err| EncodeFullError::EncodingError(WandEncError::Unsupported(err), None))?;
            }
            let spec = deps.require_dep::<MirSpecEnc>((def_id, def_id, MirSpecEncMode::Impure))?;
            if coupled_edges.is_empty() {
                assert!(spec.pledges.is_empty());
                return Ok((
                    (),
                    WandEncOutput {
                        function_data: task_key.data,
                        inputs,
                        outputs,
                        wands: vec![],
                        consumed: moved,
                    },
                ));
            }
            let pledges = spec.pledges;
            let wands: Vec<WandData<'vir>> = coupled_edges
                .into_iter()
                .filter_map(|hyper_edge| {
                    let (sources, mut targets) = hyper_edge.into_tuple();
                    // We don't want to emit an identity wand, like P --* P. This can happen when
                    // PCG returns self-edges, like for fn(x: &'a mut &'b i32) where 'b is in
                    // invariant position and we therefore have an edge x|'b -> x|'b.
                    // Currently, these edges also prevent us from emitting indirect postconditions.
                    // TODO: we might want to emit these identity wands in the future to attach functional
                    // specifications to them. We still need to emit the resources on the wand's LHS.
                    let mut sources_as_nodes = sources
                        .iter()
                        .map(|&s| s.to_function_shape_node())
                        .collect::<Vec<_>>();
                    sources_as_nodes.sort();
                    targets.sort();
                    if sources_as_nodes == targets {
                        return None;
                    }
                    Some(WandData::new(targets, sources, pledges.clone()))
                })
                .collect();
            let mut output: WandEncOutput<'vir> = WandEncOutput {
                function_data: task_key.data,
                inputs,
                outputs,
                wands: Vec::new(),
                consumed: moved,
            };
            let wands = output.merge_enclosed(wands, vcx, deps);
            let wands = output.drop_owned_transfers(wands, !pledges.is_empty(), vcx, deps);
            output.wands = output
                .select_wands(wands, !pledges.is_empty(), &edges, vcx, deps)
                .map_err(|err| EncodeFullError::EncodingError(err, None))?;
            Ok(((), output))
        })
    }
}

impl<'vir> WandEncOutput<'vir> {
    /// Does node `a` enclose node `b` (both of the same local): a piece of
    /// `b` is keyed through a piece of `a`?
    fn encloses(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        a: FunctionShapeNode<Generalized>,
        b: FunctionShapeNode<Generalized>,
    ) -> bool {
        if a == b || a.mir_local() != b.mir_local() {
            return false;
        }
        let (Ok(Some((nodes, na))), Ok(Some((_, nb)))) = (
            self.function_shape_node(vcx, deps, a, None),
            self.function_shape_node(vcx, deps, b, None),
        ) else {
            return false;
        };
        nodes
            .pieces_of(nb)
            .into_iter()
            .any(|p| nodes.parents(p).iter().any(|q| nodes.piece(*q).node == na))
    }

    /// Is `g` a `RegionsIn(T)` node?
    fn is_regions_in(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: FunctionShapeNode<Generalized>,
    ) -> bool {
        matches!(self.function_shape_node(vcx, deps, g, None), Ok(Some((nodes, node))) if nodes.is_regions_in(node))
    }

    /// One wand per group of targets enclosing each other: an enclosed
    /// target's pieces are keyed through its encloser's, so they expire
    /// together (e.g. `RegionsIn(T)` of a result `&'a mut T` with `'a`). A
    /// `RegionsIn(T)` target that is the post-state of an argument whose node
    /// the wand gives back is dropped: behind the argument's reference, our
    /// encoding gives back that piece itself (keyed by the old reference, it
    /// holds the current referent); of a by-value argument, it is moved.
    fn merge_enclosed<E: TaskEncoder>(
        &self,
        mut wands: Vec<WandData<'vir>>,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, E>,
    ) -> Vec<WandData<'vir>> {
        'merge: loop {
            for i in 0..wands.len() {
                for j in i + 1..wands.len() {
                    let related = wands[i].lhs.iter().any(|a| {
                        wands[j].lhs.iter().any(|b| {
                            self.encloses(vcx, deps, *a, *b) || self.encloses(vcx, deps, *b, *a)
                        })
                    });
                    if related {
                        let other = wands.remove(j);
                        for g in other.lhs {
                            if !wands[i].lhs.contains(&g) {
                                wands[i].lhs.push(g);
                            }
                        }
                        for g in other.rhs {
                            if !wands[i].rhs.contains(&g) {
                                wands[i].rhs.push(g);
                            }
                        }
                        continue 'merge;
                    }
                }
            }
            break;
        }
        for wand in &mut wands {
            let returned = wand.rhs.iter().map(|g| g.to_function_shape_node()).collect::<Vec<_>>();
            let lhs = wand
                .lhs
                .iter()
                .copied()
                .filter(|g| !(returned.contains(g) && self.is_regions_in(vcx, deps, *g)))
                .collect::<Vec<_>>();
            if !lhs.is_empty() {
                wand.lhs = lhs;
            }
            wand.lhs.sort();
            wand.rhs.sort();
        }
        wands
    }

    /// Is `g` a `RegionsIn(T)` node of an owned position (not behind a
    /// reference of its local)?
    fn is_owned_regions_in(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: FunctionShapeNode<Generalized>,
    ) -> bool {
        match self.function_shape_node(vcx, deps, g, None) {
            Ok(Some((nodes, node))) if nodes.is_regions_in(node) => nodes
                .pieces_of(node)
                .into_iter()
                .all(|p| nodes.parents(p).is_empty()),
            _ => false,
        }
    }

    /// Without pledges, a wand whose targets (with resources) are all owned
    /// `RegionsIn(T)` nodes gives back nothing borrowed: those pieces moved
    /// with the values. It is dropped; its owned `RegionsIn(T)` sources are
    /// consumed, the others are given back unblocked. TODO: a caller cannot
    /// get the moved pieces back to the lenders (`id(&mut x)`).
    fn drop_owned_transfers<E: TaskEncoder>(
        &mut self,
        wands: Vec<WandData<'vir>>,
        has_pledges: bool,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, E>,
    ) -> Vec<WandData<'vir>> {
        if has_pledges {
            return wands;
        }
        let mut kept = Vec::new();
        for wand in wands {
            let transfer = wand.lhs.iter().all(|g| {
                !self.has_resources(vcx, deps, *g, None) || self.is_owned_regions_in(vcx, deps, *g)
            });
            if !transfer {
                kept.push(wand);
                continue;
            }
            for g in wand.rhs {
                if self.is_owned_regions_in(vcx, deps, g.to_function_shape_node())
                    && !self.consumed.contains(&g)
                {
                    self.consumed.push(g);
                }
            }
        }
        kept
    }

    /// Selects the wands to emit among those of the coupled edges, rejecting
    /// the shapes that cannot be encoded precisely.
    fn select_wands<E: TaskEncoder>(
        &self,
        wands: Vec<WandData<'vir>>,
        has_pledges: bool,
        edges: &FxHashSet<(WandRhsKey, WandLhsKey)>,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, E>,
    ) -> Result<Vec<WandData<'vir>>, WandEncError> {
        let has_resources = |g: FunctionShapeNode<Generalized>,
                             deps: &mut TaskEncoderDependencies<'vir, E>| {
            self.has_resources(vcx, deps, g, None)
        };
        // A wand is only needed if it gives back a resource, or to carry the
        // pledges if none does (e.g. for shared references).
        let (mut wands, resourceless): (Vec<_>, Vec<_>) = wands
            .into_iter()
            .partition(|wand_data| wand_data.rhs.iter().any(|g| has_resources((*g).into(), deps)));
        // A wand gives back its sources only once all of its targets have
        // expired. That matches the signature only if each source with
        // resources flows into each target with resources.
        for wand_data in &wands {
            let sources = wand_data
                .rhs
                .iter()
                .filter(|g| has_resources((**g).into(), deps))
                .collect::<Vec<_>>();
            // an enclosed target expires with its encloser: one group
            let mut targets: Vec<Vec<WandLhsKey>> = Vec::new();
            for t in &wand_data.lhs {
                if !has_resources(*t, deps)
                    || wand_data.lhs.iter().any(|a| self.encloses(vcx, deps, *a, *t))
                {
                    continue;
                }
                let mut group = vec![*t];
                for g in &wand_data.lhs {
                    if self.encloses(vcx, deps, *t, *g) {
                        group.push(*g);
                    }
                }
                targets.push(group);
            }
            let precise = sources.iter().all(|source| {
                targets
                    .iter()
                    .all(|group| group.iter().any(|target| edges.contains(&(**source, *target))))
            });
            if !precise {
                return Err(WandEncError::Unsupported(
                    "borrows in the result that expire separately but depend on a common \
                     argument are not supported"
                        .to_string(),
                ));
            }
        }
        if has_pledges {
            if wands.is_empty() {
                wands.extend(resourceless.into_iter().take(1));
            } else if wands.len() > 1 {
                // It is unclear which expiry the pledges refer to.
                return Err(WandEncError::Unsupported(
                    "pledges on a function whose result contains borrows that expire \
                     separately are not supported"
                        .to_string(),
                ));
            }
        }
        Ok(wands)
    }

    pub fn viper_wands(&self) -> Vec<WandData<'vir>> {
        self.wands.clone()
    }

    /// All lifetime projections in the arguments that are blocked by any of the
    /// lifetime projections in the function's result.
    pub fn blocked_inputs(&self) -> FxHashSet<FunctionShapeInput<Generalized>> {
        self.wands
            .iter()
            .flat_map(|wand| wand.rhs.iter().copied())
            .chain(self.consumed.iter().copied())
            .collect()
    }

    /// The callee's `RegionsIn(T)` nodes, in its view instantiated with
    /// `call_ctx`: a caller has to convert its pieces at those positions.
    pub(crate) fn generic_nodes(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        call_ctx: GArgs<'vir>,
    ) -> Vec<GenericNode<'vir>> {
        let blocked = self.blocked_inputs();
        let inputs = self.inputs().map(|i| (FunctionShapeNode::from(i), Some(blocked.contains(&i))));
        let outputs = self.outputs().map(|o| (FunctionShapeNode::from(o), None));
        inputs
            .chain(outputs)
            .filter_map(|(g, blocked)| {
                let (nodes, node) = self.function_shape_node(vcx, deps, g, Some(call_ctx)).ok()??;
                nodes.is_regions_in(node).then(|| GenericNode {
                    local: g.mir_local(),
                    node,
                    nodes,
                    blocked,
                })
            })
            .collect()
    }

    /// The `RegionsIn(T)` nodes of the arguments that the postcondition gives
    /// back unblocked.
    pub(crate) fn unblocked_regions_in(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
    ) -> Vec<(mir::Local, RegionIdx)> {
        let blocked = self.blocked_inputs();
        self.inputs()
            .filter(|i| !blocked.contains(i))
            .filter_map(|i| {
                let g = FunctionShapeNode::from(i);
                let (nodes, node) = self.function_shape_node(vcx, deps, g, None).ok()??;
                nodes.is_regions_in(node).then(|| (g.mir_local(), node))
            })
            .collect()
    }

    pub fn inputs(&self) -> impl Iterator<Item = FunctionShapeInput<Generalized>> + '_ {
        self.inputs.iter().copied()
    }

    pub fn outputs(&self) -> impl Iterator<Item = FunctionShapeOutput<Generalized>> + '_ {
        self.outputs.iter().copied()
    }
}
