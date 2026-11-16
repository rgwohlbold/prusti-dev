use crate::encoders::{
    EncodeResult, ImpureEncVisitor, MirLocalDefEncOutput, MirSpecEnc,
    pure::spec::{EncodedPledge, MirSpecEncMode, PledgeArgs, PledgeExpr},
    ty::{
        RustTyDecomposition,
        generics::{GArgs, GArgsTyEnc, GParams, GenericParamsEnc},
        indirect::{IndirectPredicatesEnc, projection_for_generalized_idx},
    },
};
use pcg::borrow_pcg::{
    ArgIdxOrResult, FunctionData, FunctionShape, FunctionShapeInput, FunctionShapeNode,
    FunctionShapeOutput, GeneralizedLifetime, MakeFunctionShapeError,
    region_projection::{ExtractRegionsCtxt, Generalized},
    state::BorrowsState,
};
use prusti_interface::PrustiError;
use prusti_rustc_interface::{
    data_structures::fx::{FxHashMap, FxHashSet},
    index::IndexVec,
    middle::{mir, ty},
    span::def_id::DefId,
};
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, HasType};

/// Encodes the magic wands given a function signature.
pub struct WandEnc;

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
            let rhs = wand_data
                .rhs
                .iter()
                .map(|g| {
                    let resources = self.wands.encode_predicates_for_function_shape_node(
                        self.vcx,
                        self.deps,
                        *g,
                        None,
                        |i| args[i],
                    );
                    (
                        mir::Place::from(g.mir_local()),
                        resources.into_iter().collect(),
                    )
                })
                .collect();
            let mut package_script = self.pcs_package_script(final_borrow_state, rhs, label)?;

            for EncodedPledge {
                expiry_postcondition,
                ..
            } in &wand_data.pledges
            {
                let span = expiry_postcondition.span();
                self.vcx.with_span(span, |vcx| {
                    vcx.handle_error("exhale.failed:assertion.false", move |_| {
                        Some(vec![PrustiError::verification(
                            "pledge in the postcondition might not hold.",
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
/// use in the callee's own contract; `call_wands` is for caller use and
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

    /// The (unreified) predicates associated with the given node, or `None` if
    /// there are no resources associated with it.
    #[allow(clippy::type_complexity)]
    fn predicates_for_function_shape_node(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: FunctionShapeNode<Generalized>,
        call_ctx: WandCallContext<'vir>,
    ) -> Option<Vec<vir::ExprGenBool<'vir, vir::ExprSnap<'vir>, vir::ExprKind<'vir>>>> {
        let arg_ty = g.ty(self.fn_sig(vcx, call_ctx));
        let decomp = RustTyDecomposition::from_ty(arg_ty, self.g_params(vcx, call_ctx));
        let region_proj =
            projection_for_generalized_idx(arg_ty, g.region_idx(), decomp, vcx.tcx())?;
        let predicates = deps
            .require_dep::<IndirectPredicatesEnc>(region_proj)
            .unwrap()
            .predicate_applications;
        (!predicates.is_empty()).then_some(predicates)
    }

    fn encode_predicates_for_function_shape_node(
        &self,
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, impl TaskEncoder>,
        g: impl Into<FunctionShapeNode<Generalized>>,
        call_ctx: WandCallContext<'vir>,
        mut snap: impl FnMut(mir::Local) -> vir::ExprSnap<'vir>,
    ) -> Option<vir::ExprBool<'vir>> {
        use vir::Reify;
        let g = g.into();
        let predicates = self.predicates_for_function_shape_node(vcx, deps, g, call_ctx)?;

        let local = g.mir_local();
        let local_snap = snap(local);
        Some(
            vcx.mk_conj(
                &predicates
                    .iter()
                    .map(|p| p.reify(vcx, local_snap))
                    .collect::<Vec<_>>(),
            ),
        )
    }

    pub fn indirect_pres<'a, E: TaskEncoder>(
        &'a self,
        vcx: &'vir vir::VirCtxt<'vir>,
        local_defs: &'a MirLocalDefEncOutput<'vir>,
        deps: &'a mut TaskEncoderDependencies<'vir, E>,
    ) -> impl Iterator<Item = vir::ExprBool<'vir>> + 'a {
        self.inputs().filter_map(|g| {
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
        let unblocked_input_posts = self
            .inputs()
            .filter(|i| !self.blocked_inputs().contains(i))
            .filter_map(|lp| {
                self.encode_predicates_for_function_shape_node(vcx, deps, lp, None, |i| {
                    vcx.mk_old_expr(local_defs[i].impure_snap)
                })
            })
            .collect::<Vec<_>>()
            .into_iter();

        let output_posts = self.outputs().filter_map(|g| {
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

    pub fn call_wands<E: TaskEncoder>(
        &self,
        arguments: &[vir::ExprSnap<'vir>],
        label_pre: &'vir str,
        label_post: &'vir str,
        call_ctx: GArgs<'vir>,
        visitor: &mut ImpureEncVisitor<'vir, '_, E>,
    ) -> Vec<vir::Wand<'vir>> {
        let result = visitor
            .vcx
            .mk_local_labelled_old_expr(arguments[mir::RETURN_PLACE.as_usize()], label_post);
        let args = (1..arguments.len()).map(|l| {
            visitor
                .vcx
                .mk_local_labelled_old_expr(arguments[l], label_pre)
        });
        let args = PledgeExpr::pledge_args(result, args);
        let mut wands = Vec::new();
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
            wands.push(wand);
        }
        wands
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

            let (inputs, outputs) = shape.take_inputs_and_outputs();
            // TODO: The PCG gives the signature shape extra nodes: one for each
            // region in the trait bounds of a type parameter (`'a` for `x: T`
            // with `T: Tr<'a>`), inserted right after `RegionsIn(T)`, and one for
            // each input region, appended, if the result is a type parameter.
            // The indices of the shape's nodes refer to this extended list, but
            // `projection_for_generalized_idx` interprets them in the list of
            // `extract_regions`. Extra nodes at the end are out of range there
            // and get no resources, which matches their lack of positions; what
            // they stand for is part of `RegionsIn(T)`. An extra node before
            // another entry, however, shifts the indices of the entries after
            // it, which then get the resources of the wrong lifetime. Until the
            // PCG and the encoding agree on these nodes, such signatures are
            // rejected. The shape has a node for each entry of its list, so it
            // has extra nodes iff it has more nodes than `extract_regions` has
            // entries, and they can only precede an entry after a `RegionsIn`.
            let sig = task_key.data.identity_fn_sig(vcx.tcx());
            let bases = sig
                .inputs()
                .iter()
                .enumerate()
                .map(|(i, &ty)| (ArgIdxOrResult::Argument(i.into()), ty))
                .chain([(ArgIdxOrResult::Result, sig.output())]);
            for (base, ty) in bases {
                let nodes = inputs
                    .iter()
                    .map(|input| input.to_function_shape_node())
                    .chain(outputs.iter().copied())
                    .filter(|node| node.base() == base)
                    .map(|node| node.region_idx())
                    .collect::<FxHashSet<_>>();
                let lifetimes: IndexVec<_, GeneralizedLifetime<'vir>> =
                    vcx.tcx().extract_regions(ty);
                let regions_in_before_end = lifetimes
                    .iter()
                    .rev()
                    .skip(1)
                    .any(|lifetime| matches!(lifetime, GeneralizedLifetime::RegionsIn(_)));
                if nodes.len() > lifetimes.len() && regions_in_before_end {
                    return Err(EncodeFullError::EncodingError(
                        WandEncError::Unsupported(
                            "lifetimes in the trait bounds of a type parameter are not \
                             supported where other lifetimes follow the type parameter"
                                .to_string(),
                        ),
                        None,
                    ));
                }
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
            };
            output.wands = output
                .select_wands(wands, !pledges.is_empty(), &edges, vcx, deps)
                .map_err(|err| EncodeFullError::EncodingError(err, None))?;
            Ok(((), output))
        })
    }
}

impl<'vir> WandEncOutput<'vir> {
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
        let mut has_resources = |g: FunctionShapeNode<Generalized>| {
            self.predicates_for_function_shape_node(vcx, deps, g, None)
                .is_some()
        };
        // A wand is only needed if it gives back a resource, or to carry the
        // pledges if none does (e.g. for shared references).
        let (mut wands, resourceless): (Vec<_>, Vec<_>) = wands
            .into_iter()
            .partition(|wand_data| wand_data.rhs.iter().any(|g| has_resources((*g).into())));
        // A wand gives back its sources only once all of its targets have
        // expired. That matches the signature only if each source with
        // resources flows into each target with resources.
        for wand_data in &wands {
            let sources = wand_data
                .rhs
                .iter()
                .filter(|g| has_resources((**g).into()))
                .collect::<Vec<_>>();
            let targets = wand_data
                .lhs
                .iter()
                .filter(|g| has_resources(**g))
                .collect::<Vec<_>>();
            let precise = sources.iter().all(|source| {
                let mut targets = targets.iter();
                targets.all(|target| edges.contains(&(**source, **target)))
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
            .collect()
    }

    pub fn inputs(&self) -> impl Iterator<Item = FunctionShapeInput<Generalized>> + '_ {
        self.inputs.iter().copied()
    }

    pub fn outputs(&self) -> impl Iterator<Item = FunctionShapeOutput<Generalized>> + '_ {
        self.outputs.iter().copied()
    }
}

/// Tests that show the wands `WandEnc` builds for a function's signature.
#[cfg(test)]
mod tests {
    use pcg::borrow_pcg::FunctionData;
    use prusti_rustc_interface::{
        hir::def::DefKind,
        span::{DUMMY_SP, def_id::DefId},
    };
    use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
    use vir::CastType;

    use super::{WandEnc, WandEncTask};
    use crate::encoders::{
        TyUseImpureEnc,
        pure::spec::PledgeExpr,
        ty::{
            RustTyDecomposition,
            test_util::{dedent, simplified, with_crate},
        },
    };

    /// The wands of a function, one per line, or the error of `WandEnc`. As in
    /// `wand_posts`, the wands are on the result's snapshot and the old
    /// snapshots of the arguments, each named after its parameter. Building a
    /// wand requires the dependencies of an encoder.
    struct WandTestEnc;

    impl TaskEncoder for WandTestEnc {
        task_encoder::encoder_cache!(WandTestEnc);
        const ENCODER_NAME: &'static str = "wand test encoder";

        type TaskDescription<'vir> = DefId;
        type OutputFullDependency<'vir> = Vec<String>;

        fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
            *task
        }

        fn do_encode_full<'vir>(
            task_key: &Self::TaskKey<'vir>,
            deps: &mut TaskEncoderDependencies<'vir, Self>,
        ) -> EncodeFullResult<'vir, Self> {
            let def_id = *task_key;
            deps.emit_output_ref(def_id, ())?;
            let task = WandEncTask {
                data: FunctionData::new(def_id),
            };
            let wands = match deps.require_dep::<WandEnc>(task) {
                Ok(wands) => wands,
                // The error of `WandEnc` itself comes last, after its context.
                Err(EncodeFullError::DependencyError(errors)) => {
                    let (_, message, _) = errors.last().unwrap();
                    return Ok(((), vec![format!("error: {message}")]));
                }
                Err(error) => return Err(error),
            };
            vir::with_vcx(|vcx| {
                let sig = wands.fn_sig(vcx, None);
                let params = vcx.tcx().fn_arg_idents(def_id).iter();
                let names = std::iter::once("result".to_string())
                    .chain(params.map(|ident| ident.unwrap().to_string()));
                let tys = std::iter::once(sig.output()).chain(sig.inputs().iter().copied());
                let mut snaps = Vec::new();
                for (ty, name) in tys.zip(names) {
                    let ty = RustTyDecomposition::from_ty(ty, def_id);
                    let snapshot = deps.require_dep::<TyUseImpureEnc>(ty)?.snapshot();
                    let decl = vcx.mk_local_decl(vcx.alloc_str(&name), snapshot);
                    snaps.push(vcx.mk_local_ex(decl));
                }
                let args = snaps[1..].iter().map(|&snap| vcx.mk_old_expr(snap));
                let args = PledgeExpr::pledge_args(snaps[0], args);
                let mut lines = Vec::new();
                for wand_data in wands.viper_wands() {
                    if let Some(wand) = wands.mk_wand(&wand_data, args, None, None, vcx, deps) {
                        lines.push(format!("{} --* {}", show(wand.lhs), show(wand.rhs)));
                    }
                }
                Ok(((), lines))
            })
        }
    }

    /// Replaces `inhabited(..)` by `true`, which the simplifier then drops:
    /// it is not about borrows.
    struct DropInhabited;

    impl<'vir> vir::Folder<'vir, (), !> for DropInhabited {
        fn fold_expr(
            &mut self,
            e: vir::ExprGenDyn<'vir, (), !>,
        ) -> Option<vir::ExprGenDyn<'vir, (), !>> {
            match e.kind {
                vir::ExprKindGenData::FuncApp(app) if app.target == "inhabited" => {
                    Some(vir::with_vcx(|vcx| vcx.mk_bool::<true>().as_dyn()))
                }
                _ => e.super_fold_with(self),
            }
        }
    }

    fn show(e: vir::ExprBool<'_>) -> String {
        use vir::Foldable;
        format!(
            "{:?}",
            simplified(e.fold_with(&mut DropInhabited).unwrap_or(e))
        )
    }

    /// Asserts that the wands of the function `f` defined in `src` are
    /// `expected`.
    fn assert_wands(src: &str, expected: &str) {
        let lines = with_crate(src, |tcx| {
            let f = tcx
                .hir_crate_items(())
                .definitions()
                .find(|&def_id| {
                    tcx.def_kind(def_id) == DefKind::Fn && tcx.item_name(def_id).as_str() == "f"
                })
                .expect("the input defines no function `f`")
                .to_def_id();
            let (_, _, lines) = WandTestEnc::encode(f, true, DUMMY_SP).unwrap().unwrap();
            lines
        });
        assert_eq!(lines.join("\n"), dedent(expected));
    }

    #[test]
    fn reborrow() {
        assert_wands(
            "fn f<'a>(x: &'a mut i32) -> &'a mut i32 { x }",
            r#"
            p_Param(result.s_Ref_mutable_0, s_Int_i32_type()) --* p_Param(old(x).s_Ref_mutable_0, s_Int_i32_type())
            "#,
        );
    }

    #[test]
    fn nested_references_reborrow() {
        assert_wands(
            "fn f<'a, 'b>(x: &'a mut &'b mut i32) -> &'a mut i32 { *x }",
            r#"
            error: borrows in the result that expire separately but depend on a common argument are not supported
            "#,
        );
    }

    #[test]
    fn lifetime_in_struct() {
        assert_wands(
            "struct S<'a> { r: &'a mut i32 }
             fn f<'a, 'b>(s: &'b mut S<'a>) -> &'b mut i32 { s.r }",
            r#"
            error: borrows in the result that expire separately but depend on a common argument are not supported
            "#,
        );
    }

    #[test]
    fn generic_referent() {
        assert_wands(
            "fn f<'a, T>(x: &'a mut T) -> &'a mut T { x }",
            r#"
            p_Param(result.s_Ref_mutable_0, T$0) --* p_Param(old(x).s_Ref_mutable_0, T$0)
            "#,
        );
    }

    #[test]
    fn reborrow_into_adt() {
        assert_wands(
            "fn f<'a, T>(x: &'a mut Option<T>) -> Option<&'a mut T> { x.as_mut() }",
            r#"
            ((s_Option_discr(result)) == (s_Int_isize_cons(1))) ==> (p_Param(result.s_Option_1_0.make_concrete_s_Ref_mutable.s_Ref_mutable_0, T$0)) --* p_Param(old(x).s_Ref_mutable_0, s_Option_type(T$0))
            "#,
        );
    }
    /// Result borrows that expire separately get a wand each.
    #[test]
    fn two_wands() {
        assert_wands(
            "fn f<'a, 'b>(x: &'a mut i32, y: &'b mut i32) -> (&'a mut i32, &'b mut i32) { (x, y) }",
            r#"
            p_Param(result.s_2_Tuple_0.make_concrete_s_Ref_mutable.s_Ref_mutable_0, s_Int_i32_type()) --* p_Param(old(x).s_Ref_mutable_0, s_Int_i32_type())
            p_Param(result.s_2_Tuple_1.make_concrete_s_Ref_mutable.s_Ref_mutable_0, s_Int_i32_type()) --* p_Param(old(y).s_Ref_mutable_0, s_Int_i32_type())
            "#,
        );
    }

    /// The result may borrow from both arguments, which have different
    /// lifetimes.
    #[test]
    fn two_sources() {
        assert_wands(
            "fn f<'a, 'b: 'a>(x: &'a mut i32, y: &'b mut i32) -> &'a mut i32 { x }",
            r#"
            p_Param(result.s_Ref_mutable_0, s_Int_i32_type()) --* (p_Param(old(x).s_Ref_mutable_0, s_Int_i32_type())) && (p_Param(old(y).s_Ref_mutable_0, s_Int_i32_type()))
            "#,
        );
    }

    /// Two result lifetimes that both borrow from the same argument are given
    /// back together.
    #[test]
    fn two_targets() {
        assert_wands(
            "fn f<'a: 'b + 'c, 'b, 'c>(x: &'a mut i32) -> (&'b mut i32, &'c mut i32) { (x, todo!()) }",
            r#"
            (p_Param(result.s_2_Tuple_0.make_concrete_s_Ref_mutable.s_Ref_mutable_0, s_Int_i32_type())) && (p_Param(result.s_2_Tuple_1.make_concrete_s_Ref_mutable.s_Ref_mutable_0, s_Int_i32_type())) --* p_Param(old(x).s_Ref_mutable_0, s_Int_i32_type())
            "#,
        );
    }

    /// A lifetime that occurs twice in the result is one node.
    #[test]
    fn lifetime_twice() {
        assert_wands(
            "fn f<'a>(x: &'a mut (i32, i32)) -> (&'a mut i32, &'a mut i32) { (&mut x.0, &mut x.1) }",
            r#"
            (p_Param(result.s_2_Tuple_0.make_concrete_s_Ref_mutable.s_Ref_mutable_0, s_Int_i32_type())) && (p_Param(result.s_2_Tuple_1.make_concrete_s_Ref_mutable.s_Ref_mutable_0, s_Int_i32_type())) --* p_Param(old(x).s_Ref_mutable_0, s_2_Tuple_type(s_Int_i32_type(), s_Int_i32_type()))
            "#,
        );
    }

    /// The callee stores `y` in `*x`, so the caller gets back `**x` from
    /// before the call, and `*y`, once it gives up `**x` after the call.
    // TODO: The right side reads `**x` after the call as well: `old` covers
    // the snapshot of `x` but not the read of `*x`, which is outside it. So the
    // right side gives back `*y` twice, and `**x` from before the call never.
    #[test]
    fn store_into_argument() {
        assert_wands(
            "fn f<'a, 'b>(x: &'a mut &'b mut i32, y: &'b mut i32) { *x = y; }",
            r#"
            p_Param(p_Param_snap(old(x).s_Ref_mutable_0, s_Ref_mutable_type(s_Int_i32_type())).make_concrete_s_Ref_mutable.s_Ref_mutable_0, s_Int_i32_type()) --* (p_Param(p_Param_snap(old(x).s_Ref_mutable_0, s_Ref_mutable_type(s_Int_i32_type())).make_concrete_s_Ref_mutable.s_Ref_mutable_0, s_Int_i32_type())) && (p_Param(old(y).s_Ref_mutable_0, s_Int_i32_type()))
            "#,
        );
    }
}
