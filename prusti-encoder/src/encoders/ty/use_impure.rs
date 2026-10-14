use prusti_rustc_interface::abi;
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, PredicateIdn};

use crate::encoders::{
    Impure,
    ty::{
        LazyRustTy, RustTyDatas, RustTyDecomposition,
        generics::{GArgs, GArgsCastEnc, GArgsTyEnc, GParams},
        pure::TyPureEnc,
        use_inhabited::{TyUseInhabitedEnc, TyUseInhabitedRef},
    },
};

use super::{
    TyUseEnc, UseTyDatas,
    data::*,
    generics::{
        GArgCaster, GArgsTy,
        ind_casters::{IndCasters, IndCastersEnc},
    },
    impure::{ImpureTyDatas, TyImpureEnc},
    indirect_node::{IndirectNodes, IndirectNodesEnc, PrustiPcgCtxt, RegionIdx},
    indirect_pred::{TyIndirectEnc, TyIndirectExt},
    occurrences::Step,
    rust_ty::RustTySpecial,
};
use pcg::borrow_pcg::region_projection::ExtractRegionsCtxt;

pub(super) type UseImpureTyDatas = UseTyDatas<Impure>;

type FieldCaster<'vir> = GArgCaster<'vir, Impure>;

impl<'vir> TyDatas<'vir> for UseImpureTyDatas {
    type TyData = TyUseImpureData<'vir>;
    type PrimitiveData = ();
    type ArrayData = TyUseImpureArrayData<'vir>;
    type ImmRefData = TyUseImpureImmRef<'vir>;
    type MutRefData = TyUseImpureMutRef<'vir>;
    type RawData = TyUseImpureRaw<'vir>;
    type FieldData = TyUseImpureField<'vir>;
    type StructData = TyUseImpureStructData<'vir>;
    type VariantData = ();
    type EnumData = TyUseImpureEnumData<'vir>;
}

pub type TyUseImpure<'vir> = Ty<'vir, UseImpureTyDatas>;

pub type TyUseImpureArray<'vir> = ArrayData<'vir, UseImpureTyDatas>;
pub type TyUseImpureStruct<'vir> = StructData<'vir, UseImpureTyDatas>;
pub type TyUseImpureEnum<'vir> = EnumData<'vir, UseImpureTyDatas>;

#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureData<'vir> {
    args: GArgsTy<'vir>,
    impure: <ImpureTyDatas as TyDatas<'vir>>::TyData,
    inhabited: TyUseInhabitedRef<'vir>,
}

#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureImmRef<'vir> {
    #[allow(dead_code)]
    referent_caster: FieldCaster<'vir>,
    #[allow(dead_code)]
    metadata_caster: GArgCaster<'vir, crate::encoders::Pure>,
    #[allow(dead_code)]
    args: GArgsTy<'vir>,
    #[allow(dead_code)]
    impure: <ImpureTyDatas as TyDatas<'vir>>::ImmRefData,
}

#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureMutRef<'vir> {
    #[allow(dead_code)]
    referent_caster: FieldCaster<'vir>,
    #[allow(dead_code)]
    metadata_caster: GArgCaster<'vir, crate::encoders::Pure>,
    args: GArgsTy<'vir>,
    impure: <ImpureTyDatas as TyDatas<'vir>>::MutRefData,
    ref_to_snap: vir::FunctionIdn<'vir, (vir::Ref, vir::ManyTyVal, vir::ManyCSnap), vir::Snap>,
    /// The indirect nodes of this reference type; the reference's own
    /// lifetime (region 0) holds the referent. `None` if not encodable.
    nodes: Option<&'vir IndirectNodes<'vir>>,
    /// The pieces of the reference's own lifetime inside the referent (if it
    /// occurs there), which move with the referent on every fold/unfold.
    inner: Option<ReferentPieces<'vir>>,
    /// The pieces of the referent's own lifetimes, which a borrow `&mut p`
    /// moves into the reference's inner lifetimes (if the referent is concrete
    /// and does not mention the reference's own lifetime).
    referent: Option<ReferentPieces<'vir>>,
}

type RefToSnap<'vir> = vir::FunctionIdn<'vir, (vir::Ref, vir::ManyTyVal, vir::ManyCSnap), vir::Snap>;

/// Pieces of a concrete referent `p: U` (occurrence `o`) and the pieces of the
/// reference holding them (occurrence `In(0) ++ o`, whose body is the generic
/// piece `p_Param_ind(.., U, Q_o)`).
#[derive(Debug, Clone, Copy)]
pub struct ReferentPieces<'vir> {
    nodes: &'vir IndirectNodes<'vir>,
    casters: IndCasters<'vir>,
    /// The concrete referent's type arguments.
    ty_args: GArgsTy<'vir>,
    generic_snap: RefToSnap<'vir>,
    concrete_snap: RefToSnap<'vir>,
    /// `ext_Param`, for the keys of the generic pieces.
    param_ext: TyIndirectExt<'vir>,
    /// `(piece of the reference, piece of the referent)`, enclosing ones first.
    pairs: &'vir [(usize, usize)],
}

impl<'vir> ReferentPieces<'vir> {
    /// Cast the referent's pieces to generic, enclosed ones first (their keys
    /// read the enclosing concrete pieces).
    fn make_generic(&self, addr: vir::ExprRef<'vir>) -> Vec<vir::Stmt<'vir>> {
        let p_snap = self.concrete_snap.call()(addr, self.ty_args.get_ty(), self.ty_args.get_const());
        self.pairs
            .iter()
            .rev()
            .map(|(_, r)| {
                (self.casters.make_generic)(
                    self.nodes.key(*r, p_snap),
                    self.ty_args.get_ty(),
                    self.ty_args.get_const(),
                    self.nodes.piece(*r).query,
                )
                .alloc()
            })
            .collect()
    }

    /// Cast the referent's pieces to concrete, enclosed ones first (their keys
    /// read the enclosing generic pieces). `tys` are the generic type args.
    /// The keys are `slots` (per pair) if given, else read from the referent.
    fn make_concrete(
        &self,
        addr: vir::ExprRef<'vir>,
        tys: &'vir [vir::ExprTyVal<'vir>],
        slots: Option<Vec<vir::ExprSnap<'vir>>>,
    ) -> Vec<vir::Stmt<'vir>> {
        let p_generic = self.generic_snap.call()(addr, tys, &[]);
        self.pairs
            .iter()
            .enumerate()
            .rev()
            .map(|(i, (_, r))| {
                let key = match &slots {
                    Some(slots) => slots[i],
                    None => self.generic_key(*r, p_generic, tys),
                };
                (self.casters.make_concrete)(
                    key,
                    self.ty_args.get_ty(),
                    self.ty_args.get_const(),
                    self.nodes.piece(*r).query,
                )
                .alloc()
            })
            .collect()
    }

    /// The key of the generic piece of `r`, for the generic snapshot `p`.
    fn generic_key(
        &self,
        r: usize,
        p: vir::ExprSnap<'vir>,
        tys: &'vir [vir::ExprTyVal<'vir>],
    ) -> vir::ExprSnap<'vir> {
        self.nodes.parents(r).iter().fold(p, |k, parent| {
            let parent_key = self.generic_key(*parent, p, tys);
            (self.param_ext)(k, tys, &[], parent_key, self.nodes.piece(*parent).query)
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureRaw<'vir> {
    #[allow(dead_code)]
    metadata_caster: GArgCaster<'vir, crate::encoders::Pure>,
    #[allow(dead_code)]
    args: GArgsTy<'vir>,
    #[allow(dead_code)]
    impure: <ImpureTyDatas as TyDatas<'vir>>::RawData,
}

#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureArrayData<'vir> {
    #[allow(dead_code)]
    args: GArgsTy<'vir>,
    #[allow(dead_code)]
    impure: <ImpureTyDatas as TyDatas<'vir>>::ArrayData,
    element_caster: FieldCaster<'vir>,
}

/// Use-side data for the hardcoded extras of a `Box`.
#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureBoxData<'vir> {
    metadata_caster: GArgCaster<'vir, crate::encoders::Pure>,
    /// The types of the `Unique` pointer and of the allocator.
    unique: TyUseImpure<'vir>,
    alloc: TyUseImpure<'vir>,
}

#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureStructData<'vir> {
    args: GArgsTy<'vir>,
    ref_to_pred: PredicateIdn<'vir, (vir::Ref, vir::ManyTyVal, vir::ManyCSnap)>,
    impure: <ImpureTyDatas as TyDatas<'vir>>::StructData,
    box_data: Option<TyUseImpureBoxData<'vir>>,
}

#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureField<'vir> {
    caster: FieldCaster<'vir>,
    args: GArgsTy<'vir>,
    impure: <ImpureTyDatas as TyDatas<'vir>>::FieldData,
}

#[derive(Debug, Clone, Copy)]
pub struct TyUseImpureEnumData<'vir> {
    #[allow(dead_code)]
    args: GArgsTy<'vir>,
    impure: <ImpureTyDatas as TyDatas<'vir>>::EnumData,
    /// The inhabitedness of each variant's fields after normalization in this
    /// use site's generic context, paired with that variant's discriminant.
    variant_inhabited: &'vir [(vir::ExprCSnap<'vir>, vir::ExprBool<'vir>)],
}

/// Encodes a type into the predicate representation. Takes an arbitrary Rust
/// `Ty` and provides a wrapper around the results of the `TyImpureEnc` encoder.
/// This wrapper handles all the generic casts required (e.g. when fold/unfolding).
pub type TyUseImpureEnc = TyUseEnc<Impure>;

type EncResult<'vir, T> = Result<T, EncodeFullError<'vir, TyUseImpureEnc>>;

impl TaskEncoder for TyUseImpureEnc {
    task_encoder::encoder_cache!(TyUseImpureEnc);
    const ENCODER_NAME: &'static str = "impure type use encoder";

    type TaskDescription<'vir> = super::RustTyDecomposition<'vir>;

    type OutputFullDependency<'vir> = TyUseImpure<'vir>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut task_encoder::TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        deps.emit_output_ref(*task_key, ())?;

        let ty_impure = deps.require_dep::<TyImpureEnc>(task_key.ty)?;
        let inhabited = deps.require_ref::<TyUseInhabitedEnc>(*task_key)?;
        let mut walker = TyUseImpureWalker::new(deps, *task_key, inhabited)?;
        let ty_use_impure = walker.encode_ty(task_key.ty.zip(ty_impure))?;
        Ok(((), ty_use_impure.alloc()))
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        TyImpureEnc::emit_outputs(program)
    }
}

struct TyUseImpureWalker<'a, 'vir> {
    deps: &'a mut TaskEncoderDependencies<'vir, TyUseImpureEnc>,
    decomp: RustTyDecomposition<'vir>,
    args_t: GArgsTy<'vir>,
    args: GArgs<'vir>,
    inhabited: TyUseInhabitedRef<'vir>,
}

impl<'a, 'vir> TyUseImpureWalker<'a, 'vir> {
    fn new(
        deps: &'a mut TaskEncoderDependencies<'vir, TyUseImpureEnc>,
        decomp: RustTyDecomposition<'vir>,
        inhabited: TyUseInhabitedRef<'vir>,
    ) -> EncResult<'vir, Self> {
        let args = decomp.args;
        let args_t = deps.require_dep::<GArgsTyEnc>(args)?;
        Ok(Self {
            deps,
            decomp,
            args_t,
            args,
            inhabited,
        })
    }

    /// The indirect nodes of the reference type being encoded, the pieces of
    /// its own lifetime inside the referent, and those of the referent's
    /// lifetimes (see `TyUseImpureMutRef`).
    fn mutref_nodes(
        &mut self,
        referent: LazyRustTy<'vir>,
        params: GParams<'vir>,
    ) -> EncResult<
        'vir,
        (
            Option<&'vir IndirectNodes<'vir>>,
            Option<ReferentPieces<'vir>>,
            Option<ReferentPieces<'vir>>,
        ),
    > {
        // An unencodable type is rejected where its contracts are encoded.
        let Ok(nodes) = self.deps.require_dep::<IndirectNodesEnc>(self.decomp) else {
            return Ok((None, None, None));
        };
        if nodes.find(&[Step::Here(0)]).is_none() {
            return Ok((None, None, None));
        }
        let nodes = vir::with_vcx(|vcx| &*vcx.alloc(nodes));
        let own = RegionIdx::from(0);
        let own_region = PrustiPcgCtxt.extract_regions(self.decomp)[own];
        let normalized = referent.decompose_compare_normalize(params, self.args);
        let Some(normalized) = normalized else {
            return Ok((Some(nodes), None, None));
        };
        // TODO: a referent whose type cannot be encoded keeps its pieces.
        let Ok(referent_nodes) = self.deps.require_dep::<IndirectNodesEnc>(normalized.concrete) else {
            return Ok((Some(nodes), None, None));
        };
        let mentions_own = PrustiPcgCtxt
            .extract_regions(normalized.concrete)
            .iter()
            .any(|r| *r == own_region);
        let mut pairs = if mentions_own {
            nodes
                .pieces_of(own)
                .into_iter()
                .filter_map(|p| match nodes.piece(p).path.split_first() {
                    Some((Step::In(0), rest)) => referent_nodes.find(rest).map(|r| (p, r)),
                    _ => None,
                })
                .collect::<Vec<_>>()
        } else {
            (0..referent_nodes.len())
                .filter_map(|r| {
                    let mut path = vec![Step::In(0)];
                    path.extend(referent_nodes.piece(r).path.iter().copied());
                    nodes.find(&path).map(|p| (p, r))
                })
                .collect::<Vec<_>>()
        };
        if pairs.is_empty() {
            return Ok((Some(nodes), None, None));
        }
        pairs.sort();
        let pieces = ReferentPieces {
            nodes: vir::with_vcx(|vcx| &*vcx.alloc(referent_nodes)),
            casters: self
                .deps
                .require_ref::<IndCastersEnc>((normalized.param, normalized.concrete.ty))?,
            ty_args: self.deps.require_dep::<GArgsTyEnc>(normalized.concrete.args)?,
            generic_snap: self.deps.require_ref::<TyImpureEnc>(normalized.param)?.ref_to_snap,
            concrete_snap: self
                .deps
                .require_ref::<TyImpureEnc>(normalized.concrete.ty)?
                .ref_to_snap,
            param_ext: self.deps.require_ref::<TyIndirectEnc>(normalized.param)?.ext,
            pairs: vir::with_vcx(|vcx| vcx.alloc_slice(&pairs)),
        };
        Ok(match mentions_own {
            true => (Some(nodes), Some(pieces), None),
            false => (Some(nodes), None, Some(pieces)),
        })
    }

    fn encode_ty(
        &mut self,
        ty: TyData<'vir, (RustTyDatas, ImpureTyDatas)>,
    ) -> EncResult<'vir, TyData<'vir, UseImpureTyDatas>> {
        let specifics = match &ty.specifics {
            TySpecifics::Param(..) => TySpecifics::mk_param(()),
            TySpecifics::Opaque(..) => TySpecifics::mk_opaque(()),
            TySpecifics::Primitive(..) => TySpecifics::mk_primitive(()),
            TySpecifics::ImmRef(data) => {
                let referent_caster = self.encode_normalized(data.0.referent, ty.0.params)?;
                let metadata_caster = self.encode_normalized_pure(data.0.metadata, ty.0.params)?;
                TySpecifics::mk_immref(TyUseImpureImmRef {
                    referent_caster,
                    metadata_caster,
                    args: self.args_t,
                    impure: *data.1,
                })
            }
            TySpecifics::MutRef(data) => {
                let referent_caster = self.encode_normalized(data.0.referent, ty.0.params)?;
                let metadata_caster = self.encode_normalized_pure(data.0.metadata, ty.0.params)?;
                let (nodes, inner, referent) =
                    self.mutref_nodes(data.0.referent, ty.0.params)?;
                TySpecifics::mk_mutref(TyUseImpureMutRef {
                    referent_caster,
                    metadata_caster,
                    args: self.args_t,
                    impure: *data.1,
                    ref_to_snap: ty.1.ref_to_snap,
                    nodes,
                    inner,
                    referent,
                })
            }
            TySpecifics::Raw(data) => {
                let metadata_caster = self.encode_normalized_pure(data.0.metadata, ty.0.params)?;
                TySpecifics::mk_raw(TyUseImpureRaw {
                    metadata_caster,
                    args: self.args_t,
                    impure: *data.1,
                })
            }
            TySpecifics::ArrayLike(data) => {
                TySpecifics::ArrayLike(self.encode_array(data, ty.1.ref_to_pred, ty.0.params)?)
            }
            TySpecifics::StructLike(data) => {
                let box_data = match ty.0.special {
                    RustTySpecial::Box => Some(TyUseImpureBoxData {
                        metadata_caster: self
                            .encode_normalized_pure(ty.box_metadata_ty(), ty.0.params)?,
                        unique: self.deps.require_dep::<TyUseImpureEnc>(
                            ty.box_unique_ty().decompose_normalize(self.args),
                        )?,
                        alloc: self.deps.require_dep::<TyUseImpureEnc>(
                            ty.box_alloc_ty().decompose_normalize(self.args),
                        )?,
                    }),
                    RustTySpecial::None => None,
                };
                TySpecifics::StructLike(self.encode_structlike(
                    data,
                    ty.1.ref_to_pred,
                    ty.0.params,
                    box_data,
                )?)
            }
            TySpecifics::EnumLike(data) => {
                TySpecifics::EnumLike(self.encode_enumlike(data, ty.0.params)?)
            }
            TySpecifics::Builtin(..) => TySpecifics::mk_builtin(()),
        };
        let data = TyUseImpureData {
            args: self.args_t,
            impure: *ty.1,
            inhabited: self.inhabited,
        };
        Ok(TyData::new(data, specifics))
    }

    fn encode_normalized(
        &mut self,
        inner: LazyRustTy<'vir>,
        params: GParams<'vir>,
    ) -> EncResult<'vir, FieldCaster<'vir>> {
        let normalized = inner.decompose_compare_normalize(params, self.args);
        self.deps.require_dep::<GArgsCastEnc<Impure>>(normalized)
    }

    fn encode_normalized_pure(
        &mut self,
        inner: LazyRustTy<'vir>,
        params: GParams<'vir>,
    ) -> EncResult<'vir, GArgCaster<'vir, crate::encoders::Pure>> {
        let normalized = inner.decompose_compare_normalize(params, self.args);
        self.deps
            .require_dep::<GArgsCastEnc<crate::encoders::Pure>>(normalized)
    }

    fn encode_array(
        &mut self,
        data: &ArrayData<'vir, (RustTyDatas, ImpureTyDatas)>,
        _ref_to_pred: PredicateIdn<'vir, (vir::Ref, vir::ManyTyVal, vir::ManyCSnap)>,
        params: GParams<'vir>,
    ) -> EncResult<'vir, ArrayData<'vir, UseImpureTyDatas>> {
        let caster = self.encode_normalized(*data.0, params)?;
        let slice = data.slice;
        let data = TyUseImpureArrayData {
            args: self.args_t,
            impure: *data.data.1,
            element_caster: caster,
        };
        Ok(ArrayData::new(data, slice))
    }

    fn encode_structlike(
        &mut self,
        data: &StructData<'vir, (RustTyDatas, ImpureTyDatas)>,
        ref_to_pred: PredicateIdn<'vir, (vir::Ref, vir::ManyTyVal, vir::ManyCSnap)>,
        params: GParams<'vir>,
        box_data: Option<TyUseImpureBoxData<'vir>>,
    ) -> EncResult<'vir, StructData<'vir, UseImpureTyDatas>> {
        let fields = data
            .fields
            .iter()
            .map(|field| {
                let caster = self.encode_normalized(field.0.ty(), params)?;
                Ok(TyUseImpureField {
                    caster,
                    args: self.args_t,
                    impure: *field.1,
                })
            })
            .collect::<EncResult<'vir, Vec<_>>>()?;
        let data = TyUseImpureStructData {
            args: self.args_t,
            ref_to_pred,
            impure: *data.1,
            box_data,
        };
        Ok(StructData::new(data, fields))
    }

    fn encode_enumlike(
        &mut self,
        data: &EnumData<'vir, (RustTyDatas, ImpureTyDatas)>,
        params: GParams<'vir>,
    ) -> EncResult<'vir, EnumData<'vir, UseImpureTyDatas>> {
        let discr_ty = self
            .deps
            .require_dep::<TyPureEnc>(RustTyDecomposition::from_prim_ty(data.0.discr).ty)?;
        let discr_prim = *discr_ty.expect_primitive();
        let variant_inhabited = data
            .variants
            .iter()
            .map(|variant| {
                let fields = variant
                    .inner
                    .fields
                    .iter()
                    .map(|field| {
                        let field = field.0.ty().decompose_normalize(self.args);
                        Ok(self
                            .deps
                            .require_ref::<TyUseInhabitedEnc>(field)?
                            .inhabited())
                    })
                    .collect::<EncResult<'vir, Vec<_>>>()?;
                let inhabited = vir::with_vcx(|vcx| vcx.mk_conj(&fields));
                let discr = discr_prim
                    .prim_to_snap(discr_prim.expr_from_bits(data.0.discr, variant.0.discr_val));
                Ok((discr, inhabited))
            })
            .collect::<EncResult<'vir, Vec<_>>>()?;
        let variant_inhabited = vir::with_vcx(|vcx| vcx.alloc_slice(&variant_inhabited));

        let variants = data
            .variants
            .iter()
            .map(|variant| {
                let structlike =
                    self.encode_structlike(&variant.inner, variant.1.predicate, params, None)?;
                Ok(VariantData::new((), structlike))
            })
            .collect::<EncResult<'vir, Vec<_>>>()?;
        let data = TyUseImpureEnumData {
            args: self.args_t,
            impure: *data.1,
            variant_inhabited,
        };
        Ok(EnumData::new(data, variants))
    }
}

impl<'vir> TyUseImpureData<'vir> {
    /// Generates a call to `method_assign`, which asserts that the snapshot of
    /// `self_ref` is `self_new_snap`. Appropriate type arguments are used.
    pub fn apply_method_assign<'tcx>(
        &self,
        vcx: &'vir vir::VirCtxt<'tcx>,
        self_ref: vir::ExprRef<'vir>,
        self_new_snap: vir::ExprSnap<'vir>,
    ) -> vir::Stmt<'vir> {
        vcx.alloc(vir::StmtData::new(vcx.alloc((self.impure.method_assign)(
            self_ref,
            self.args.get_ty(),
            self.args.get_const(),
            self_new_snap,
        ))))
    }

    /// Constructs the Viper predicate application expression.
    pub fn ref_to_pred<'tcx>(
        &self,
        vcx: &'vir vir::VirCtxt<'tcx>,
        self_ref: vir::ExprRef<'vir>,
        perm: Option<vir::ExprPerm<'vir>>,
    ) -> vir::ExprBool<'vir> {
        let inhabited = self.inhabited.inhabited();
        let predicate = vcx.mk_predicate_app_expr(self.ref_to_pred_app(self_ref, perm));
        vcx.mk_conj(&[inhabited, predicate])
    }

    /// Constructs the Viper predicate application.
    pub fn ref_to_pred_app(
        &self,
        self_ref: vir::ExprRef<'vir>,
        perm: Option<vir::ExprPerm<'vir>>,
    ) -> vir::PredicateApp<'vir> {
        (self.impure.ref_to_pred)(self_ref, self.args.get_ty(), self.args.get_const())(perm)
    }

    /// Calls the predicate (heap) dependent snapshot construction function.
    pub fn ref_to_snap<Curr, Next>(
        &self,
        self_ref: vir::ExprGenRef<'vir, Curr, Next>,
    ) -> vir::ExprGenSnap<'vir, Curr, Next> {
        self.impure.ref_to_snap.call()(self_ref, self.args.get_ty(), self.args.get_const())
    }

    pub fn snapshot(&self) -> vir::TypeSnap<'vir> {
        self.impure.ref_to_snap.result()
    }
}

impl<'vir> TyData<'vir, UseImpureTyDatas> {
    pub fn inhabited<Curr, Next>(&self) -> vir::ExprGenBool<'vir, Curr, Next> {
        self.data.inhabited.inhabited()
    }

    /// The predicates that `fold` consumes besides those of the places the
    /// predicate is expanded into (its fields, variant, element, or the value
    /// of a `Box`): the discriminant of an enum, the rest of an array whose
    /// element is folded back, and the other fields of a `Box`.
    pub fn fold_frame(
        &self,
        variant: Option<abi::VariantIdx>,
        self_ref: vir::ExprRef<'vir>,
        index: Option<vir::ExprInt<'vir>>,
    ) -> Vec<vir::ExprBool<'vir>> {
        if variant.is_some() {
            return Vec::new();
        }
        vir::with_vcx(|vcx| match &self.specifics {
            TySpecifics::EnumLike(data) => {
                let discr = data.discr(self_ref);
                vec![data.discr_ty().ref_to_pred(vcx, discr, None)]
            }
            TySpecifics::ArrayLike(array) => {
                let index = index.expect("cannot fold array type without index");
                vec![array.ref_to_index_frame_pred(self_ref, index)]
            }
            TySpecifics::StructLike(data) => match data.data.box_data {
                Some(box_data) => vec![
                    box_data
                        .unique
                        .ref_to_pred(vcx, data.fields[0].field_ref(self_ref), None),
                    box_data
                        .alloc
                        .ref_to_pred(vcx, data.fields[1].field_ref(self_ref), None),
                ],
                None => Vec::new(),
            },
            _ => Vec::new(),
        })
    }

    /// Fold the predicate (including generic casts).
    pub fn fold(
        &self,
        variant: Option<abi::VariantIdx>,
        self_ref: vir::ExprRef<'vir>,
        index: Option<vir::ExprInt<'vir>>,
        perm: Option<vir::ExprPerm<'vir>>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        if let Some(variant) = variant {
            return self
                .expect_variant(variant)
                .inner
                .fold(self_ref, perm)
                .collect();
        };
        match &self.specifics {
            TySpecifics::Param(_) | TySpecifics::Primitive(_) | TySpecifics::Builtin(_) => {
                unreachable!()
            }
            TySpecifics::Opaque(_) => panic!("cannot fold opaque type"),
            TySpecifics::ArrayLike(array) => {
                let index = index.expect("cannot fold array type without index");
                array
                    .element_caster
                    .cast_to_callee_ctx(array.ref_to_index_ref(self_ref, index))
                    .into_iter()
                    .chain([vir::with_vcx(|vcx| {
                        vcx.alloc(vir::StmtData::new(vcx.alloc(
                            (array.data.impure.method_fold)(
                                index,
                                self_ref,
                                self.args.get_ty(),
                                self.args.get_const(),
                            ),
                        )))
                    })])
                    .collect()
            }
            TySpecifics::ImmRef(..) | TySpecifics::Raw(..) => Vec::new(),
            TySpecifics::MutRef(data) => data.fold(self_ref, label),
            TySpecifics::StructLike(data) => data.fold(self_ref, perm).collect(),
            TySpecifics::EnumLike(..) => {
                let pred_app = self.ref_to_pred_app(self_ref, perm);
                vec![vir::with_vcx(|vcx| vcx.mk_fold_stmt(pred_app))]
            }
        }
    }

    /// Fold the predicate of a reference just created by a borrow `&mut p`:
    /// as [Self::fold], and `p`'s indirect pieces move into the reference.
    pub fn fold_borrow(&self, self_ref: vir::ExprRef<'vir>) -> Vec<vir::Stmt<'vir>> {
        match &self.specifics {
            TySpecifics::MutRef(data) => data.fold_borrow(self_ref, None),
            _ => self.fold(None, self_ref, None, None, None),
        }
    }

    /// Unfold the predicate of a reference whose borrow `&mut p` expires: as
    /// [Self::unfold], and the reference's inner pieces move back to `p`.
    pub fn unfold_borrow(
        &self,
        variant: Option<abi::VariantIdx>,
        self_ref: vir::ExprRef<'vir>,
        old: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        match &self.specifics {
            TySpecifics::MutRef(data) if variant.is_none() => data.unfold_borrow(self_ref, old),
            _ => self.unfold(variant, self_ref, None, None, old),
        }
    }

    /// Unfold the predicate (including generic casts).
    pub fn unfold(
        &self,
        variant: Option<abi::VariantIdx>,
        self_ref: vir::ExprRef<'vir>,
        index: Option<vir::ExprInt<'vir>>,
        perm: Option<vir::ExprPerm<'vir>>,
        old: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        if let Some(variant) = variant {
            return self
                .expect_variant(variant)
                .inner
                .unfold(self_ref, perm)
                .collect();
        };
        match &self.specifics {
            TySpecifics::Param(_) | TySpecifics::Primitive(_) | TySpecifics::Builtin(_) => {
                unreachable!()
            }
            TySpecifics::Opaque(_) => panic!("cannot unfold opaque type"),
            TySpecifics::ArrayLike(array) => {
                let index = index.expect("cannot unfold array type without index");
                [vir::with_vcx(|vcx| {
                    vcx.alloc(vir::StmtData::new(vcx.alloc(
                        (array.data.impure.method_unfold)(
                            index,
                            self_ref,
                            self.args.get_ty(),
                            self.args.get_const(),
                        ),
                    )))
                })]
                .into_iter()
                .chain(
                    array
                        .element_caster
                        .cast_to_caller_ctx(array.ref_to_index_ref(self_ref, index)),
                )
                .collect()
            }
            TySpecifics::ImmRef(..) | TySpecifics::Raw(..) => Vec::new(),
            TySpecifics::MutRef(data) => data.unfold(self_ref, old),
            TySpecifics::StructLike(data) => data.unfold(self_ref, perm).collect(),
            TySpecifics::EnumLike(data) => {
                let pred_app = self.ref_to_pred_app(self_ref, perm);
                vir::with_vcx(|vcx| {
                    let discr = data
                        .discr_ty()
                        .ref_to_snap(data.discr(self_ref))
                        .downcast_ty::<vir::CSnap>();
                    std::iter::once(vcx.mk_unfold_stmt(pred_app))
                        .chain(data.variant_inhabited.iter().map(
                            |(variant_discr, variant_inhabited)| {
                                let is_variant = vcx.mk_eq_expr(discr, *variant_discr);
                                let implication = vcx
                                    .mk_bin_op_expr(
                                        vir::BinOpKind::Implies,
                                        is_variant,
                                        *variant_inhabited,
                                    )
                                    .downcast_ty();
                                vcx.mk_inhale_stmt(implication)
                            },
                        ))
                        .collect()
                })
            }
        }
    }
}

impl<'vir> TyUseImpureArray<'vir> {
    /// Get the (Ref) address of an index. Identical to the function one would
    /// call in `use_pure`.
    pub fn ref_to_index_ref<Curr, Next>(
        &self,
        self_ref: vir::ExprGenRef<'vir, Curr, Next>,
        index: vir::ExprGenInt<'vir, Curr, Next>,
    ) -> vir::ExprGenRef<'vir, Curr, Next> {
        self.data.impure.ref_to_index_ref.call()(self_ref, index, self.args.get_ty())
    }

    /// The predicate of the array without its element at `index`.
    fn ref_to_index_frame_pred(
        &self,
        self_ref: vir::ExprRef<'vir>,
        index: vir::ExprInt<'vir>,
    ) -> vir::ExprBool<'vir> {
        let pred_app = (self.data.impure.index_predicate)(
            self_ref,
            index,
            self.args.get_ty(),
            self.args.get_const(),
        )(None);
        vir::with_vcx(|vcx| vcx.mk_predicate_app_expr(pred_app))
    }
}

impl<'vir> TyUseImpureStruct<'vir> {
    /// The pointer metadata of a `Box`, read out of the (folded) `Unique`
    /// field's predicate.
    #[track_caller]
    pub fn box_metadata(&self, self_ref: vir::ExprRef<'vir>) -> vir::ExprSnap<'vir> {
        let box_data = self.data.impure.box_data.expect("expected box");
        let metadata = box_data.metadata.call()(
            self_ref,
            self.data.args.get_ty(),
            self.data.args.get_const(),
        );
        self.data
            .box_data
            .expect("expected box")
            .metadata_caster
            .cast_to_caller_ctx(metadata.upcast_ty())
    }

    /// The (Ref) address of the boxed `T` (the box's last field), read out of
    /// the (folded) `Unique` field's predicate.
    #[track_caller]
    pub fn box_address(&self, self_ref: vir::ExprRef<'vir>) -> vir::ExprRef<'vir> {
        assert!(self.data.impure.box_data.is_some(), "expected box");
        self.fields.last().unwrap().field_ref(self_ref)
    }

    fn ref_to_pred_app(
        &self,
        self_ref: vir::ExprRef<'vir>,
        perm: Option<vir::ExprPerm<'vir>>,
    ) -> vir::PredicateApp<'vir> {
        (self.ref_to_pred)(self_ref, self.args.get_ty(), self.args.get_const())(perm)
    }

    /// Fold the predicate (including generic casts).
    fn fold(
        &self,
        self_ref: vir::ExprRef<'vir>,
        perm: Option<vir::ExprPerm<'vir>>,
    ) -> impl Iterator<Item = vir::Stmt<'vir>> + '_ {
        let pred_app = self.ref_to_pred_app(self_ref, perm);
        let fold = vir::with_vcx(|vcx| vcx.mk_fold_stmt(pred_app));
        self.cast_to_callee_ctx(self_ref).chain([fold])
    }

    /// Unfold the predicate (including generic casts).
    fn unfold(
        &self,
        self_ref: vir::ExprRef<'vir>,
        perm: Option<vir::ExprPerm<'vir>>,
    ) -> impl Iterator<Item = vir::Stmt<'vir>> + '_ {
        let pred_app = self.ref_to_pred_app(self_ref, perm);
        let unfold = vir::with_vcx(|vcx| vcx.mk_unfold_stmt(pred_app));
        [unfold]
            .into_iter()
            .chain(self.cast_to_caller_ctx(self_ref))
    }

    fn cast_to_caller_ctx(
        &self,
        self_ref: vir::ExprRef<'vir>,
    ) -> impl Iterator<Item = vir::Stmt<'vir>> {
        self.fields
            .iter()
            .filter_map(|f| f.cast_to_caller_ctx(self_ref))
    }

    fn cast_to_callee_ctx(
        &self,
        self_ref: vir::ExprRef<'vir>,
    ) -> impl Iterator<Item = vir::Stmt<'vir>> {
        self.fields
            .iter()
            .filter_map(|f| f.cast_to_callee_ctx(self_ref))
    }
}

impl<'vir> TyUseImpureField<'vir> {
    /// Get the (Ref) address of a field. Identical to the function one would
    /// call in `use_pure`.
    pub fn field_ref<Curr, Next>(
        &self,
        self_ref: vir::ExprGenRef<'vir, Curr, Next>,
    ) -> vir::ExprGenRef<'vir, Curr, Next> {
        self.impure.ref_to_field_ref.call()(self_ref, self.args.get_ty(), self.args.get_const())
    }

    fn cast_to_caller_ctx(&self, self_ref: vir::ExprRef<'vir>) -> Option<vir::Stmt<'vir>> {
        self.caster.cast_to_caller_ctx(self.field_ref(self_ref))
    }

    fn cast_to_callee_ctx(&self, self_ref: vir::ExprRef<'vir>) -> Option<vir::Stmt<'vir>> {
        self.caster.cast_to_callee_ctx(self.field_ref(self_ref))
    }
}

impl<'vir> TyUseImpureEnum<'vir> {
    pub fn discr(&self, self_ref: vir::ExprRef<'vir>) -> vir::ExprRef<'vir> {
        (self.impure.discr)(self_ref)
    }

    pub fn discr_ty(&self) -> TyUseImpure<'vir> {
        self.impure.discr_ty
    }
}

impl<'vir> TyUseImpureImmRef<'vir> {}

impl<'vir> TyUseImpureMutRef<'vir> {
    pub fn deref(
        &self,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> vir::ExprRef<'vir> {
        let snap = self.ref_to_snap.call()(self_ref, self.args.get_ty(), self.args.get_const())
            .downcast_ty();
        let deref = self.impure.pure.deref_access.call()(snap);
        vir::with_vcx(|vcx| vcx.maybe_apply_label(deref, label))
    }

    pub fn prim_to_snap_assign(
        &self,
        self_ref: vir::ExprRef<'vir>,
        metadata: vir::ExprSnap<'vir>,
    ) -> vir::ExprCSnap<'vir> {
        let metadata = self.metadata_caster.cast_to_callee_ctx(metadata);
        (self.impure.arbitrary_value)(
            self_ref,
            metadata.downcast_ty(),
            self.args.get_ty(),
            self.args.get_const(),
        )
    }

    /// The indirect piece of the reference's own lifetime itself, which
    /// holds the referent: `p_Ref_mutable_ind(S, U, q_here(0))` for the
    /// snapshot `S` of the reference (its key, as nothing encloses it).
    fn own_piece(
        &self,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Option<vir::PredicateApp<'vir>> {
        let nodes = self.nodes?;
        let own = nodes.find(&[Step::Here(0)])?;
        Some(nodes.app(own, self.snap(self_ref, label), None))
    }

    /// The snapshot of the reference (at `label`).
    fn snap(
        &self,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> vir::ExprSnap<'vir> {
        let snap = self.ref_to_snap.call()(self_ref, self.args.get_ty(), self.args.get_const());
        vir::with_vcx(|vcx| vcx.maybe_apply_label(snap, label))
    }

    /// Fold the reference's pieces holding `pieces`, enclosing ones first
    /// (their keys read the own piece).
    fn fold_held(
        &self,
        pieces: &ReferentPieces<'vir>,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        let nodes = self.nodes.unwrap();
        let snap = self.snap(self_ref, label);
        pieces
            .pairs
            .iter()
            .map(|(p, _)| {
                let piece = nodes.app(*p, nodes.key(*p, snap), None);
                vir::with_vcx(|vcx| vcx.mk_fold_stmt(piece))
            })
            .collect()
    }

    /// Unfold the reference's pieces holding `pieces`, enclosed ones first,
    /// while the own piece (from which their keys are read) is folded. For a
    /// reference that is no longer current (e.g. moved into a call), the whole
    /// key is that of its label.
    fn unfold_held(
        &self,
        pieces: &ReferentPieces<'vir>,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        let nodes = self.nodes.unwrap();
        let snap = self.ref_to_snap.call()(self_ref, self.args.get_ty(), self.args.get_const());
        pieces
            .pairs
            .iter()
            .rev()
            .map(|(p, _)| {
                vir::with_vcx(|vcx| {
                    let key = vcx.maybe_apply_label(nodes.key(*p, snap), label);
                    vcx.mk_unfold_stmt(nodes.app(*p, key, None))
                })
            })
            .collect()
    }

    /// For a reference at `label`: the generic keys of the held pieces, i.e.
    /// the value slots of the labelled keys `unfold_held` used (the referent's
    /// own predicate is inside the own piece at the label, so its snapshot
    /// cannot be read there).
    fn held_slots(
        &self,
        pieces: &ReferentPieces<'vir>,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Option<Vec<vir::ExprSnap<'vir>>> {
        label?;
        let nodes = self.nodes.unwrap();
        let snap = self.ref_to_snap.call()(self_ref, self.args.get_ty(), self.args.get_const());
        Some(
            pieces
                .pairs
                .iter()
                .map(|(p, _)| {
                    let key = vir::with_vcx(|vcx| vcx.maybe_apply_label(nodes.key(*p, snap), label));
                    self.impure.pure.value_access.call()(key.downcast_ty()).upcast_ty()
                })
                .collect(),
        )
    }

    /// A borrow `self_ref = &mut p` is created: as [Self::fold], and the
    /// pieces of `p`'s lifetimes move into the reference's inner lifetimes
    /// (occurrence `o` of `p`'s type becomes `In(0) ++ o`).
    fn fold_borrow(
        &self,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        let Some(referent) = self.referent else {
            return self.fold(self_ref, label);
        };
        let addr = self.deref(self_ref, label);
        // while `p` is concrete
        let mut stmts = referent.make_generic(addr);
        stmts.extend(self.fold(self_ref, label));
        stmts.extend(self.fold_held(&referent, self_ref, label));
        stmts
    }

    /// The borrow `self_ref = &mut p` expires: the inner pieces move back to
    /// `p`'s own lifetimes, then as [Self::unfold].
    fn unfold_borrow(
        &self,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        let Some(referent) = self.referent else {
            return self.unfold(self_ref, label);
        };
        // A callee that does not take the inner pieces (a generic
        // `RegionsIn(T)` position) leaves them keyed by their value before
        // the call.
        let mut stmts = self.unfold_held(&referent, self_ref, label);
        let addr = self.deref(self_ref, label);
        let mut own = self.unfold(self_ref, label);
        // between unfolding the own piece and the referent's `make_concrete`
        let make_concrete = own.pop();
        stmts.extend(own);
        // while `p` is generic
        let slots = self.held_slots(&referent, self_ref, label);
        stmts.extend(referent.make_concrete(addr, &self.args.get_ty()[..1], slots));
        stmts.extend(make_concrete);
        stmts
    }

    /// Give the referent back to the reference: cast it to the generic
    /// representation and fold the reference's own pieces.
    fn fold(
        &self,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        let addr = self.deref(self_ref, label);
        let mut stmts = Vec::new();
        // the inner pieces first, while the concrete referent still exists
        if let Some(inner) = self.inner {
            stmts.extend(inner.make_generic(addr));
        }
        stmts.extend(self.referent_caster.cast_to_callee_ctx(addr));
        if let Some(piece) = self.own_piece(self_ref, label) {
            stmts.push(vir::with_vcx(|vcx| vcx.mk_fold_stmt(piece)));
        }
        if let Some(inner) = self.inner {
            stmts.extend(self.fold_held(&inner, self_ref, label));
        }
        stmts
    }

    /// Take the referent out of the reference: unfold the reference's own
    /// pieces and cast the referent to its concrete representation.
    fn unfold(
        &self,
        self_ref: vir::ExprRef<'vir>,
        label: Option<vir::OldLabel<'vir>>,
    ) -> Vec<vir::Stmt<'vir>> {
        let addr = self.deref(self_ref, label);
        let mut stmts = Vec::new();
        if let Some(inner) = self.inner {
            stmts.extend(self.unfold_held(&inner, self_ref, label));
        }
        if let Some(piece) = self.own_piece(self_ref, label) {
            stmts.push(vir::with_vcx(|vcx| vcx.mk_unfold_stmt(piece)));
        }
        // the inner pieces, while the generic referent still exists
        if let Some(inner) = self.inner {
            let slots = self.held_slots(&inner, self_ref, label);
            stmts.extend(inner.make_concrete(addr, &self.args.get_ty()[..1], slots));
        }
        stmts.extend(self.referent_caster.cast_to_caller_ctx(addr));
        stmts
    }
}
