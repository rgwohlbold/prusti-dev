use prusti_rustc_interface::{abi, middle::ty};
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CastType, FunctionIdn, PredicateIdn, Reify};

use crate::encoders::TyUseImpureEnc;

use super::{
    RustTy, RustTyDecomposition, RustTySpecial, TySpecifics,
    arity::TyArityEnc,
    data::StructData,
    generics::{GArgsTyEnc, GParams, GenericParamsEnc},
    occurrences::{End, Occurrence, Step, occurrences, param_position},
    pure::{PureTyDatas, TyPureEnc},
    range::{Range, int, pos_expr},
    rust_ty::RustTyDatas,
};

/// Encodes for a type constructor `X`:
/// - the indirect predicate `p_X_ind(key: s_X, T.., lo: Int, hi: Int)`: what
///   `X` reaches through the references of its lifetime positions `lo .. hi`
///   (see `occurrences`). Its body holds, per field and per occurrence of a
///   parameter of `X` in the field's type, the field's piece for the part of
///   the range at that occurrence, keyed by the field's key extended by the
///   earlier occurrences in the range.
/// - the key extension `ext_X(K: s_X, T.., Kp: s_X, lo: Int, hi: Int): s_X`:
///   the key `K` with the cells owned by the piece `p_X_ind(Kp, .., lo, hi)`
///   read in (`K` itself for an empty range). `K` supplies the values, `Kp`
///   only the permission; both must agree on the discriminants that `ext_X`
///   branches on (a precondition).
/// - for `Param` only: the trusted range algebra `split_Param`/`merge_Param`
///   and `ind_empty` (see `ParamRanges`).
pub struct TyIndirectEnc;

pub type TyIndirectPred<'vir> =
    PredicateIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::Int, vir::Int)>;

pub type TyIndirectExt<'vir> = FunctionIdn<
    'vir,
    (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::Snap, vir::Int, vir::Int),
    vir::Snap,
>;

pub type TyIndirectEmpty<'vir> =
    vir::MethodIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::Int, vir::Int)>;

/// `split_Param`/`merge_Param(K, T, lo, m, hi)`
pub type TyIndirectSplit<'vir> =
    vir::MethodIdn<'vir, (vir::Snap, vir::ManyTyVal, vir::ManyCSnap, vir::Int, vir::Int, vir::Int)>;

/// The trusted methods of `p_Param_ind`:
/// ```text
/// method ind_empty(k, T, lo, hi)     requires !(lo < hi)  ensures p_Param_ind(k, T, lo, hi)
/// method split_Param(K, T, lo, m, hi)
///   requires lo <= m && m <= hi && p_Param_ind(K, T, lo, hi)
///   ensures  p_Param_ind(K, T, lo, m) && p_Param_ind(E, T, m, hi)
///   ensures  ext_Param(E, T, E, m, hi) == old(ext_Param(K, T, K, lo, hi))
/// method merge_Param(K, T, lo, m, hi)   // the inverse
/// ```
/// where `E = ext_Param(K, T, K, lo, m)`.
#[derive(Debug, Clone, Copy)]
pub struct ParamRanges<'vir> {
    pub empty: TyIndirectEmpty<'vir>,
    pub split: TyIndirectSplit<'vir>,
    pub merge: TyIndirectSplit<'vir>,
}

#[derive(Debug, Clone, Copy)]
pub struct TyIndirectRef<'vir> {
    pub pred: TyIndirectPred<'vir>,
    pub ext: TyIndirectExt<'vir>,
    /// For `Param`: its trusted methods.
    pub param: Option<ParamRanges<'vir>>,
}

impl<'vir> task_encoder::OutputRefAny for TyIndirectRef<'vir> {}

/// The discriminant agreement of two keys `(K, Kp)` that `ext_X(K, .., Kp, ..)`
/// requires.
pub(crate) type Agree<'vir> =
    vir::ExprGenBool<'vir, (vir::ExprSnap<'vir>, vir::ExprSnap<'vir>), vir::ExprKind<'vir>>;

#[derive(Clone)]
pub struct TyIndirectDep<'vir> {
    /// `None` if `ext_X` requires no agreement.
    pub(crate) agree: Option<Agree<'vir>>,
    /// Per type parameter `T_j`: where `T_j` occurs in the fields.
    pub(crate) ty_param_parts: Vec<Vec<TyParamPart<'vir>>>,
}

/// An occurrence of a type parameter `T_j` in a field: the body holds the
/// piece `p_F_ind(read(key), ..)` of `T_j`'s part of the range there (for an
/// enum only in `vid`), `p_Param_ind(read(key), T_j, ..)` if the field is
/// `T_j` itself (`path` empty).
#[derive(Clone, Debug)]
pub struct TyParamPart<'vir> {
    /// For an enum: the variant.
    pub vid: Option<abi::VariantIdx>,
    /// The field (`0` for the referent of a reference).
    pub field_idx: usize,
    pub read: PartRead<'vir>,
    /// The path to `T_j` inside the field's type.
    pub path: Vec<Step>,
}

/// How a part's key is read from the constructor's key.
#[derive(Clone, Copy, Debug)]
pub enum PartRead<'vir> {
    Field(FieldRead<'vir>),
    /// The value slot of a reference.
    RefSlot(vir::AdtDestructor<'vir, vir::CSnap, vir::PSnap>),
}

impl<'vir> PartRead<'vir> {
    pub fn read(&self, key: vir::ExprSnap<'vir>) -> vir::ExprSnap<'vir> {
        match self {
            PartRead::Field(read) => read.call()(key.downcast_ty()),
            PartRead::RefSlot(slot) => slot.call()(key.downcast_ty()).upcast_ty(),
        }
    }
}

#[derive(Clone)]
pub struct TyIndirectLocal<'vir> {
    predicate: vir::Predicate<'vir>,
    ext: vir::Function<'vir>,
    /// `ind_empty`, `split_Param`, `merge_Param`
    param: Vec<vir::Method<'vir>>,
}

type EncResult<'vir, T> = Result<T, EncodeFullError<'vir, TyIndirectEnc>>;

impl TaskEncoder for TyIndirectEnc {
    task_encoder::encoder_cache!(TyIndirectEnc);
    const ENCODER_NAME: &'static str = "indirect predicate encoder";

    type TaskDescription<'vir> = RustTy<'vir>;
    type OutputRef<'vir> = TyIndirectRef<'vir>;
    type OutputFullLocal<'vir> = TyIndirectLocal<'vir>;
    type OutputFullDependency<'vir> = TyIndirectDep<'vir>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        let ty = *task_key;
        let snap = deps.require_dep::<TyPureEnc>(ty)?;
        let params = deps.require_dep::<GenericParamsEnc>(ty.params)?;
        vir::with_vcx(|vcx| {
            let pred = PredicateIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "p_{}_ind", ty.name())),
                (
                    snap.snapshot,
                    params.ty_args(),
                    params.const_args(),
                    vir::TYPE_INT,
                    vir::TYPE_INT,
                ),
            );
            let ext_idn = FunctionIdn::new(
                vir::ViperIdent::new(vir::vir_format!(vcx, "ext_{}", ty.name())),
                (
                    snap.snapshot,
                    params.ty_args(),
                    params.const_args(),
                    snap.snapshot,
                    vir::TYPE_INT,
                    vir::TYPE_INT,
                ),
                snap.snapshot,
            );
            let is_param = matches!(ty.specifics, TySpecifics::Param(_)) && params.ty_exprs().len() == 1;
            let range_args = (snap.snapshot, params.ty_args(), params.const_args(), vir::TYPE_INT, vir::TYPE_INT);
            let split_args = (
                snap.snapshot,
                params.ty_args(),
                params.const_args(),
                vir::TYPE_INT,
                vir::TYPE_INT,
                vir::TYPE_INT,
            );
            let param_ranges = is_param.then(|| ParamRanges {
                empty: vir::MethodIdn::new(vir::ViperIdent::new("ind_empty"), range_args),
                split: vir::MethodIdn::new(vir::ViperIdent::new("split_Param"), split_args),
                merge: vir::MethodIdn::new(vir::ViperIdent::new("merge_Param"), split_args),
            });
            deps.emit_output_ref(ty, TyIndirectRef { pred, ext: ext_idn, param: param_ranges })?;

            let key_decl = vcx.mk_local_decl("key", snap.snapshot);
            let lo_decl = vcx.mk_local_decl("lo", vir::TYPE_INT);
            let hi_decl = vcx.mk_local_decl("hi", vir::TYPE_INT);
            let range = Range { lo: vcx.mk_local_ex(lo_decl), hi: vcx.mk_local_ex(hi_decl) };
            let mut builder = IndirectBuilder::new(vcx, deps, ty, vcx.alloc_slice(params.ty_exprs()), range);
            let key = vcx.mk_local_ex(key_decl);
            let combined = ty.zip(snap);
            let body = match &combined.specifics {
                TySpecifics::MutRef((rust, pure)) => {
                    let key = key.downcast_ty::<vir::CSnap>();
                    let referent = rust.referent.decompose(ty.params);
                    let referent_impure = builder.deps.require_dep::<TyUseImpureEnc>(referent)?;
                    let addr = pure.deref_access.call()(key);
                    // Position 0, the reference itself, owns the referent; the
                    // referent's positions are keyed by its snapshot if position
                    // 0 is in the range, by the value slot otherwise.
                    let own_in = range.contains(int(0));
                    let own = referent_impure.ref_to_pred(vcx, addr, None);
                    let owned = referent_impure.ref_to_snap(addr);
                    let slot: vir::ExprSnap<'vir> = pure.value_access.call()(key).upcast_ty();
                    let inner_key = vir::expr! { vcx; ([own_in]) ? ([owned]) : ([slot]) };
                    let inner_range = builder.referent_range();
                    let inner = builder.apply(referent, inner_key, inner_range)?;
                    let inner_nonempty = inner_range.nonempty();
                    Some(vir::expr! { vcx;
                        (([own_in]) ==> ([own])) && (([inner_nonempty]) ==> ([inner]))
                    })
                }
                TySpecifics::StructLike(data) => {
                    let fields = builder.fields(key.downcast_ty::<vir::CSnap>(), data)?;
                    Some(vcx.mk_conj(&fields))
                }
                TySpecifics::EnumLike(data) => {
                    let key = key.downcast_ty::<vir::CSnap>();
                    let discr = data.data.1.snap_to_discr_snap.call()(key);
                    let mut variants = Vec::new();
                    for variant in &data.variants {
                        let fields = builder.fields(key, &variant.inner)?;
                        if !fields.is_empty() {
                            let fields = vcx.mk_conj(&fields);
                            variants.push(vir::expr! { vcx;
                                (([discr]) == ([variant.data.1.discr])) ==> ([fields])
                            });
                        }
                    }
                    Some(vcx.mk_conj(&variants))
                }
                // Nothing behind these (an opaque type that is not trusted is
                // e.g. a function item or `str`). TODO: shared references and
                // the elements of arrays/slices (see `occurrences`).
                TySpecifics::Primitive(_)
                | TySpecifics::Raw(_)
                | TySpecifics::Builtin(_)
                | TySpecifics::ImmRef(_)
                | TySpecifics::ArrayLike(_) => Some(vcx.mk_bool::<true>()),
                TySpecifics::Opaque(_) if ty.special != RustTySpecial::Trusted => {
                    Some(vcx.mk_bool::<true>())
                }
                // Abstract: the instantiation (or the trusted type) is unknown.
                TySpecifics::Param(_) | TySpecifics::Opaque(_) => None,
            };
            let predicate = vcx.mk_predicate(
                pred,
                (key_decl, params.ty_decls(), params.const_decls(), lo_decl, hi_decl),
                body,
            );

            // ext_X
            let k_decl = vcx.mk_local_decl("K", snap.snapshot);
            let kp_decl = vcx.mk_local_decl("Kp", snap.snapshot);
            let k = vcx.mk_local_ex(k_decl);
            let kp = vcx.mk_local_ex(kp_decl);
            // `acc(p_X_ind(Kp, .., lo, hi), wildcard)`
            let parent = || {
                pred(kp, params.ty_exprs(), params.const_exprs(), range.lo, range.hi)(Some(
                    vcx.mk_wildcard(),
                ))
            };
            let nonempty = range.nonempty();
            // `K` for an empty range, which needs no permission
            let unfolding = |expr: vir::ExprSnap<'vir>| {
                let unfolded = vcx.mk_unfolding_expr(parent(), expr);
                vir::expr! { vcx; ([nonempty]) ? ([unfolded]) : ([k]) }
            };
            let parent_app = vcx.mk_predicate_app_expr(parent());
            let mut pres = vec![vir::expr! { vcx; ([nonempty]) ==> ([parent_app]) }];
            let (body, agree) = match &combined.specifics {
                TySpecifics::MutRef((rust, pure)) => {
                    let (k, kp) = (k.downcast_ty::<vir::CSnap>(), kp.downcast_ty::<vir::CSnap>());
                    let referent = rust.referent.decompose(ty.params);
                    let referent_impure = builder.deps.require_dep::<TyUseImpureEnc>(referent)?;
                    // With position 0, the value slot becomes the snapshot of the
                    // referent the parent owns, extended inside; otherwise the
                    // slot is extended.
                    let own_in = range.contains(int(0));
                    let owned = referent_impure.ref_to_snap(pure.deref_access.call()(kp));
                    let inner_range = builder.referent_range();
                    let owned_ext = builder.ext(referent, owned, owned, inner_range)?;
                    let slot_ext = builder.ext(
                        referent,
                        pure.value_access.call()(k).upcast_ty(),
                        pure.value_access.call()(kp).upcast_ty(),
                        inner_range,
                    )?;
                    let value = vir::expr! { vcx; ([own_in]) ? ([owned_ext]) : ([slot_ext]) };
                    let cons = (pure.prim_to_snap)(
                        pure.deref_access.call()(k),
                        pure.metadata_access.call()(k),
                        value.downcast_ty::<vir::PSnap>(),
                    );
                    (Some(unfolding(cons.upcast_ty())), None)
                }
                TySpecifics::StructLike(data) => {
                    let (k, kp) = (k.downcast_ty::<vir::CSnap>(), kp.downcast_ty::<vir::CSnap>());
                    let fields = builder.ext_fields(k, kp, data)?;
                    let body = match fields {
                        Some(fields) => {
                            let cons = (data.data.1.field_snaps_to_snap)(fields.as_slice());
                            unfolding(cons.upcast_ty())
                        }
                        None => k.upcast_ty(),
                    };
                    let fields = builder.agree_fields(data)?;
                    let agree = (!fields.is_empty()).then(|| {
                        vcx.mk_lazy_expr(
                            "agree_struct",
                            vir::TYPE_BOOL,
                            Box::new(move |vcx, (k, kp): (vir::ExprSnap<'vir>, vir::ExprSnap<'vir>)| {
                                agree_fields(vcx, &fields, k, kp).kind
                            }),
                        )
                    });
                    (Some(body), agree)
                }
                TySpecifics::EnumLike(data) => {
                    let (k, kp) = (k.downcast_ty::<vir::CSnap>(), kp.downcast_ty::<vir::CSnap>());
                    let snap_to_discr = data.data.1.snap_to_discr_snap;
                    let discr = snap_to_discr.call()(k);
                    let mut body = k.upcast_ty();
                    let mut variants = Vec::new();
                    for variant in data.variants.iter().rev() {
                        let Some(fields) = builder.ext_fields(k, kp, &variant.inner)? else {
                            continue;
                        };
                        let cons = (variant.inner.data.1.field_snaps_to_snap)(fields.as_slice()).upcast_ty();
                        body = vir::expr! { vcx;
                            (([discr]) == ([variant.data.1.discr])) ? ([cons]) : ([body])
                        };
                        variants.push((
                            variant.data.1.discr,
                            builder.agree_fields(&variant.inner)?,
                        ));
                    }
                    // Agreement is needed iff `ext_X` branches.
                    let branches = !variants.is_empty();
                    let agree = branches.then(|| {
                        vcx.mk_lazy_expr(
                            "agree_enum",
                            vir::TYPE_BOOL,
                            Box::new(move |vcx, (k, kp): (vir::ExprSnap<'vir>, vir::ExprSnap<'vir>)| {
                                let k_discr = snap_to_discr.call()(k.downcast_ty());
                                let kp_discr = snap_to_discr.call()(kp.downcast_ty());
                                let mut conjuncts = vec![vir::expr! { vcx; ([k_discr]) == ([kp_discr]) }];
                                for (discr, fields) in &variants {
                                    if !fields.is_empty() {
                                        let fields = agree_fields(vcx, fields, k, kp);
                                        conjuncts.push(vir::expr! { vcx;
                                            (([k_discr]) == ([*discr])) ==> ([fields])
                                        });
                                    }
                                }
                                vcx.mk_conj(&conjuncts).kind
                            }),
                        )
                    });
                    let body = if branches { unfolding(body) } else { body };
                    (Some(body), agree)
                }
                TySpecifics::Primitive(_)
                | TySpecifics::Raw(_)
                | TySpecifics::Builtin(_)
                | TySpecifics::ImmRef(_)
                | TySpecifics::ArrayLike(_) => (Some(k), None),
                TySpecifics::Opaque(_) if ty.special != RustTySpecial::Trusted => (Some(k), None),
                TySpecifics::Param(_) | TySpecifics::Opaque(_) => (None, None),
            };
            let mut posts = Vec::new();
            if let Some(agree) = agree {
                pres.push(agree.reify(vcx, (k, kp)));
                // `ext_X` keeps the discriminants (so that chained extensions
                // agree again)
                posts.push(agree.reify(vcx, (vcx.mk_result(snap.snapshot), k)));
            }
            if body.is_none() {
                // an empty range extends nothing
                let result = vcx.mk_result(snap.snapshot);
                posts.push(vir::expr! { vcx; (!([nonempty])) ==> (([result]) == ([k])) });
            }
            let ext = vcx.mk_function(
                ext_idn,
                (k_decl, params.ty_decls(), params.const_decls(), kp_decl, lo_decl, hi_decl),
                vcx.alloc_slice(&pres),
                vcx.alloc_slice(&posts),
                None,
                body,
            );
            let mut ty_param_parts = vec![Vec::new(); params.ty_exprs().len()];
            match &combined.specifics {
                TySpecifics::MutRef((_, pure)) => ty_param_parts[0].push(TyParamPart {
                    vid: None,
                    field_idx: 0,
                    read: PartRead::RefSlot(pure.value_access),
                    path: Vec::new(),
                }),
                TySpecifics::StructLike(data) => builder.collect_parts(data, None, &mut ty_param_parts),
                TySpecifics::EnumLike(data) => {
                    for variant in &data.variants {
                        let vid = Some(variant.data.0.vid);
                        builder.collect_parts(&variant.inner, vid, &mut ty_param_parts);
                    }
                }
                _ => (),
            }
            let param = match param_ranges {
                Some(ranges) => param_methods(vcx, ranges, pred, ext_idn, snap.snapshot, &params),
                None => Vec::new(),
            };
            Ok((TyIndirectLocal { predicate, ext, param }, TyIndirectDep { agree, ty_param_parts }))
        })
    }

    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        for output in Self::all_outputs_local_no_errors(program) {
            program.add_predicate(output.predicate);
            program.add_function(output.ext);
            for method in output.param {
                program.add_method(method);
            }
        }
    }
}

/// `ind_empty`, `split_Param` and `merge_Param` (see `ParamRanges`).
fn param_methods<'vir>(
    vcx: &'vir vir::VirCtxt<'vir>,
    ranges: ParamRanges<'vir>,
    pred: TyIndirectPred<'vir>,
    ext: TyIndirectExt<'vir>,
    snap: vir::TypeSnap<'vir>,
    params: &crate::encoders::ty::generics::GenericParams<'vir>,
) -> Vec<vir::Method<'vir>> {
    let tys = params.ty_exprs();
    let consts = params.const_exprs();
    let k_decl = vcx.mk_local_decl("K", snap);
    let lo_decl = vcx.mk_local_decl("lo", vir::TYPE_INT);
    let m_decl = vcx.mk_local_decl("m", vir::TYPE_INT);
    let hi_decl = vcx.mk_local_decl("hi", vir::TYPE_INT);
    let k = vcx.mk_local_ex(k_decl);
    let [lo, m, hi] = [lo_decl, m_decl, hi_decl].map(|decl| vcx.mk_local_ex(decl));
    let piece = |key: vir::ExprSnap<'vir>, lo: vir::ExprInt<'vir>, hi: vir::ExprInt<'vir>| {
        vir::expr! { vcx; acc([pred](key, [..[tys]], [..[consts]], lo, hi)) }
    };
    let ext_of = |key: vir::ExprSnap<'vir>, lo: vir::ExprInt<'vir>, hi: vir::ExprInt<'vir>| {
        ext(key, tys, consts, key, lo, hi)
    };
    let empty = vcx.mk_method(
        ranges.empty,
        (k_decl, params.ty_decls(), params.const_decls(), lo_decl, hi_decl),
        &[],
        vcx.alloc_slice(&[vir::expr! { vcx; !(([lo]) < ([hi])) }]),
        vcx.alloc_slice(&[piece(k, lo, hi)]),
        None,
    );
    let ordered = vir::expr! { vcx; (([lo]) <= ([m])) && (([m]) <= ([hi])) };
    // `[lo, hi)` on `K` is `[lo, m)` on `K` and `[m, hi)` on `K` extended by `[lo, m)`
    let (whole, whole_ext) = (piece(k, lo, hi), ext_of(k, lo, hi));
    let upper_key = ext_of(k, lo, m);
    let (lower, upper, upper_ext) = (piece(k, lo, m), piece(upper_key, m, hi), ext_of(upper_key, m, hi));
    let decls = (k_decl, params.ty_decls(), params.const_decls(), lo_decl, m_decl, hi_decl);
    let split = vcx.mk_method(
        ranges.split,
        decls,
        &[],
        vcx.alloc_slice(&[ordered, whole]),
        vcx.alloc_slice(&[lower, upper, vir::expr! { vcx; ([upper_ext]) == (old([whole_ext])) }]),
        None,
    );
    let merge = vcx.mk_method(
        ranges.merge,
        decls,
        &[],
        vcx.alloc_slice(&[ordered, lower, upper]),
        vcx.alloc_slice(&[whole, vir::expr! { vcx; ([whole_ext]) == (old([upper_ext])) }]),
        None,
    );
    vec![empty, split, merge]
}

/// A generic parameter of the constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Param {
    Lifetime(usize),
    Ty(usize),
}

/// The occurrences of the constructor's parameters in the type of a field.
struct FieldOccurrences<'vir> {
    ty: RustTyDecomposition<'vir>,
    occurrences: Vec<Occurrence<'vir>>,
    /// Per occurrence: the parameter it is, `None` for others (e.g. `'static`).
    params: Vec<Option<Param>>,
}

struct IndirectBuilder<'a, 'vir> {
    vcx: &'vir vir::VirCtxt<'vir>,
    deps: &'a mut TaskEncoderDependencies<'vir, TyIndirectEnc>,
    /// The constructor's generic parameters.
    params: GParams<'vir>,
    /// The constructor's type parameters (as Viper type values).
    ty_args: &'vir [vir::ExprTyVal<'vir>],
    /// The constructor's range `lo .. hi`.
    range: Range<'vir>,
    /// Per tracked parameter: its first position and width.
    offsets: Vec<(Param, vir::ExprInt<'vir>, vir::ExprInt<'vir>)>,
}

impl<'a, 'vir> IndirectBuilder<'a, 'vir> {
    fn new(
        vcx: &'vir vir::VirCtxt<'vir>,
        deps: &'a mut TaskEncoderDependencies<'vir, TyIndirectEnc>,
        ty: RustTy<'vir>,
        ty_args: &'vir [vir::ExprTyVal<'vir>],
        range: Range<'vir>,
    ) -> Self {
        let mut builder = Self {
            vcx,
            deps,
            params: ty.params,
            ty_args,
            range,
            offsets: Vec::new(),
        };
        // the positions of the identity type `X<'a.., T..>`
        for occurrence in occurrences(RustTyDecomposition::identity(ty)).unwrap_or_default() {
            let (param, width) = match (occurrence.end, occurrence.path.as_slice()) {
                (End::Region(_), [Step::Here(i)]) => (Param::Lifetime(*i), int(1)),
                (End::Param(_), [Step::In(j)]) => (Param::Ty(*j), TyArityEnc::arity(ty_args[*j])),
                _ => continue,
            };
            let offset = builder.pos(&occurrence.pos);
            builder.offsets.push((param, offset, width));
        }
        builder
    }

    /// A position in terms of the constructor's type parameters.
    fn pos(&self, pos: &super::occurrences::Pos) -> vir::ExprInt<'vir> {
        pos_expr(pos, |param| {
            self.ty_args[param_position(self.params, param.index, |arg| arg.as_type().is_some())]
        })
    }

    /// The part of the range inside the referent of `&'a mut T` (positions
    /// `1 .. 1 + arity(T)`), in the referent's positions.
    fn referent_range(&self) -> Range<'vir> {
        self.range.clamp(int(1), TyArityEnc::arity(self.ty_args[0]))
    }

    /// The part of the range at occurrence `idx` of a field (in the field
    /// type's positions; empty if disjoint), and whether it is nonempty.
    fn occurrence_range(
        &self,
        field: &FieldOccurrences<'vir>,
        idx: usize,
    ) -> Option<(Range<'vir>, vir::ExprBool<'vir>)> {
        let param = field.params[idx]?;
        let (_, offset, width) = self.offsets.iter().find(|(p, ..)| *p == param)?;
        let part = self.range.clamp(*offset, *width);
        Some((part.shift(self.pos(&field.occurrences[idx].pos)), part.nonempty()))
    }

    /// Per occurrence of a parameter in the fields: `guard ==> piece`.
    fn fields(
        &mut self,
        key: vir::ExprCSnap<'vir>,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
    ) -> EncResult<'vir, Vec<vir::ExprBool<'vir>>> {
        let mut pieces = Vec::new();
        for (rust, pure) in &data.fields {
            let field = self.field_occurrences(rust.ty().decompose(self.params));
            let keys = self.occurrence_keys(&field, pure.read.call()(key))?;
            for (idx, key) in keys.into_iter().enumerate() {
                let (Some(key), Some(range)) = (key, self.occurrence_range(&field, idx)) else {
                    continue;
                };
                let piece = self.apply(field.ty, key, range.0)?;
                let vcx = self.vcx;
                pieces.push(vir::expr! { vcx; ([range.1]) ==> ([piece]) });
            }
        }
        Ok(pieces)
    }

    /// The keys of the occurrences in a field with key `key` (`None` for an
    /// occurrence of no parameter): extended by the earlier occurrences in the
    /// range (the lower positions outside it are in `key` already).
    fn occurrence_keys(
        &mut self,
        field: &FieldOccurrences<'vir>,
        key: vir::ExprSnap<'vir>,
    ) -> EncResult<'vir, Vec<Option<vir::ExprSnap<'vir>>>> {
        let mut keys: Vec<Option<vir::ExprSnap<'vir>>> = Vec::new();
        for idx in 0..field.occurrences.len() {
            if field.params[idx].is_none() {
                keys.push(None);
                continue;
            }
            let mut occurrence_key = key;
            for earlier in 0..idx {
                let Some(range) = self.occurrence_range(field, earlier) else { continue };
                occurrence_key = self.ext(field.ty, occurrence_key, occurrence_key, range.0)?;
            }
            keys.push(Some(occurrence_key));
        }
        Ok(keys)
    }

    /// Per field: the field of `K` extended by the occurrences in the range
    /// (with keys read from `Kp`). `None` if no parameter occurs in a field.
    fn ext_fields(
        &mut self,
        k: vir::ExprCSnap<'vir>,
        kp: vir::ExprCSnap<'vir>,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
    ) -> EncResult<'vir, Option<Vec<vir::ExprSnap<'vir>>>> {
        let mut extended = false;
        let mut values = Vec::new();
        for (rust, pure) in &data.fields {
            let field = self.field_occurrences(rust.ty().decompose(self.params));
            let mut value = pure.read.call()(k);
            let keys = self.occurrence_keys(&field, pure.read.call()(kp))?;
            for (idx, key) in keys.into_iter().enumerate() {
                let (Some(key), Some(range)) = (key, self.occurrence_range(&field, idx)) else {
                    continue;
                };
                extended = true;
                value = self.ext(field.ty, value, key, range.0)?;
            }
            values.push(value);
        }
        Ok(extended.then_some(values))
    }

    /// The agreement `ext_F` requires, for every field `F` that `ext_X` may
    /// extend.
    fn agree_fields(
        &mut self,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
    ) -> EncResult<'vir, Vec<(FieldRead<'vir>, Agree<'vir>)>> {
        let mut result = Vec::new();
        for (rust, pure) in &data.fields {
            let field_ty = rust.ty().decompose(self.params);
            if self.field_occurrences(field_ty).params.iter().all(Option::is_none) {
                continue;
            }
            if let Some(agree) = self.deps.require_dep::<TyIndirectEnc>(field_ty.ty)?.agree {
                result.push((pure.read, agree));
            }
        }
        Ok(result)
    }

    /// `ext_F(k, .., kp, range)` for the (decomposed) type `F`.
    fn ext(
        &mut self,
        ty: RustTyDecomposition<'vir>,
        k: vir::ExprSnap<'vir>,
        kp: vir::ExprSnap<'vir>,
        range: Range<'vir>,
    ) -> EncResult<'vir, vir::ExprSnap<'vir>> {
        let ext = self.deps.require_ref::<TyIndirectEnc>(ty.ty)?.ext;
        let args = self.deps.require_dep::<GArgsTyEnc>(ty.args)?;
        Ok(ext(k, args.get_ty(), args.get_const(), kp, range.lo, range.hi))
    }

    /// `p_F_ind(key, .., range)` for the (decomposed) type `F`.
    fn apply(
        &mut self,
        ty: RustTyDecomposition<'vir>,
        key: vir::ExprSnap<'vir>,
        range: Range<'vir>,
    ) -> EncResult<'vir, vir::ExprBool<'vir>> {
        let pred = self.deps.require_ref::<TyIndirectEnc>(ty.ty)?.pred;
        let args = self.deps.require_dep::<GArgsTyEnc>(ty.args)?;
        let (lo, hi) = (range.lo, range.hi);
        Ok(vir::expr! {
            acc([pred](key, [..[args.get_ty()]], [..[args.get_const()]], lo, hi))
        })
    }

    /// The occurrences of type parameters in the fields (see `TyParamPart`).
    fn collect_parts(
        &self,
        data: &StructData<'vir, (RustTyDatas, PureTyDatas)>,
        vid: Option<abi::VariantIdx>,
        parts: &mut [Vec<TyParamPart<'vir>>],
    ) {
        for (field_idx, (rust, pure)) in data.fields.iter().enumerate() {
            let field = self.field_occurrences(rust.ty().decompose(self.params));
            for (idx, occurrence) in field.occurrences.iter().enumerate() {
                if let Some(Param::Ty(j)) = field.params[idx] {
                    parts[j].push(TyParamPart {
                        vid,
                        field_idx,
                        read: PartRead::Field(pure.read),
                        path: occurrence.path.clone(),
                    });
                }
            }
        }
    }

    fn field_occurrences(&self, ty: RustTyDecomposition<'vir>) -> FieldOccurrences<'vir> {
        // TODO: lifetimes enclosing each other are not supported (the nodes
        // of such a type cannot be encoded either).
        let occurrences = occurrences(ty).unwrap_or_default();
        let params = occurrences
            .iter()
            .map(|o| match o.end {
                End::Region(region) => match region.kind() {
                    ty::RegionKind::ReEarlyParam(param) => Some(Param::Lifetime(param_position(
                        self.params,
                        param.index,
                        |arg| arg.as_region().is_some(),
                    ))),
                    // Not a lifetime of the constructor (e.g. `'static`).
                    _ => None,
                },
                End::Param(param) => Some(Param::Ty(param_position(self.params, param.index, |arg| {
                    arg.as_type().is_some()
                }))),
            })
            .collect();
        FieldOccurrences { ty, occurrences, params }
    }
}

type FieldRead<'vir> = vir::AdtDestructor<'vir, vir::CSnap, vir::Snap>;

/// The conjunction of the fields' agreements on `(K.f, Kp.f)`.
fn agree_fields<'vir>(
    vcx: &'vir vir::VirCtxt<'_>,
    fields: &[(FieldRead<'vir>, Agree<'vir>)],
    k: vir::ExprSnap<'vir>,
    kp: vir::ExprSnap<'vir>,
) -> vir::ExprBool<'vir> {
    let conjuncts = fields
        .iter()
        .map(|(read, agree)| {
            agree.reify(
                vcx,
                (
                    read.call()(k.downcast_ty()),
                    read.call()(kp.downcast_ty()),
                ),
            )
        })
        .collect::<Vec<_>>();
    vcx.mk_conj(&conjuncts)
}
