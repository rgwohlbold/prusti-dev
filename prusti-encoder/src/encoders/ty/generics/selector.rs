use prusti_rustc_interface::middle::ty;
use task_encoder::{EncodeFullError, EncodeFullResult, TaskEncoder, TaskEncoderDependencies};
use vir::{CallableIdn, FunctionIdn};

use crate::encoders::ty::{RustTy, RustTyDecomposition};

/// Encodes the selector ADT `p_X_lft_index` of a type constructor `X`, whose
/// values name lifetime positions of `X` by their path through its generic
/// arguments: `p_X_all()` names all of them, `p_X_lft_i()` the `i`-th
/// lifetime parameter, and `p_X_ty_j(s)` the positions `s` of the `j`-th type
/// parameter. As `X`'s definition does not know what a type parameter is
/// instantiated with, `s` is a selector of a type parameter, a value of the
/// one ADT `p_Param_lft_index`, which wraps the selectors of the types
/// instantiating parameters. Selectors are compared by term equality.
///
/// At a type parameter, the selector ADT is `p_Param_lft_index` itself.
pub struct SelectorEnc;

#[derive(Debug, Clone, Copy)]
pub struct Selector<'vir> {
    /// The selector type, `p_X_lft_index`.
    pub ty: vir::TypeSel<'vir>,
    /// The selector of all positions.
    pub all: FunctionIdn<'vir, (), vir::Sel>,
    /// The selectors of the generic arguments of the identity decomposition,
    /// by their index.
    pub args: &'vir [SelectorArg<'vir>],
    /// The casts from and to the selectors of a type parameter, `None` at a
    /// type parameter.
    pub casts: Option<SelectorCasts<'vir>>,
}

/// The selector of a generic argument of a type constructor.
#[derive(Debug, Clone, Copy)]
pub enum SelectorArg<'vir> {
    /// `p_X_lft_i()`.
    Lifetime(FunctionIdn<'vir, (), vir::Sel>),
    /// `p_X_ty_j(s)`, with its destructor, which reads `s`, and the name of
    /// the constructor, which its discriminator `is` tests for.
    Type {
        constructor: FunctionIdn<'vir, vir::Sel, vir::Sel>,
        index: vir::AdtDestructor<'vir, vir::Sel, vir::Sel>,
        name: &'vir str,
    },
    /// A const argument, which has no positions.
    Const,
}

impl<'vir> SelectorArg<'vir> {
    /// Whether `sel` selects positions of this argument: is it this lifetime
    /// or does it select positions of this type parameter.
    pub fn selects(self, sel: vir::ExprSel<'vir>) -> vir::ExprBool<'vir> {
        vir::with_vcx(|vcx| match self {
            SelectorArg::Lifetime(lifetime) => vcx.mk_eq_expr(sel, lifetime()),
            SelectorArg::Type { name, .. } => vcx.mk_adt_discriminator_expr(sel, name),
            SelectorArg::Const => unreachable!("a const argument has no positions"),
        })
    }
}

/// `make_generic_sel_X(s)`, which casts a selector of `X` to the selector of
/// a type parameter instantiated with `X`, and `make_concrete_sel_X(p)`, its
/// inverse on the selectors of `X` and `p_Param_all()`.
#[derive(Debug, Clone, Copy)]
pub struct SelectorCasts<'vir> {
    pub make_generic: FunctionIdn<'vir, vir::Sel, vir::Sel>,
    pub make_concrete: FunctionIdn<'vir, vir::Sel, vir::Sel>,
    /// `p_Param_X(s)`, the selector `s` of `X` as one of a type parameter.
    wrap: FunctionIdn<'vir, vir::Sel, vir::Sel>,
}

impl<'vir> task_encoder::OutputRefAny for Selector<'vir> {}

/// The declarations of a type's selector: its ADT, its constructor in
/// `p_Param_lft_index` and its casts.
#[derive(Debug, Clone)]
pub struct SelectorDecls<'vir> {
    adt: vir::Adt<'vir>,
    wrap: vir::AdtConstructor<'vir>,
    casts: [vir::Function<'vir>; 2],
}

impl SelectorEnc {
    const PARAM_NAME: &str = "Param";
    const UNKNOWN_NAME: &str = "p_Param_Unknown";
    const UNKNOWN_ID: &str = "p_Param_Unknown_id";

    fn adt_name(name: &str) -> String {
        format!("p_{name}_lft_index")
    }

    fn selector_ty<'vir>(name: &str) -> vir::TypeSel<'vir> {
        vir::with_vcx(|vcx| {
            vir::DomainIdn::<vir::Sel>::new(
                vir::ViperIdent::new(vcx.alloc_str(&Self::adt_name(name))),
                0,
            )()
        })
    }
}

impl TaskEncoder for SelectorEnc {
    task_encoder::encoder_cache!(SelectorEnc);
    const ENCODER_NAME: &'static str = "selector encoder";

    type TaskDescription<'vir> = RustTy<'vir>;
    type OutputRef<'vir> = Selector<'vir>;
    type OutputFullLocal<'vir> = Option<SelectorDecls<'vir>>;

    fn task_to_key<'vir>(task: &Self::TaskDescription<'vir>) -> Self::TaskKey<'vir> {
        *task
    }

    fn do_encode_full<'vir>(
        task_key: &Self::TaskKey<'vir>,
        deps: &mut TaskEncoderDependencies<'vir, Self>,
    ) -> EncodeFullResult<'vir, Self> {
        vir::with_vcx(|vcx| {
            let param_ty = Self::selector_ty(Self::PARAM_NAME);
            let param_all = FunctionIdn::new(
                vir::vir_format_identifier!(vcx, "p_{}_all", Self::PARAM_NAME),
                (),
                param_ty,
            );
            // Every type parameter, and `dyn Trait`, has the selectors of a
            // type parameter.
            if task_key.specifics.is_param() {
                deps.emit_output_ref(
                    *task_key,
                    Selector {
                        ty: param_ty,
                        all: param_all,
                        args: &[],
                        casts: None,
                    },
                )?;
                return Ok((None, ()));
            }
            let name = task_key.name();
            let ty = Self::selector_ty(name);
            let all = FunctionIdn::new(vir::vir_format_identifier!(vcx, "p_{name}_all"), (), ty);
            let no_fields: &[vir::LocalDeclDyn<'vir>] = &[];
            let mut constructors = vec![vcx.mk_adt_constructor(all.name().to_str(), no_fields)];
            let mut args = Vec::new();
            let (mut lifetimes, mut types) = (0, 0);
            for arg in RustTyDecomposition::identity(task_key).args.args() {
                args.push(match arg.kind() {
                    ty::GenericArgKind::Lifetime(_) => {
                        let lifetime = FunctionIdn::new(
                            vir::vir_format_identifier!(vcx, "p_{name}_lft_{lifetimes}"),
                            (),
                            ty,
                        );
                        lifetimes += 1;
                        constructors
                            .push(vcx.mk_adt_constructor(lifetime.name().to_str(), no_fields));
                        SelectorArg::Lifetime(lifetime)
                    }
                    ty::GenericArgKind::Type(_) => {
                        let constructor = FunctionIdn::new(
                            vir::vir_format_identifier!(vcx, "p_{name}_ty_{types}"),
                            param_ty,
                            ty,
                        );
                        let index = vcx.mk_adt_destructor(
                            vir::vir_format!(vcx, "p_{name}_ty_{types}_index"),
                            ty,
                            param_ty,
                        );
                        types += 1;
                        let field = vcx.mk_local_decl(index.name, param_ty);
                        constructors.push(vcx.mk_adt_constructor(
                            constructor.name().to_str(),
                            vcx.alloc_slice(&[field]),
                        ));
                        SelectorArg::Type {
                            constructor,
                            index,
                            name: constructor.name().to_str(),
                        }
                    }
                    ty::GenericArgKind::Const(_) => SelectorArg::Const,
                });
            }

            // The selector of `X` as one of a type parameter.
            let wrap = FunctionIdn::new(
                vir::vir_format_identifier!(vcx, "p_{}_{name}", Self::PARAM_NAME),
                ty,
                param_ty,
            );
            let unwrap = vcx.mk_adt_destructor(
                vir::vir_format!(vcx, "p_{}_{name}_index", Self::PARAM_NAME),
                param_ty,
                ty,
            );
            let wrap_constructor = vcx.mk_adt_constructor(
                wrap.name().to_str(),
                vcx.alloc_slice(&[vcx.mk_local_decl(unwrap.name, ty)]),
            );
            let make_generic = FunctionIdn::new(
                vir::vir_format_identifier!(vcx, "make_generic_sel_{name}"),
                ty,
                param_ty,
            );
            let make_concrete = FunctionIdn::new(
                vir::vir_format_identifier!(vcx, "make_concrete_sel_{name}"),
                param_ty,
                ty,
            );
            deps.emit_output_ref(
                *task_key,
                Selector {
                    ty,
                    all,
                    args: vcx.alloc_slice(&args),
                    casts: Some(SelectorCasts {
                        make_generic,
                        make_concrete,
                        wrap,
                    }),
                },
            )?;

            // All positions of `X` are all positions of the parameter, and
            // any other selector is wrapped. The ADT axioms make the casts
            // inverse to each other, except on `p_Param_X(p_X_all())`, which
            // no selector is.
            let s_decl = vcx.mk_local_decl("s", ty);
            let s = vcx.mk_local_ex(s_decl);
            let is_all = vcx.mk_eq_expr(s, all());
            let make_generic_body = vcx.mk_ternary_expr(is_all, param_all(), wrap(s));
            let p_decl = vcx.mk_local_decl("p", param_ty);
            let p = vcx.mk_local_ex(p_decl);
            let p_is_all = vcx.mk_adt_discriminator_expr(p, param_all.name().to_str());
            let make_concrete_body = vcx.mk_ternary_expr(p_is_all, all(), unwrap.call()(p));
            let casts = [
                vcx.mk_function(
                    make_generic,
                    (s_decl,),
                    &[],
                    &[],
                    None,
                    Some(make_generic_body),
                ),
                vcx.mk_function(
                    make_concrete,
                    (p_decl,),
                    &[],
                    &[],
                    None,
                    Some(make_concrete_body),
                ),
            ];
            let adt = vcx.mk_adt(
                vir::ViperIdent::new(vcx.alloc_str(&Self::adt_name(name))),
                &[],
                vcx.alloc_slice(&constructors),
            );
            Ok((
                Some(SelectorDecls {
                    adt,
                    wrap: wrap_constructor,
                    casts,
                }),
                (),
            ))
        })
    }

    /// Emits the selector ADTs and casts of the types, and the ADT
    /// `p_Param_lft_index` of the selectors of a type parameter: all of them,
    /// those of each type, wrapped, and the fallback `p_Param_Unknown` for
    /// those of types without a selector ADT in this program. The fallback
    /// keeps the ADT open, as `s_Param_Unknown` keeps the `s_Param` ADT open:
    /// without it, generic code could reason by case analysis over the types
    /// of this program, which would make its verdict depend on them.
    fn emit_outputs<'vir>(program: &mut task_encoder::Program<'vir>) {
        let outputs = Self::all_outputs_local(program);
        if outputs.is_empty() {
            return;
        }
        vir::with_vcx(|vcx| {
            let param_all = vir::vir_format!(vcx, "p_{}_all", Self::PARAM_NAME);
            let no_fields: &[vir::LocalDeclDyn<'vir>] = &[];
            let mut constructors = vec![vcx.mk_adt_constructor(param_all, no_fields)];
            for decls in outputs.into_iter().flatten() {
                program.add_adt(decls.adt);
                for cast in decls.casts {
                    program.add_function(cast);
                }
                constructors.push(decls.wrap);
            }
            let unknown_args =
                vcx.alloc_array(&[vcx.mk_local_decl(Self::UNKNOWN_ID, vir::TYPE_INT)]);
            constructors.push(vcx.mk_adt_constructor(Self::UNKNOWN_NAME, unknown_args));
            program.add_adt(vcx.mk_adt(
                vir::ViperIdent::new(vcx.alloc_str(&Self::adt_name(Self::PARAM_NAME))),
                &[],
                vcx.alloc_slice(&constructors),
            ));
        })
    }
}

/// The selector of the positions at `path` in `ty`, a path of generic argument
/// indices as [`EnclosureGraph::path`] gives it: that of a lifetime if the path
/// ends at one, otherwise of the positions `tail` of the type argument it ends
/// at, all of them if `tail` is `None`.
///
/// [`EnclosureGraph::path`]: super::lifetime_position::EnclosureGraph::path
pub fn path_selector<'vir, E: TaskEncoder>(
    ty: RustTyDecomposition<'vir>,
    path: &[usize],
    tail: Option<vir::ExprSel<'vir>>,
    deps: &mut TaskEncoderDependencies<'vir, E>,
) -> Result<vir::ExprSel<'vir>, EncodeFullError<'vir, E>> {
    let selector = deps.require_ref::<SelectorEnc>(ty.ty)?;
    let Some((&arg, rest)) = path.split_first() else {
        return Ok(match tail {
            Some(tail) if ty.ty.specifics.is_param() => tail,
            Some(_) => unreachable!("a path ends at a type parameter"),
            None => (selector.all)(),
        });
    };
    Ok(match (selector.args[arg], ty.args.args()[arg].kind()) {
        (SelectorArg::Lifetime(lifetime), _) => {
            assert!(rest.is_empty(), "a path ends at a lifetime");
            lifetime()
        }
        (SelectorArg::Type { constructor, .. }, ty::GenericArgKind::Type(arg_ty)) => {
            let arg_ty = RustTyDecomposition::from_ty(arg_ty, ty.args.context());
            // All positions of the argument, and those of a type parameter, are
            // already a selector of a type parameter. A concrete argument's own
            // selector is wrapped into one.
            let inner = if rest.is_empty() && !arg_ty.ty.specifics.is_param() {
                match tail {
                    Some(tail) => tail,
                    None => (deps
                        .require_ref::<SelectorEnc>(RustTyDecomposition::param())?
                        .all)(),
                }
            } else {
                let inner = path_selector(arg_ty, rest, tail, deps)?;
                match deps.require_ref::<SelectorEnc>(arg_ty.ty)?.casts {
                    Some(casts) => (casts.wrap)(inner),
                    None => inner,
                }
            };
            constructor(inner)
        }
        _ => unreachable!("a path leads through a constant"),
    })
}
