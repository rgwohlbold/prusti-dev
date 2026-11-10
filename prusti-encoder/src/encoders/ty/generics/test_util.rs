//! Helpers for the unit tests of the type encoders.

use prusti_interface::{environment::EnvBody, specs::typed::DefSpecificationMap};
use prusti_rustc_interface::{
    driver::{self, Compilation},
    hir::def::DefKind,
    interface::{Config, interface::Compiler},
    middle::ty::TyCtxt,
    session::config::Input,
    span::FileName,
};

use crate::encoders::ty::RustTyDecomposition;

/// Compiles `src`, which defines the type alias `X`, and calls `f` with the
/// aliased type, decomposed in the generic context of `X`.
pub(super) fn with_type_alias<R: Send>(
    src: &str,
    f: impl for<'tcx> FnOnce(RustTyDecomposition<'tcx>) -> R + Send,
) -> R {
    struct Callbacks<F, R> {
        src: String,
        f: Option<F>,
        result: Option<R>,
    }

    impl<F, R> driver::Callbacks for Callbacks<F, R>
    where
        F: for<'tcx> FnOnce(RustTyDecomposition<'tcx>) -> R + Send,
        R: Send,
    {
        fn config(&mut self, config: &mut Config) {
            config.input = Input::Str {
                name: FileName::Custom("test.rs".to_string()),
                input: self.src.clone(),
            };
        }

        fn after_analysis(&mut self, _: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
            vir::init_vcx(vir::VirCtxt::new(
                tcx,
                EnvBody::new(tcx),
                DefSpecificationMap::new(),
                vir::IdentStyle::ItemName,
            ));
            let alias = tcx
                .hir_crate_items(())
                .definitions()
                .find(|&def_id| {
                    tcx.def_kind(def_id) == DefKind::TyAlias
                        && tcx.item_name(def_id).as_str() == "X"
                })
                .expect("the input defines no type alias `X`")
                .to_def_id();
            let ty = tcx
                .type_of(alias)
                .instantiate_identity()
                .skip_normalization();
            let f = self.f.take().unwrap();
            self.result = Some(f(RustTyDecomposition::from_ty(ty, alias)));
            Compilation::Stop
        }
    }

    let mut callbacks = Callbacks {
        src: src.to_string(),
        f: Some(f),
        result: None,
    };
    let args = [
        "rustc",
        "test.rs",
        "--crate-type=lib",
        "--edition=2021",
        "--cap-lints=allow",
    ];
    driver::run_compiler(&args.map(String::from), &mut callbacks);
    callbacks.result.unwrap()
}
