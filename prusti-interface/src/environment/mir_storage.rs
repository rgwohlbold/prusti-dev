//! This module allows storing mir bodies with borrowck facts in thread-local
//! storage.
//!
//! Unfortunately, thread-local storage requires all data stored in it to have
//! a `'static` lifetime. Therefore, we transmute the lifetime `'tcx` away when
//! storing the data. To ensure that nothing gets messed up, we require the
//! client to provide a witness: an instance of type `TyCtxt<'tcx>` that is
//! used to show that the lifetime that the client provided is indeed `'tcx`.

use crate::specs::is_spec_fn;
use prusti_rustc_interface::{
    borrowck::{consumers, consumers::BodyWithBorrowckFacts},
    data_structures::{fx::FxHashMap, steal::Steal},
    hir::{def::DefKind, def_id::LocalDefId},
    index::IndexVec,
    middle::{
        mir,
        queries::{mir_borrowck::ProvidedValue as MirBorrowck, Providers},
        ty::TyCtxt,
        util,
    },
};
use prusti_utils::config;
use std::{cell::RefCell, thread_local};

thread_local! {
    pub static SHARED_STATE_WITH_FACTS:
        RefCell<FxHashMap<LocalDefId, BodyWithBorrowckFacts<'static>>> =
        RefCell::new(FxHashMap::default());
    pub static SHARED_STATE_WITHOUT_FACTS:
        RefCell<FxHashMap<LocalDefId, mir::Body<'static>>> =
        RefCell::new(FxHashMap::default());
}

/// # Safety
///
/// See the module level comment.
pub unsafe fn store_mir_body<'tcx>(
    _tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    body_with_facts: BodyWithBorrowckFacts<'tcx>,
) {
    // SAFETY: See the module level comment.
    let body_with_facts: BodyWithBorrowckFacts<'static> =
        unsafe { std::mem::transmute(body_with_facts) };
    assert_eq!(
        body_with_facts.body.phase,
        mir::MirPhase::Analysis(mir::AnalysisPhase::Initial)
    );
    SHARED_STATE_WITH_FACTS.with(|state| {
        let mut map = state.borrow_mut();
        assert!(map.insert(def_id, body_with_facts).is_none());
    });
}

/// # Safety
///
/// See the module level comment.
#[allow(clippy::needless_lifetimes)] // We want to be very explicit about lifetimes here.
pub(crate) unsafe fn retrieve_mir_body<'tcx>(
    _tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> BodyWithBorrowckFacts<'tcx> {
    let body_with_facts: BodyWithBorrowckFacts<'static> = SHARED_STATE_WITH_FACTS.with(|state| {
        let mut map = state.borrow_mut();
        map.remove(&def_id).unwrap_or_else(|| {
            panic!("No MIR body found for {def_id:?}");
        })
    });
    // SAFETY: See the module level comment.
    unsafe { std::mem::transmute(body_with_facts) }
}

/// # Safety
///
/// See the module level comment.
pub unsafe fn store_promoted_mir_body<'tcx>(
    _tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    body: mir::Body<'tcx>,
) {
    // SAFETY: See the module level comment.
    let body: mir::Body<'static> = unsafe { std::mem::transmute(body) };
    assert_eq!(
        body.phase,
        mir::MirPhase::Analysis(mir::AnalysisPhase::Initial)
    );
    SHARED_STATE_WITHOUT_FACTS.with(|state| {
        let mut map = state.borrow_mut();
        assert!(map.insert(def_id, body).is_none());
    });
}

#[allow(clippy::needless_lifetimes)] // We want to be very explicit about lifetimes here.
pub(crate) unsafe fn retrieve_promoted_mir_body<'tcx>(
    _tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> mir::Body<'tcx> {
    let body_without_facts: mir::Body<'static> = SHARED_STATE_WITHOUT_FACTS.with(|state| {
        let mut map = state.borrow_mut();
        map.remove(&def_id).unwrap()
    });
    // SAFETY: See the module level comment.
    unsafe { std::mem::transmute(body_without_facts) }
}

/// Overrides the queries that produce the MIR bodies so that they also store
/// them here, where the encoder retrieves them.
pub fn override_queries(providers: &mut util::Providers) {
    providers.queries.mir_borrowck = mir_borrowck;
    providers.queries.mir_promoted = mir_promoted;
}

// Running `get_body_with_borrowck_facts` can be very slow, therefore we avoid it when not
// necessary; for crates which won't be verified or spec_fns it suffices to load just the fn body

#[allow(clippy::needless_lifetimes)]
#[tracing::instrument(level = "debug", skip(tcx))]
fn mir_borrowck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> MirBorrowck<'tcx> {
    let def_kind = tcx.def_kind(def_id.to_def_id());
    let is_anon_const = matches!(def_kind, DefKind::AnonConst);
    // Anon Const bodies have already been stolen and so will result in a crash
    // when calling `get_body_with_borrowck_facts`. TODO: figure out if we need
    // (anon) const bodies at all, and if so, how to get them?
    if !is_anon_const {
        let consumer_opts =
            if is_spec_fn(tcx, def_id.to_def_id()) || config::verify_mode().is_compile_only() {
                consumers::ConsumerOptions::RegionInferenceContext
            } else {
                consumers::ConsumerOptions::PoloniusInputFacts
            };
        let body_with_facts = consumers::get_bodies_with_borrowck_facts(tcx, def_id, consumer_opts);
        for (def_id, body) in body_with_facts {
            // SAFETY: This is safe because we are feeding in the same `tcx` that is
            // going to be used as a witness when pulling out the data.
            unsafe {
                store_mir_body(tcx, def_id, body);
            }
        }
    }
    let mut providers = Providers::default();
    prusti_rustc_interface::borrowck::provide(&mut providers);
    let original_mir_borrowck = providers.mir_borrowck;
    original_mir_borrowck(tcx, def_id)
}

#[allow(clippy::needless_lifetimes)]
#[tracing::instrument(level = "debug", skip(tcx))]
fn mir_promoted<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> (
    &'tcx Steal<mir::Body<'tcx>>,
    &'tcx Steal<IndexVec<mir::Promoted, mir::Body<'tcx>>>,
) {
    let original_mir_promoted = prusti_rustc_interface::interface::DEFAULT_QUERY_PROVIDERS
        .queries
        .mir_promoted;
    let result = original_mir_promoted(tcx, def_id);
    // SAFETY: This is safe because we are feeding in the same `tcx` that is
    // going to be used as a witness when pulling out the data.
    unsafe {
        store_promoted_mir_body(tcx, def_id, result.0.borrow().clone());
    }
    result
}
