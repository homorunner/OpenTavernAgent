//! A separate test binary keeps the allocation counter isolated from other tests.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tavernagent::net::{InferenceBackend, InferenceWorkspace, NetConfig, PolicyValueNet};
static COUNT: AtomicUsize = AtomicUsize::new(0);
static ENABLED: AtomicBool = AtomicBool::new(false);
struct Alloc;
unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static ALLOC: Alloc = Alloc;
#[test]
fn warmed_inference_only_allocates_the_returned_policy() {
    let net = PolicyValueNet::new(1, NetConfig::default()).unwrap();
    let mut game = taverngym::Game::new(1, Default::default());
    tavernagent::heroes::select_heroes(&mut game, 1).unwrap();
    let o = tavernagent::mcts::observe_seat(&game, 0);
    let mut ws = InferenceWorkspace::default();
    for backend in [InferenceBackend::Fma, InferenceBackend::Dot] {
        if backend == InferenceBackend::Dot && !tavernagent::linalg::native_bf16_available() {
            continue;
        }
        net.infer_with_workspace(&o.x, &o.cards, &o.mask, backend, &mut ws)
            .unwrap();
        COUNT.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        let result = net.infer_with_workspace(&o.x, &o.cards, &o.mask, backend, &mut ws);
        ENABLED.store(false, Ordering::Relaxed);
        result.unwrap();
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    }
    net.predict(&o.x, &o.cards, &o.mask).unwrap();
    COUNT.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Relaxed);
    let result = net.predict(&o.x, &o.cards, &o.mask);
    ENABLED.store(false, Ordering::Relaxed);
    result.unwrap();
    assert_eq!(COUNT.load(Ordering::Relaxed), 1);
}
