//! Runes modules shipped with neuropool.
//!
//! These modules provide verbs for `.rune` programs that operate on
//! ternary signal data. All verbs are pure functions — no host access.
//!
//! Requires the `runes` feature flag.

mod signal;

pub use signal::SignalModule;
