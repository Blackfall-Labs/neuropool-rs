//! `:signal` module — ternary signal math verbs for nuclei programs.
//!
//! Signals are represented as `[polarity, magnitude, multiplier]` integer triples.
//! Signal vectors are arrays of such triples. All verbs are pure functions.
//!
//! ## Signal Representation
//!
//! ```rune
//! # A single signal: [polarity, magnitude, multiplier]
//! sig = [1, 50, 1]    # excitatory, magnitude 50, multiplier 1
//! sig = [-1, 30, 2]   # inhibitory, magnitude 30, multiplier 2
//! sig = [0, 0, 1]     # silent
//!
//! # A signal vector: array of signals
//! vec = [[1, 50, 1], [-1, 30, 1], [1, 80, 1]]
//! ```
//!
//! ## Current (i32)
//!
//! The effective value of a signal: `polarity * magnitude * multiplier`.

use runes_core::error::RuneError;
use runes_core::span::Span;
use runes_core::traits::{EvalContext, Module, ModuleVersion, Verb, VerbResult};
use runes_core::value::Value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a signal triple from a Value::Array of 3 integers.
/// Returns (polarity, magnitude, multiplier).
fn extract_signal(val: &Value, span: Span) -> Result<(i64, i64, i64), RuneError> {
    let arr = val.as_array().ok_or_else(|| {
        RuneError::type_error(
            format!("expected Signal [pol, mag, mul], got {}", val.type_name()),
            Some(span),
        )
    })?;
    if arr.len() != 3 {
        return Err(RuneError::type_error(
            format!("Signal must be [pol, mag, mul] (3 elements), got {}", arr.len()),
            Some(span),
        ));
    }
    let p = arr[0].as_integer().ok_or_else(|| {
        RuneError::type_error("Signal polarity must be Integer", Some(span))
    })?;
    let m = arr[1].as_integer().ok_or_else(|| {
        RuneError::type_error("Signal magnitude must be Integer", Some(span))
    })?;
    let k = arr[2].as_integer().ok_or_else(|| {
        RuneError::type_error("Signal multiplier must be Integer", Some(span))
    })?;
    Ok((p, m, k))
}

/// Compute the current (effective value) of a signal: polarity * magnitude * multiplier.
fn signal_current(p: i64, m: i64, k: i64) -> i64 {
    p * m * k
}

/// Build a signal Value from components.
fn make_signal(p: i64, m: i64, k: i64) -> Value {
    Value::Array(vec![
        Value::Integer(p),
        Value::Integer(m),
        Value::Integer(k),
    ])
}

/// Extract a signal vector (array of signal triples) from a Value.
fn extract_signal_vec(val: &Value, span: Span) -> Result<Vec<(i64, i64, i64)>, RuneError> {
    let arr = val.as_array().ok_or_else(|| {
        RuneError::type_error(
            format!("expected signal vector (array of signals), got {}", val.type_name()),
            Some(span),
        )
    })?;
    arr.iter().map(|v| extract_signal(v, span)).collect()
}

fn require_args(args: &[Value], expected: usize, verb: &str, span: Span) -> Result<(), RuneError> {
    if args.len() != expected {
        return Err(RuneError::argument(
            format!("{}() takes {} argument(s), got {}", verb, expected, args.len()),
            Some(span),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Verbs
// ---------------------------------------------------------------------------

/// `current(signal)` — compute effective value: polarity * magnitude * multiplier.
struct CurrentVerb;
impl Verb for CurrentVerb {
    fn name(&self) -> &str { "current" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 1, "current", ctx.span)?;
        let (p, m, k) = extract_signal(&args[0], ctx.span)?;
        Ok(Value::Integer(signal_current(p, m, k)))
    }
}

/// `dot(a, b)` — dot product of two signal vectors.
/// Returns the sum of (current_a * current_b) for each pair.
struct DotVerb;
impl Verb for DotVerb {
    fn name(&self) -> &str { "dot" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "dot", ctx.span)?;
        let a = extract_signal_vec(&args[0], ctx.span)?;
        let b = extract_signal_vec(&args[1], ctx.span)?;
        if a.len() != b.len() {
            return Err(RuneError::argument(
                format!("dot() vectors must be same length: {} vs {}", a.len(), b.len()),
                Some(ctx.span),
            ));
        }
        let sum: i64 = a.iter().zip(b.iter())
            .map(|((pa, ma, ka), (pb, mb, kb))| {
                signal_current(*pa, *ma, *ka) * signal_current(*pb, *mb, *kb)
            })
            .sum();
        Ok(Value::Integer(sum))
    }
}

/// `scale(signal, factor)` — scale a signal's magnitude by an integer factor.
/// Returns a new signal with magnitude = original * factor (clamped to 0..=255).
struct ScaleVerb;
impl Verb for ScaleVerb {
    fn name(&self) -> &str { "scale" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "scale", ctx.span)?;
        let (p, m, k) = extract_signal(&args[0], ctx.span)?;
        let factor = args[1].as_integer().ok_or_else(|| {
            RuneError::type_error(
                format!("scale() factor must be Integer, got {}", args[1].type_name()),
                Some(ctx.span),
            )
        })?;
        let scaled = (m * factor).clamp(0, 255);
        Ok(make_signal(p, scaled, k))
    }
}

/// `gate(signal, control)` — gate a signal by a control signal.
/// If control current is 0, output is silent. Otherwise, scales magnitude
/// by abs(control current), clamped to 0..=255.
struct GateVerb;
impl Verb for GateVerb {
    fn name(&self) -> &str { "gate" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "gate", ctx.span)?;
        let (p, m, k) = extract_signal(&args[0], ctx.span)?;
        let (cp, cm, ck) = extract_signal(&args[1], ctx.span)?;
        let control_current = signal_current(cp, cm, ck);
        if control_current == 0 {
            Ok(make_signal(0, 0, 1))
        } else {
            let gated_mag = (m * control_current.abs() / 128).clamp(0, 255);
            Ok(make_signal(p, gated_mag, k))
        }
    }
}

/// `combine(a, b)` — add two signals by their currents.
/// Returns a new signal with the combined current, decomposed back into
/// polarity + magnitude + multiplier=1.
struct CombineVerb;
impl Verb for CombineVerb {
    fn name(&self) -> &str { "combine" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "combine", ctx.span)?;
        let (pa, ma, ka) = extract_signal(&args[0], ctx.span)?;
        let (pb, mb, kb) = extract_signal(&args[1], ctx.span)?;
        let sum = signal_current(pa, ma, ka) + signal_current(pb, mb, kb);
        let polarity = if sum > 0 { 1 } else if sum < 0 { -1 } else { 0 };
        let magnitude = sum.unsigned_abs().min(255) as i64;
        Ok(make_signal(polarity, magnitude, 1))
    }
}

/// `threshold(signal, level)` — returns 1 if abs(current) >= level, else 0.
struct ThresholdVerb;
impl Verb for ThresholdVerb {
    fn name(&self) -> &str { "threshold" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "threshold", ctx.span)?;
        let (p, m, k) = extract_signal(&args[0], ctx.span)?;
        let level = args[1].as_integer().ok_or_else(|| {
            RuneError::type_error(
                format!("threshold() level must be Integer, got {}", args[1].type_name()),
                Some(ctx.span),
            )
        })?;
        let current = signal_current(p, m, k).abs();
        Ok(Value::Integer(if current >= level { 1 } else { 0 }))
    }
}

/// `max_reduce(signals)` — find the signal with the largest absolute current.
/// Returns the signal triple itself (not the current value).
struct MaxReduceVerb;
impl Verb for MaxReduceVerb {
    fn name(&self) -> &str { "max_reduce" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 1, "max_reduce", ctx.span)?;
        let signals = extract_signal_vec(&args[0], ctx.span)?;
        if signals.is_empty() {
            return Ok(make_signal(0, 0, 1));
        }
        let (best_p, best_m, best_k) = signals.iter()
            .max_by_key(|(p, m, k)| signal_current(*p, *m, *k).abs())
            .copied()
            .unwrap();
        Ok(make_signal(best_p, best_m, best_k))
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// `:signal` module — ternary signal math for nuclei programs.
pub struct SignalModule {
    verbs: Vec<Box<dyn Verb>>,
}

impl SignalModule {
    pub fn new() -> Self {
        Self {
            verbs: vec![
                Box::new(CurrentVerb),
                Box::new(DotVerb),
                Box::new(ScaleVerb),
                Box::new(GateVerb),
                Box::new(CombineVerb),
                Box::new(ThresholdVerb),
                Box::new(MaxReduceVerb),
            ],
        }
    }
}

impl Default for SignalModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SignalModule {
    fn name(&self) -> &str { "signal" }
    fn version(&self) -> ModuleVersion { ModuleVersion::new(1, 0, 0) }
    fn verbs(&self) -> &[Box<dyn Verb>] { &self.verbs }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use runes_core::symbol::SymbolTable;
    use std::any::Any;

    fn make_ctx(host: &mut dyn Any) -> EvalContext<'_> {
        let symbols = Box::leak(Box::new(SymbolTable::new()));
        EvalContext::new(host, Span::empty(), symbols)
    }

    fn find_verb<'a>(module: &'a SignalModule, name: &str) -> &'a dyn Verb {
        module.verbs().iter().find(|v| v.name() == name).unwrap().as_ref()
    }

    fn sig(p: i64, m: i64, k: i64) -> Value {
        make_signal(p, m, k)
    }

    fn sig_vec(signals: &[(i64, i64, i64)]) -> Value {
        Value::Array(signals.iter().map(|(p, m, k)| sig(*p, *m, *k)).collect())
    }

    // -- current --

    #[test]
    fn test_current_excitatory() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "current");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(1, 50, 1)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(50));
    }

    #[test]
    fn test_current_inhibitory() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "current");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(-1, 30, 2)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(-60));
    }

    #[test]
    fn test_current_silent() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "current");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(0, 0, 1)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    // -- dot --

    #[test]
    fn test_dot_parallel() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "dot");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // [1,10,1] · [1,10,1] = 100, [1,5,1] · [1,5,1] = 25 → 125
        let a = sig_vec(&[(1, 10, 1), (1, 5, 1)]);
        let b = sig_vec(&[(1, 10, 1), (1, 5, 1)]);
        let result = verb.call(&[a, b], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(125));
    }

    #[test]
    fn test_dot_opposing() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "dot");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // [1,10,1] · [-1,10,1] = -100
        let a = sig_vec(&[(1, 10, 1)]);
        let b = sig_vec(&[(-1, 10, 1)]);
        let result = verb.call(&[a, b], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(-100));
    }

    #[test]
    fn test_dot_length_mismatch() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "dot");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let a = sig_vec(&[(1, 10, 1)]);
        let b = sig_vec(&[(1, 10, 1), (1, 5, 1)]);
        assert!(verb.call(&[a, b], &mut ctx).is_err());
    }

    // -- scale --

    #[test]
    fn test_scale() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "scale");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(1, 50, 1), Value::Integer(3)], &mut ctx).unwrap();
        assert_eq!(result, sig(1, 150, 1));
    }

    #[test]
    fn test_scale_clamps() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "scale");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(1, 200, 1), Value::Integer(2)], &mut ctx).unwrap();
        assert_eq!(result, sig(1, 255, 1)); // clamped
    }

    // -- gate --

    #[test]
    fn test_gate_open() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "gate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // control = [1, 128, 1] → current 128 → abs/128 = 1 → magnitude unchanged
        let result = verb.call(&[sig(1, 80, 1), sig(1, 128, 1)], &mut ctx).unwrap();
        assert_eq!(result, sig(1, 80, 1));
    }

    #[test]
    fn test_gate_closed() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "gate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(1, 80, 1), sig(0, 0, 1)], &mut ctx).unwrap();
        assert_eq!(result, sig(0, 0, 1)); // silent
    }

    // -- combine --

    #[test]
    fn test_combine_same_polarity() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "combine");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // 30 + 50 = 80 excitatory
        let result = verb.call(&[sig(1, 30, 1), sig(1, 50, 1)], &mut ctx).unwrap();
        assert_eq!(result, sig(1, 80, 1));
    }

    #[test]
    fn test_combine_opposing() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "combine");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // 30 + (-50) = -20 → inhibitory magnitude 20
        let result = verb.call(&[sig(1, 30, 1), sig(-1, 50, 1)], &mut ctx).unwrap();
        assert_eq!(result, sig(-1, 20, 1));
    }

    #[test]
    fn test_combine_cancel() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "combine");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(1, 50, 1), sig(-1, 50, 1)], &mut ctx).unwrap();
        assert_eq!(result, sig(0, 0, 1));
    }

    // -- threshold --

    #[test]
    fn test_threshold_above() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "threshold");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(1, 80, 1), Value::Integer(50)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_threshold_below() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "threshold");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[sig(1, 30, 1), Value::Integer(50)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    #[test]
    fn test_threshold_inhibitory() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "threshold");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // abs(-80) = 80 >= 50
        let result = verb.call(&[sig(-1, 80, 1), Value::Integer(50)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    // -- max_reduce --

    #[test]
    fn test_max_reduce() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "max_reduce");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let signals = sig_vec(&[(1, 30, 1), (-1, 90, 1), (1, 50, 1)]);
        let result = verb.call(&[signals], &mut ctx).unwrap();
        assert_eq!(result, sig(-1, 90, 1)); // abs(90) is largest
    }

    #[test]
    fn test_max_reduce_empty() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "max_reduce");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[Value::Array(vec![])], &mut ctx).unwrap();
        assert_eq!(result, sig(0, 0, 1)); // silent
    }

    // -- type errors --

    #[test]
    fn test_current_wrong_type() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "current");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        assert!(verb.call(&[Value::Integer(42)], &mut ctx).is_err());
    }

    #[test]
    fn test_current_wrong_arity() {
        let module = SignalModule::new();
        let verb = find_verb(&module, "current");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        assert!(verb.call(&[], &mut ctx).is_err());
    }

    // -- module metadata --

    #[test]
    fn test_module_name() {
        let module = SignalModule::new();
        assert_eq!(module.name(), "signal");
    }

    #[test]
    fn test_module_verb_count() {
        let module = SignalModule::new();
        assert_eq!(module.verbs().len(), 7);
    }
}
