//! `:cascade` module — neuron firing context verbs for nuclei programs.
//!
//! These verbs operate on zone inputs and membrane state during cascade-triggered
//! execution. The cascade engine pre-populates program variables with zone data;
//! these verbs do the computation.
//!
//! ## Calling Convention
//!
//! The cascade engine sets variables before evaluating a nuclei program:
//! - `ff_zone` — feedforward dendritic zone signals (array of signal triples)
//! - `ctx_zone` — lateral/context zone signals
//! - `fb_zone` — top-down feedback zone signals
//! - `membrane` — current membrane potential (integer)
//! - `cluster_act` — bound neurons' activation signals
//!
//! The program's return value is the output signal (or array of output signals).

use runes_core::error::RuneError;
use runes_core::span::Span;
use runes_core::traits::{EvalContext, Module, ModuleVersion, Verb, VerbResult};
use runes_core::value::Value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_args(args: &[Value], expected: usize, verb: &str, span: Span) -> Result<(), RuneError> {
    if args.len() != expected {
        return Err(RuneError::argument(
            format!("{}() takes {} argument(s), got {}", verb, expected, args.len()),
            Some(span),
        ));
    }
    Ok(())
}

fn require_int(val: &Value, name: &str, span: Span) -> Result<i64, RuneError> {
    val.as_integer().ok_or_else(|| {
        RuneError::type_error(
            format!("{} must be Integer, got {}", name, val.type_name()),
            Some(span),
        )
    })
}

/// Sum currents of a signal vector. Each signal is [polarity, magnitude, multiplier].
fn sum_currents(val: &Value, span: Span) -> Result<i64, RuneError> {
    let arr = val.as_array().ok_or_else(|| {
        RuneError::type_error(
            format!("expected signal vector, got {}", val.type_name()),
            Some(span),
        )
    })?;
    let mut total: i64 = 0;
    for sig in arr {
        let triple = sig.as_array().ok_or_else(|| {
            RuneError::type_error("signal vector element must be [pol, mag, mul]", Some(span))
        })?;
        if triple.len() != 3 {
            return Err(RuneError::type_error(
                format!("signal must be [pol, mag, mul], got {} elements", triple.len()),
                Some(span),
            ));
        }
        let p = require_int(&triple[0], "polarity", span)?;
        let m = require_int(&triple[1], "magnitude", span)?;
        let k = require_int(&triple[2], "multiplier", span)?;
        total += p * m * k;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Verbs
// ---------------------------------------------------------------------------

/// `zone_integrate(ff_zone, ctx_zone, fb_zone, ff_weight, ctx_weight, fb_weight)`
/// Weighted sum of three dendritic zone currents.
/// Returns integer: `sum(ff)*ff_w/256 + sum(ctx)*ctx_w/256 + sum(fb)*fb_w/256`.
///
/// Typical weights (from ZoneWeights presets):
/// - Pyramidal: ff=128(50%), ctx=77(30%), fb=51(20%)
/// - Gate: ff=51(20%), ctx=128(50%), fb=77(30%)
/// - Relay: ff=179(70%), ctx=51(20%), fb=26(10%)
struct ZoneIntegrateVerb;
impl Verb for ZoneIntegrateVerb {
    fn name(&self) -> &str { "zone_integrate" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 6, "zone_integrate", ctx.span)?;
        let ff_sum = sum_currents(&args[0], ctx.span)?;
        let ctx_sum = sum_currents(&args[1], ctx.span)?;
        let fb_sum = sum_currents(&args[2], ctx.span)?;
        let ff_w = require_int(&args[3], "ff_weight", ctx.span)?;
        let ctx_w = require_int(&args[4], "ctx_weight", ctx.span)?;
        let fb_w = require_int(&args[5], "fb_weight", ctx.span)?;
        let integrated = ff_sum * ff_w / 256 + ctx_sum * ctx_w / 256 + fb_sum * fb_w / 256;
        Ok(Value::Integer(integrated))
    }
}

/// `membrane_update(membrane, input, leak)` — integrate input and apply leak.
/// `new_membrane = (membrane - membrane * leak / 256) + input`
/// Leak factor: 26 ≈ 10% leak, 51 ≈ 20%, 128 ≈ 50%.
struct MembraneUpdateVerb;
impl Verb for MembraneUpdateVerb {
    fn name(&self) -> &str { "membrane_update" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 3, "membrane_update", ctx.span)?;
        let membrane = require_int(&args[0], "membrane", ctx.span)?;
        let input = require_int(&args[1], "input", ctx.span)?;
        let leak = require_int(&args[2], "leak", ctx.span)?;
        let leaked = membrane - membrane * leak / 256;
        Ok(Value::Integer(leaked + input))
    }
}

/// `fire_check(membrane, threshold)` — does the membrane cross firing threshold?
/// Returns 1 if membrane >= threshold, else 0.
struct FireCheckVerb;
impl Verb for FireCheckVerb {
    fn name(&self) -> &str { "fire_check" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "fire_check", ctx.span)?;
        let membrane = require_int(&args[0], "membrane", ctx.span)?;
        let threshold = require_int(&args[1], "threshold", ctx.span)?;
        Ok(Value::Integer(if membrane >= threshold { 1 } else { 0 }))
    }
}

/// `refractory_check(time_since_fire, refractory_period)` — is neuron in refractory?
/// Returns 1 if still refractory (can't fire), 0 if ready.
struct RefractoryCheckVerb;
impl Verb for RefractoryCheckVerb {
    fn name(&self) -> &str { "refractory_check" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "refractory_check", ctx.span)?;
        let elapsed = require_int(&args[0], "time_since_fire", ctx.span)?;
        let period = require_int(&args[1], "refractory_period", ctx.span)?;
        Ok(Value::Integer(if elapsed < period { 1 } else { 0 }))
    }
}

/// `spike(polarity, magnitude)` — construct an output spike signal.
/// Convenience for building [polarity, magnitude, 1] triples.
struct SpikeVerb;
impl Verb for SpikeVerb {
    fn name(&self) -> &str { "spike" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "spike", ctx.span)?;
        let p = require_int(&args[0], "polarity", ctx.span)?;
        let m = require_int(&args[1], "magnitude", ctx.span)?;
        let p_clamped = p.clamp(-1, 1);
        let m_clamped = m.clamp(0, 255);
        Ok(Value::Array(vec![
            Value::Integer(p_clamped),
            Value::Integer(m_clamped),
            Value::Integer(1),
        ]))
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// `:cascade` module — neuron firing context verbs.
pub struct CascadeModule {
    verbs: Vec<Box<dyn Verb>>,
}

impl CascadeModule {
    pub fn new() -> Self {
        Self {
            verbs: vec![
                Box::new(ZoneIntegrateVerb),
                Box::new(MembraneUpdateVerb),
                Box::new(FireCheckVerb),
                Box::new(RefractoryCheckVerb),
                Box::new(SpikeVerb),
            ],
        }
    }
}

impl Default for CascadeModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CascadeModule {
    fn name(&self) -> &str { "cascade" }
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

    fn find_verb<'a>(module: &'a CascadeModule, name: &str) -> &'a dyn Verb {
        module.verbs().iter().find(|v| v.name() == name).unwrap().as_ref()
    }

    fn sig(p: i64, m: i64, k: i64) -> Value {
        Value::Array(vec![Value::Integer(p), Value::Integer(m), Value::Integer(k)])
    }

    fn sig_vec(signals: &[(i64, i64, i64)]) -> Value {
        Value::Array(signals.iter().map(|(p, m, k)| sig(*p, *m, *k)).collect())
    }

    // -- zone_integrate --

    #[test]
    fn test_zone_integrate_pyramidal() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "zone_integrate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // ff: [1,100,1] = 100, ctx: [1,50,1] = 50, fb: [1,30,1] = 30
        // pyramidal weights: ff=128, ctx=77, fb=51
        // result = 100*128/256 + 50*77/256 + 30*51/256 = 50 + 15 + 5 = 70
        let result = verb.call(&[
            sig_vec(&[(1, 100, 1)]),
            sig_vec(&[(1, 50, 1)]),
            sig_vec(&[(1, 30, 1)]),
            Value::Integer(128),
            Value::Integer(77),
            Value::Integer(51),
        ], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(70));
    }

    #[test]
    fn test_zone_integrate_empty_zones() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "zone_integrate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[
            Value::Array(vec![]),
            Value::Array(vec![]),
            Value::Array(vec![]),
            Value::Integer(128),
            Value::Integer(77),
            Value::Integer(51),
        ], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    #[test]
    fn test_zone_integrate_inhibitory() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "zone_integrate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // Inhibitory context zone suppresses
        let result = verb.call(&[
            sig_vec(&[(1, 100, 1)]),
            sig_vec(&[(-1, 200, 1)]),
            Value::Array(vec![]),
            Value::Integer(128),
            Value::Integer(128),
            Value::Integer(0),
        ], &mut ctx).unwrap();
        // 100*128/256 + (-200)*128/256 = 50 + (-100) = -50
        assert_eq!(result, Value::Integer(-50));
    }

    // -- membrane_update --

    #[test]
    fn test_membrane_integrate_and_leak() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "membrane_update");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // membrane=1000, input=200, leak=26 (10%)
        // leaked = 1000 - 1000*26/256 = 1000 - 101 = 899
        // new = 899 + 200 = 1099
        let result = verb.call(&[
            Value::Integer(1000), Value::Integer(200), Value::Integer(26),
        ], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(1099));
    }

    #[test]
    fn test_membrane_from_zero() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "membrane_update");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[
            Value::Integer(0), Value::Integer(500), Value::Integer(26),
        ], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(500));
    }

    // -- fire_check --

    #[test]
    fn test_fire_check_above() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "fire_check");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[Value::Integer(1500), Value::Integer(1000)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_fire_check_below() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "fire_check");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[Value::Integer(500), Value::Integer(1000)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    // -- refractory_check --

    #[test]
    fn test_refractory_still_active() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "refractory_check");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // 500μs since fire, refractory period 2000μs → still refractory
        let result = verb.call(&[Value::Integer(500), Value::Integer(2000)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_refractory_expired() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "refractory_check");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[Value::Integer(3000), Value::Integer(2000)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    // -- spike --

    #[test]
    fn test_spike_excitatory() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "spike");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[Value::Integer(1), Value::Integer(100)], &mut ctx).unwrap();
        assert_eq!(result, sig(1, 100, 1));
    }

    #[test]
    fn test_spike_clamps() {
        let module = CascadeModule::new();
        let verb = find_verb(&module, "spike");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[Value::Integer(5), Value::Integer(500)], &mut ctx).unwrap();
        assert_eq!(result, sig(1, 255, 1)); // clamped
    }

    // -- module metadata --

    #[test]
    fn test_module_name() {
        let module = CascadeModule::new();
        assert_eq!(module.name(), "cascade");
    }

    #[test]
    fn test_module_verb_count() {
        let module = CascadeModule::new();
        assert_eq!(module.verbs().len(), 5);
    }
}
