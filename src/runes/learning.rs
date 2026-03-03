//! `:learning` module — mastery learning verbs for nuclei programs.
//!
//! Pure-function verbs implementing the three-factor mastery learning rule:
//! participation-gated pressure accumulation, threshold-gated modification,
//! and weaken-before-flip polarity transitions. All integer operations.
//!
//! ## Synapse State Representation
//!
//! A synapse learning state is `[polarity, magnitude, maturity, pressure]`:
//! ```rune
//! syn = [1, 80, 50, 0]   # excitatory, mag 80, maturity 50, no pressure
//! ```
//!
//! ## Three-Factor Rule
//!
//! 1. `participate()` — gate: is this synapse active enough to learn?
//! 2. `pressure_accumulate()` — accumulate signed pressure from activity
//! 3. `pressure_commit()` — if pressure exceeds threshold, modify synapse
//! 4. `pressure_decay()` — decay pressure for temporal smoothing

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

/// Extract synapse state: [polarity, magnitude, maturity, pressure].
fn extract_synapse_state(val: &Value, span: Span) -> Result<(i64, i64, i64, i64), RuneError> {
    let arr = val.as_array().ok_or_else(|| {
        RuneError::type_error(
            format!("expected synapse state [pol, mag, mat, pressure], got {}", val.type_name()),
            Some(span),
        )
    })?;
    if arr.len() != 4 {
        return Err(RuneError::type_error(
            format!("synapse state must be [pol, mag, mat, pressure] (4 elements), got {}", arr.len()),
            Some(span),
        ));
    }
    Ok((
        require_int(&arr[0], "polarity", span)?,
        require_int(&arr[1], "magnitude", span)?,
        require_int(&arr[2], "maturity", span)?,
        require_int(&arr[3], "pressure", span)?,
    ))
}

fn make_synapse_state(polarity: i64, magnitude: i64, maturity: i64, pressure: i64) -> Value {
    Value::Array(vec![
        Value::Integer(polarity),
        Value::Integer(magnitude),
        Value::Integer(maturity),
        Value::Integer(pressure),
    ])
}

// ---------------------------------------------------------------------------
// Verbs
// ---------------------------------------------------------------------------

/// `participate(activity, peak, divisor)` — participation gate.
/// Returns 1 if `activity >= peak / divisor`, else 0.
/// Default divisor = 4 (top 25% participate).
struct ParticipateVerb;
impl Verb for ParticipateVerb {
    fn name(&self) -> &str { "participate" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 3, "participate", ctx.span)?;
        let activity = require_int(&args[0], "activity", ctx.span)?;
        let peak = require_int(&args[1], "peak", ctx.span)?;
        let divisor = require_int(&args[2], "divisor", ctx.span)?;
        if divisor == 0 {
            return Err(RuneError::argument("participate() divisor must not be 0", Some(ctx.span)));
        }
        let threshold = peak / divisor;
        Ok(Value::Integer(if activity >= threshold { 1 } else { 0 }))
    }
}

/// `pressure_accumulate(current_pressure, direction, activity, scale)` — accumulate pressure.
/// `new_pressure = current_pressure + direction * activity * scale / 256`
/// Direction: +1 (strengthen toward excitatory), -1 (strengthen toward inhibitory).
/// Scale is in 0..255 range, divided by 256 to keep integer.
struct PressureAccumulateVerb;
impl Verb for PressureAccumulateVerb {
    fn name(&self) -> &str { "pressure_accumulate" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 4, "pressure_accumulate", ctx.span)?;
        let pressure = require_int(&args[0], "pressure", ctx.span)?;
        let direction = require_int(&args[1], "direction", ctx.span)?;
        let activity = require_int(&args[2], "activity", ctx.span)?;
        let scale = require_int(&args[3], "scale", ctx.span)?;
        let delta = direction * activity * scale / 256;
        let new_pressure = (pressure + delta).clamp(-32768, 32767);
        Ok(Value::Integer(new_pressure))
    }
}

/// `pressure_decay(pressure, numerator, denominator)` — decay pressure by fraction.
/// `new_pressure = pressure * numerator / denominator`
/// Typical: (2, 3) keeps 2/3 of pressure each cycle.
struct PressureDecayVerb;
impl Verb for PressureDecayVerb {
    fn name(&self) -> &str { "pressure_decay" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 3, "pressure_decay", ctx.span)?;
        let pressure = require_int(&args[0], "pressure", ctx.span)?;
        let num = require_int(&args[1], "numerator", ctx.span)?;
        let den = require_int(&args[2], "denominator", ctx.span)?;
        if den == 0 {
            return Err(RuneError::argument("pressure_decay() denominator must not be 0", Some(ctx.span)));
        }
        Ok(Value::Integer(pressure * num / den))
    }
}

/// `pressure_commit(synapse_state, threshold, step)` — apply mastery learning step.
///
/// Takes synapse state `[polarity, magnitude, maturity, pressure]`, a pressure
/// threshold, and a magnitude step size. Returns new synapse state after applying
/// the weaken-before-flip rule:
///
/// - If abs(pressure) < threshold → no change
/// - If polarity aligned with pressure → strengthen (magnitude += step)
/// - If dormant (polarity 0) → awaken in pressure direction
/// - If polarity opposed → weaken (magnitude -= step). If depleted → flip.
/// - Maturity increments on any change. Pressure resets after commit.
struct PressureCommitVerb;
impl Verb for PressureCommitVerb {
    fn name(&self) -> &str { "pressure_commit" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 3, "pressure_commit", ctx.span)?;
        let (mut polarity, mut magnitude, mut maturity, pressure) =
            extract_synapse_state(&args[0], ctx.span)?;
        let threshold = require_int(&args[1], "threshold", ctx.span)?;
        let step = require_int(&args[2], "step", ctx.span)?;

        if pressure.abs() < threshold {
            // Not enough pressure — return unchanged (keep pressure)
            return Ok(make_synapse_state(polarity, magnitude, maturity, pressure));
        }

        let desired = if pressure > 0 { 1 } else { -1 };

        if polarity == 0 {
            // Dormant → awaken in desired direction
            polarity = desired;
            magnitude = step * 2;
            maturity += 1;
        } else if polarity == desired {
            // Aligned → strengthen
            magnitude = (magnitude + step).min(255);
            maturity += 1;
        } else {
            // Opposed → weaken first
            if magnitude > step {
                magnitude -= step;
                maturity += 1;
            } else {
                // Depleted → flip polarity
                polarity = desired;
                magnitude = step;
                maturity += 1;
            }
        }

        // Pressure resets after commit
        Ok(make_synapse_state(polarity, magnitude, maturity, 0))
    }
}

/// `eligibility_decay(trace, factor)` — decay an eligibility trace.
/// `new_trace = trace * factor / 256`
/// Factor 230 ≈ 0.9 retention, 204 ≈ 0.8, 128 ≈ 0.5.
struct EligibilityDecayVerb;
impl Verb for EligibilityDecayVerb {
    fn name(&self) -> &str { "eligibility_decay" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "eligibility_decay", ctx.span)?;
        let trace = require_int(&args[0], "trace", ctx.span)?;
        let factor = require_int(&args[1], "factor", ctx.span)?;
        Ok(Value::Integer(trace * factor / 256))
    }
}

/// `maturity_gate(maturity, ceiling)` — can this synapse still change?
/// Returns 1 if maturity < ceiling, else 0.
struct MaturityGateVerb;
impl Verb for MaturityGateVerb {
    fn name(&self) -> &str { "maturity_gate" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        require_args(args, 2, "maturity_gate", ctx.span)?;
        let maturity = require_int(&args[0], "maturity", ctx.span)?;
        let ceiling = require_int(&args[1], "ceiling", ctx.span)?;
        Ok(Value::Integer(if maturity < ceiling { 1 } else { 0 }))
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// `:learning` module — mastery learning verbs for nuclei programs.
pub struct LearningModule {
    verbs: Vec<Box<dyn Verb>>,
}

impl LearningModule {
    pub fn new() -> Self {
        Self {
            verbs: vec![
                Box::new(ParticipateVerb),
                Box::new(PressureAccumulateVerb),
                Box::new(PressureDecayVerb),
                Box::new(PressureCommitVerb),
                Box::new(EligibilityDecayVerb),
                Box::new(MaturityGateVerb),
            ],
        }
    }
}

impl Default for LearningModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LearningModule {
    fn name(&self) -> &str { "learning" }
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

    fn find_verb<'a>(module: &'a LearningModule, name: &str) -> &'a dyn Verb {
        module.verbs().iter().find(|v| v.name() == name).unwrap().as_ref()
    }

    fn syn(p: i64, m: i64, mat: i64, pres: i64) -> Value {
        make_synapse_state(p, m, mat, pres)
    }

    // -- participate --

    #[test]
    fn test_participate_above() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "participate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // activity 80, peak 100, divisor 4 → threshold 25 → 80 >= 25 → 1
        let result = verb.call(&[Value::Integer(80), Value::Integer(100), Value::Integer(4)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_participate_below() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "participate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // activity 10, peak 100, divisor 4 → threshold 25 → 10 < 25 → 0
        let result = verb.call(&[Value::Integer(10), Value::Integer(100), Value::Integer(4)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    #[test]
    fn test_participate_zero_divisor() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "participate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        assert!(verb.call(&[Value::Integer(50), Value::Integer(100), Value::Integer(0)], &mut ctx).is_err());
    }

    // -- pressure_accumulate --

    #[test]
    fn test_accumulate_positive() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_accumulate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // pressure=0, direction=+1, activity=128, scale=256 → delta = 1*128*256/256 = 128
        let result = verb.call(&[
            Value::Integer(0), Value::Integer(1), Value::Integer(128), Value::Integer(256),
        ], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(128));
    }

    #[test]
    fn test_accumulate_negative() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_accumulate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // direction=-1 → negative pressure
        let result = verb.call(&[
            Value::Integer(0), Value::Integer(-1), Value::Integer(128), Value::Integer(256),
        ], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(-128));
    }

    #[test]
    fn test_accumulate_clamps() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_accumulate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // Start at 32000, add more → clamped at 32767
        let result = verb.call(&[
            Value::Integer(32000), Value::Integer(1), Value::Integer(255), Value::Integer(256),
        ], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(32255.min(32767)));
    }

    // -- pressure_decay --

    #[test]
    fn test_decay_two_thirds() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_decay");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // 300 * 2 / 3 = 200
        let result = verb.call(&[Value::Integer(300), Value::Integer(2), Value::Integer(3)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(200));
    }

    #[test]
    fn test_decay_zero_denominator() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_decay");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        assert!(verb.call(&[Value::Integer(100), Value::Integer(2), Value::Integer(0)], &mut ctx).is_err());
    }

    // -- pressure_commit --

    #[test]
    fn test_commit_below_threshold() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_commit");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // pressure 30, threshold 50 → no change
        let result = verb.call(&[syn(1, 80, 10, 30), Value::Integer(50), Value::Integer(5)], &mut ctx).unwrap();
        assert_eq!(result, syn(1, 80, 10, 30));
    }

    #[test]
    fn test_commit_strengthen_aligned() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_commit");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // excitatory synapse, positive pressure → strengthen
        let result = verb.call(&[syn(1, 80, 10, 60), Value::Integer(50), Value::Integer(5)], &mut ctx).unwrap();
        assert_eq!(result, syn(1, 85, 11, 0)); // mag+5, mat+1, pressure reset
    }

    #[test]
    fn test_commit_weaken_opposed() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_commit");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // excitatory synapse, negative pressure → weaken
        let result = verb.call(&[syn(1, 80, 10, -60), Value::Integer(50), Value::Integer(5)], &mut ctx).unwrap();
        assert_eq!(result, syn(1, 75, 11, 0)); // mag-5, mat+1
    }

    #[test]
    fn test_commit_flip_depleted() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_commit");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // excitatory with magnitude 3, step 5, negative pressure → flip
        let result = verb.call(&[syn(1, 3, 10, -60), Value::Integer(50), Value::Integer(5)], &mut ctx).unwrap();
        assert_eq!(result, syn(-1, 5, 11, 0)); // flipped, mag=step, mat+1
    }

    #[test]
    fn test_commit_awaken_dormant() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_commit");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // dormant synapse, positive pressure → awaken excitatory
        let result = verb.call(&[syn(0, 0, 5, 60), Value::Integer(50), Value::Integer(5)], &mut ctx).unwrap();
        assert_eq!(result, syn(1, 10, 6, 0)); // pol=+1, mag=step*2, mat+1
    }

    #[test]
    fn test_commit_magnitude_clamp() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "pressure_commit");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // near max magnitude → clamp at 255
        let result = verb.call(&[syn(1, 253, 10, 60), Value::Integer(50), Value::Integer(5)], &mut ctx).unwrap();
        assert_eq!(result, syn(1, 255, 11, 0));
    }

    // -- eligibility_decay --

    #[test]
    fn test_eligibility_90_percent() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "eligibility_decay");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // 1000 * 230 / 256 ≈ 898 (0.898 retention)
        let result = verb.call(&[Value::Integer(1000), Value::Integer(230)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(898));
    }

    #[test]
    fn test_eligibility_50_percent() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "eligibility_decay");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        // 1000 * 128 / 256 = 500
        let result = verb.call(&[Value::Integer(1000), Value::Integer(128)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(500));
    }

    // -- maturity_gate --

    #[test]
    fn test_maturity_below_ceiling() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "maturity_gate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[Value::Integer(50), Value::Integer(200)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_maturity_above_ceiling() {
        let module = LearningModule::new();
        let verb = find_verb(&module, "maturity_gate");
        let mut host: () = ();
        let mut ctx = make_ctx(&mut host);
        let result = verb.call(&[Value::Integer(210), Value::Integer(200)], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    // -- module metadata --

    #[test]
    fn test_module_name() {
        let module = LearningModule::new();
        assert_eq!(module.name(), "learning");
    }

    #[test]
    fn test_module_verb_count() {
        let module = LearningModule::new();
        assert_eq!(module.verbs().len(), 6);
    }
}
