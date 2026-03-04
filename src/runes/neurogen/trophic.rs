//! `:trophic` module — growth factor verbs for neurogen programs.
//!
//! Trophic gradients guide differentiation by establishing named attraction
//! points within a region. Differentiation discs reference these gradients
//! via the `near` verb to control where neurons of a given type concentrate.
//!
//! All verbs require an active region in the NeurogenBuilder host.
//!
//! ## Verbs
//!
//! - `secrete(name, strength, radius)` — establish a named trophic gradient
//! - `gradient_strength(name)` — query gradient strength by name
//! - `affinity(gradient_a, gradient_b)` — compute overlap between two gradients

use runes_core::error::RuneError;
use runes_core::traits::{EvalContext, Module, ModuleVersion, Verb, VerbResult};
use runes_core::value::Value;

use super::builder::{GradientSpec, NeurogenBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_symbol(val: &Value, ctx: &EvalContext, verb: &str, param: &str) -> Result<String, RuneError> {
    match val {
        Value::Symbol(id) => Ok(ctx.symbols.resolve(*id).to_string()),
        Value::String(s) => Ok(s.to_string()),
        _ => Err(RuneError::type_error(
            format!("{}: {} must be a symbol or string", verb, param),
            None,
        )),
    }
}

fn extract_int(val: &Value, verb: &str, param: &str) -> Result<i64, RuneError> {
    match val {
        Value::Integer(n) => Ok(*n),
        _ => Err(RuneError::type_error(
            format!("{}: {} must be an integer", verb, param),
            None,
        )),
    }
}

// ---------------------------------------------------------------------------
// Verbs
// ---------------------------------------------------------------------------

/// `secrete(name, strength, radius)` — establish a named trophic gradient
/// in the active region.
struct SecreteVerb;

impl Verb for SecreteVerb {
    fn name(&self) -> &str {
        "secrete"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.len() < 3 {
            return Err(RuneError::argument(
                "secrete requires 3 arguments: name, strength, radius".to_string(),
                Some(ctx.span),
            ));
        }

        let name = extract_symbol(&args[0], ctx, "secrete", "name")?;
        let strength = extract_int(&args[1], "secrete", "strength")?;
        let radius = extract_int(&args[2], "secrete", "radius")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("secrete requires NeurogenBuilder host".to_string(), None)
        })?;

        if !builder.in_region() {
            return Err(RuneError::type_error(
                "secrete must be called inside a region block".to_string(),
                Some(span),
            ));
        }

        let region = builder.active_region_mut().unwrap();
        region.gradients.push(GradientSpec {
            name,
            strength,
            radius,
        });

        Ok(Value::Nil)
    }
}

/// `gradient_strength(name)` — query the strength of a named gradient
/// in the active region. Returns the strength value, or 0 if not found.
struct GradientStrengthVerb;

impl Verb for GradientStrengthVerb {
    fn name(&self) -> &str {
        "gradient_strength"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "gradient_strength requires 1 argument: name".to_string(),
                Some(ctx.span),
            ));
        }

        let name = extract_symbol(&args[0], ctx, "gradient_strength", "name")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("gradient_strength requires NeurogenBuilder host".to_string(), None)
        })?;

        if !builder.in_region() {
            return Err(RuneError::type_error(
                "gradient_strength must be called inside a region block".to_string(),
                Some(span),
            ));
        }

        let region = builder.active_region_mut().unwrap();
        let strength = region
            .gradients
            .iter()
            .find(|g| g.name == name)
            .map(|g| g.strength)
            .unwrap_or(0);

        Ok(Value::Integer(strength))
    }
}

/// `affinity(gradient_a, gradient_b)` — compute overlap between two gradients.
struct AffinityVerb;

impl Verb for AffinityVerb {
    fn name(&self) -> &str {
        "affinity"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.len() < 2 {
            return Err(RuneError::argument(
                "affinity requires 2 arguments: gradient_a, gradient_b".to_string(),
                Some(ctx.span),
            ));
        }

        let name_a = extract_symbol(&args[0], ctx, "affinity", "gradient_a")?;
        let name_b = extract_symbol(&args[1], ctx, "affinity", "gradient_b")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("affinity requires NeurogenBuilder host".to_string(), None)
        })?;

        if !builder.in_region() {
            return Err(RuneError::type_error(
                "affinity must be called inside a region block".to_string(),
                Some(span),
            ));
        }

        let region = builder.active_region_mut().unwrap();

        let grad_a = region.gradients.iter().find(|g| g.name == name_a);
        let grad_b = region.gradients.iter().find(|g| g.name == name_b);

        let overlap = match (grad_a, grad_b) {
            (Some(a), Some(b)) => {
                let sum = a.radius + b.radius;
                if sum == 0 {
                    0
                } else {
                    let smaller = a.radius.min(b.radius);
                    (smaller * 100 / sum).min(100)
                }
            }
            _ => 0,
        };

        Ok(Value::Integer(overlap))
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

pub struct TrophicModule {
    verbs: Vec<Box<dyn Verb>>,
}

impl TrophicModule {
    pub fn new() -> Self {
        Self {
            verbs: vec![
                Box::new(SecreteVerb),
                Box::new(GradientStrengthVerb),
                Box::new(AffinityVerb),
            ],
        }
    }
}

impl Default for TrophicModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TrophicModule {
    fn name(&self) -> &str {
        "trophic"
    }

    fn version(&self) -> ModuleVersion {
        ModuleVersion::new(1, 0, 0)
    }

    fn verbs(&self) -> &[Box<dyn Verb>] {
        &self.verbs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use runes_core::span::Span;
    use runes_core::symbol::SymbolTable;

    fn make_ctx<'a>(builder: &'a mut NeurogenBuilder) -> EvalContext<'a> {
        let table: &'static SymbolTable = Box::leak(Box::new(SymbolTable::new()));
        EvalContext::new(builder, Span::new(0, 0, 0, 0), table)
    }

    // Use Value::String in tests to avoid symbol table intern/borrow issues.
    fn sym(s: &str) -> Value {
        Value::String(s.into())
    }

    #[test]
    fn secrete_establishes_gradient() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());

        let verb = SecreteVerb;
        let args = vec![sym("respiratory_center"), Value::Integer(180), Value::Integer(30)];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        let region = builder.active_region_mut().unwrap();
        assert_eq!(region.gradients.len(), 1);
        assert_eq!(region.gradients[0].name, "respiratory_center");
        assert_eq!(region.gradients[0].strength, 180);
        assert_eq!(region.gradients[0].radius, 30);
    }

    #[test]
    fn secrete_outside_region_fails() {
        let mut builder = NeurogenBuilder::new();
        let verb = SecreteVerb;
        let args = vec![sym("test"), Value::Integer(100), Value::Integer(20)];
        let mut ctx = make_ctx(&mut builder);
        assert!(verb.call(&args, &mut ctx).is_err());
    }

    #[test]
    fn secrete_wrong_arg_count() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());

        let verb = SecreteVerb;
        let args = vec![sym("name")];
        let mut ctx = make_ctx(&mut builder);
        assert!(verb.call(&args, &mut ctx).is_err());
    }

    #[test]
    fn gradient_strength_returns_value() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());
        builder.active_region_mut().unwrap().gradients.push(GradientSpec {
            name: "center".to_string(),
            strength: 200,
            radius: 25,
        });

        let verb = GradientStrengthVerb;
        let args = vec![sym("center")];
        let mut ctx = make_ctx(&mut builder);
        let result = verb.call(&args, &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(200));
    }

    #[test]
    fn gradient_strength_unknown_returns_zero() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());

        let verb = GradientStrengthVerb;
        let args = vec![sym("nonexistent")];
        let mut ctx = make_ctx(&mut builder);
        let result = verb.call(&args, &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    #[test]
    fn affinity_computes_overlap() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());
        let region = builder.active_region_mut().unwrap();
        region.gradients.push(GradientSpec {
            name: "a".to_string(),
            strength: 100,
            radius: 30,
        });
        region.gradients.push(GradientSpec {
            name: "b".to_string(),
            strength: 100,
            radius: 30,
        });

        let verb = AffinityVerb;
        let args = vec![sym("a"), sym("b")];
        let mut ctx = make_ctx(&mut builder);
        let result = verb.call(&args, &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(50));
    }

    #[test]
    fn module_metadata() {
        let m = TrophicModule::new();
        assert_eq!(m.name(), "trophic");
        assert_eq!(m.verbs().len(), 3);
    }
}
