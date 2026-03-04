//! `:budget` module — metabolic allocation verbs.
//!
//! Tracks metabolic budget during neurogen evaluation. Every neuron and
//! synapse costs budget. The genetic program has a finite total; exceeding
//! it means the organism can't sustain itself.
//!
//! ## Verbs
//!
//! - `allocate(budget)` — set total metabolic budget
//! - `cost(neuron_count)` — query metabolic cost for N neurons
//! - `balance()` — read remaining budget

use runes_core::error::RuneError;
use runes_core::traits::{EvalContext, Module, ModuleVersion, Verb, VerbResult};
use runes_core::value::Value;

use super::builder::NeurogenBuilder;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_builder<'a>(ctx: &'a mut EvalContext) -> Result<&'a mut NeurogenBuilder, RuneError> {
    ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
        RuneError::type_error(
            "budget verbs require a NeurogenBuilder host".to_string(),
            None,
        )
    })
}

/// Cost per neuron in metabolic units.
const NEURON_COST: i64 = 10;

// ---------------------------------------------------------------------------
// Verbs
// ---------------------------------------------------------------------------

/// `allocate(budget)` — set total metabolic budget for the organism.
struct AllocateVerb;

impl Verb for AllocateVerb {
    fn name(&self) -> &str {
        "allocate"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "allocate requires 1 argument: budget".to_string(),
                Some(ctx.span),
            ));
        }

        let budget = match &args[0] {
            Value::Integer(n) => *n,
            _ => {
                return Err(RuneError::type_error(
                    "allocate: budget must be an integer".to_string(),
                    Some(ctx.span),
                ));
            }
        };

        if budget <= 0 {
            return Err(RuneError::argument(
                format!("allocate: budget must be positive, got {}", budget),
                Some(ctx.span),
            ));
        }

        let builder = require_builder(ctx)?;
        builder.set_budget(budget);
        Ok(Value::Nil)
    }
}

/// `cost(neuron_count)` — query metabolic cost for N neurons.
/// Returns the cost in metabolic units without spending it.
struct CostVerb;

impl Verb for CostVerb {
    fn name(&self) -> &str {
        "cost"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "cost requires 1 argument: neuron_count".to_string(),
                Some(ctx.span),
            ));
        }

        let count = match &args[0] {
            Value::Integer(n) => *n,
            _ => {
                return Err(RuneError::type_error(
                    "cost: neuron_count must be an integer".to_string(),
                    Some(ctx.span),
                ));
            }
        };

        // Don't need builder access — pure calculation
        let total_cost = count * NEURON_COST;
        Ok(Value::Integer(total_cost))
    }
}

/// `balance()` — read remaining metabolic budget.
struct BalanceVerb;

impl Verb for BalanceVerb {
    fn name(&self) -> &str {
        "balance"
    }

    fn call(&self, _args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let builder = require_builder(ctx)?;
        Ok(Value::Integer(builder.remaining_budget()))
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// `:budget` module — metabolic allocation tracking.
pub struct BudgetModule {
    verbs: Vec<Box<dyn Verb>>,
}

impl BudgetModule {
    pub fn new() -> Self {
        Self {
            verbs: vec![
                Box::new(AllocateVerb),
                Box::new(CostVerb),
                Box::new(BalanceVerb),
            ],
        }
    }
}

impl Default for BudgetModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for BudgetModule {
    fn name(&self) -> &str {
        "budget"
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

    fn make_ctx(builder: &mut NeurogenBuilder) -> EvalContext {
        let table = SymbolTable::new();
        let table_ref: &'static SymbolTable = Box::leak(Box::new(table));
        EvalContext::new(builder, Span::new(0, 0, 0, 0), table_ref)
    }

    #[test]
    fn allocate_sets_budget() {
        let mut builder = NeurogenBuilder::new();

        let verb = AllocateVerb;
        let args = vec![Value::Integer(50_000)];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        assert_eq!(builder.total_budget(), 50_000);
    }

    #[test]
    fn allocate_negative_fails() {
        let mut builder = NeurogenBuilder::new();

        let verb = AllocateVerb;
        let args = vec![Value::Integer(-100)];
        let mut ctx = make_ctx(&mut builder);
        let result = verb.call(&args, &mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn cost_computes_without_spending() {
        let mut builder = NeurogenBuilder::with_budget(10_000);

        let verb = CostVerb;
        let args = vec![Value::Integer(100)];
        let mut ctx = make_ctx(&mut builder);
        let result = verb.call(&args, &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(1000)); // 100 * 10

        // Budget unchanged
        assert_eq!(builder.remaining_budget(), 10_000);
    }

    #[test]
    fn balance_reads_remaining() {
        let mut builder = NeurogenBuilder::with_budget(5000);
        builder.spend(1500);

        let verb = BalanceVerb;
        let mut ctx = make_ctx(&mut builder);
        let result = verb.call(&[], &mut ctx).unwrap();
        assert_eq!(result, Value::Integer(3500));
    }

    #[test]
    fn module_metadata() {
        let m = BudgetModule::new();
        assert_eq!(m.name(), "budget");
        assert_eq!(m.verbs().len(), 3);
    }
}
