//! `:develop` module — tissue generation verbs and constructs.
//!
//! Two constructs (`region do...end`, `differentiate do...end`) and
//! configuration verbs that populate the NeurogenBuilder.
//!
//! ## Constructs
//!
//! - `region :name do...end` — declare a brain region
//! - `differentiate :name do...end` — declare a differentiation disc
//!
//! ## Region Config Verbs
//!
//! - `archetype(:sym)`, `neuron_count(n)`, `neighbors([...])`, `phases(g,e,d,c)`
//!
//! ## Disc Config Verbs
//!
//! - `near(:gradient)`, `target(:type)`, `threshold(n)`, `population_cap(n)`
//! - `bind_program("name")`, `period_range(min, max)`, `spread(:dist)`
//!
//! ## Tract Verb
//!
//! - `wire(:name, :from, :to, :type)` — declare whitematter tract

use runes_core::error::RuneError;
use runes_core::traits::{
    Construct, EvalContext, Module, ModuleVersion, Verb, VerbResult,
};
use runes_core::value::Value;

use super::builder::{DiscTarget, NeurogenBuilder, PhaseDurations, TractSpec, TractType};
use crate::RegionArchetype;

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

fn archetype_from_str(s: &str) -> Option<RegionArchetype> {
    match s {
        "cortical" => Some(RegionArchetype::Cortical),
        "thalamic" => Some(RegionArchetype::Thalamic),
        "hippocampal" => Some(RegionArchetype::Hippocampal),
        "basal_ganglia" => Some(RegionArchetype::BasalGanglia),
        "cerebellar" => Some(RegionArchetype::Cerebellar),
        "brainstem" => Some(RegionArchetype::Brainstem),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Constructs
// ---------------------------------------------------------------------------

struct RegionConstruct;

impl Construct for RegionConstruct {
    fn name(&self) -> &str {
        "region"
    }

    fn enter(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "region requires a name argument".to_string(),
                Some(ctx.span),
            ));
        }

        let name = extract_symbol(&args[0], ctx, "region", "name")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("region requires NeurogenBuilder host".to_string(), None)
        })?;

        if builder.in_region() {
            return Err(RuneError::type_error(
                "cannot nest region blocks".to_string(),
                Some(span),
            ));
        }

        let idx = builder.begin_region(name);
        Ok(Value::Integer(idx as i64))
    }

    fn exit(
        &self,
        _enter_value: Value,
        _body_value: Value,
        ctx: &mut EvalContext,
    ) -> VerbResult {
        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("region requires NeurogenBuilder host".to_string(), None)
        })?;
        builder.end_region();
        Ok(Value::Nil)
    }
}

struct DifferentiateConstruct;

impl Construct for DifferentiateConstruct {
    fn name(&self) -> &str {
        "differentiate"
    }

    fn enter(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "differentiate requires a name argument".to_string(),
                Some(ctx.span),
            ));
        }

        let name = extract_symbol(&args[0], ctx, "differentiate", "name")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("differentiate requires NeurogenBuilder host".to_string(), None)
        })?;

        if !builder.in_region() {
            return Err(RuneError::type_error(
                "differentiate must be inside a region block".to_string(),
                Some(span),
            ));
        }

        if builder.in_disc() {
            return Err(RuneError::type_error(
                "cannot nest differentiate blocks".to_string(),
                Some(span),
            ));
        }

        builder.begin_disc(name);
        Ok(Value::Nil)
    }

    fn exit(
        &self,
        _enter_value: Value,
        _body_value: Value,
        ctx: &mut EvalContext,
    ) -> VerbResult {
        let span = ctx.span;
        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("differentiate requires NeurogenBuilder host".to_string(), None)
        })?;
        if !builder.end_disc() {
            return Err(RuneError::type_error(
                "differentiate exit failed — no active region".to_string(),
                Some(span),
            ));
        }
        Ok(Value::Nil)
    }
}

// ---------------------------------------------------------------------------
// Region configuration verbs
// ---------------------------------------------------------------------------

struct ArchetypeVerb;

impl Verb for ArchetypeVerb {
    fn name(&self) -> &str {
        "archetype"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "archetype requires 1 argument".to_string(),
                Some(ctx.span),
            ));
        }

        let sym = extract_symbol(&args[0], ctx, "archetype", "type")?;
        let span = ctx.span;

        let arch = archetype_from_str(&sym).ok_or_else(|| {
            RuneError::argument(
                format!(
                    "unknown archetype '{}', expected: cortical, thalamic, hippocampal, \
                     basal_ganglia, cerebellar, brainstem",
                    sym
                ),
                Some(span),
            )
        })?;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("archetype requires NeurogenBuilder host".to_string(), None)
        })?;
        let region = builder.active_region_mut().ok_or_else(|| {
            RuneError::type_error(
                "archetype must be called inside a region block".to_string(),
                Some(span),
            )
        })?;

        region.archetype = arch;
        Ok(Value::Nil)
    }
}

struct NeuronCountVerb;

impl Verb for NeuronCountVerb {
    fn name(&self) -> &str {
        "neuron_count"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "neuron_count requires 1 argument".to_string(),
                Some(ctx.span),
            ));
        }

        let count = extract_int(&args[0], "neuron_count", "count")?;
        let span = ctx.span;

        if count <= 0 {
            return Err(RuneError::argument(
                format!("neuron_count must be positive, got {}", count),
                Some(span),
            ));
        }

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("neuron_count requires NeurogenBuilder host".to_string(), None)
        })?;
        let region = builder.active_region_mut().ok_or_else(|| {
            RuneError::type_error(
                "neuron_count must be called inside a region block".to_string(),
                Some(span),
            )
        })?;

        region.neuron_count = count as u32;
        Ok(Value::Nil)
    }
}

struct NeighborsVerb;

impl Verb for NeighborsVerb {
    fn name(&self) -> &str {
        "neighbors"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "neighbors requires 1 argument: array of symbols".to_string(),
                Some(ctx.span),
            ));
        }

        let names = match &args[0] {
            Value::Array(arr) => {
                let mut names = Vec::new();
                for v in arr.iter() {
                    names.push(extract_symbol(v, ctx, "neighbors", "element")?);
                }
                names
            }
            _ => {
                return Err(RuneError::type_error(
                    "neighbors argument must be an array".to_string(),
                    Some(ctx.span),
                ));
            }
        };

        let span = ctx.span;
        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("neighbors requires NeurogenBuilder host".to_string(), None)
        })?;
        let region = builder.active_region_mut().ok_or_else(|| {
            RuneError::type_error(
                "neighbors must be called inside a region block".to_string(),
                Some(span),
            )
        })?;

        region.neighbors = names;
        Ok(Value::Nil)
    }
}

struct PhasesVerb;

impl Verb for PhasesVerb {
    fn name(&self) -> &str {
        "phases"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.len() < 4 {
            return Err(RuneError::argument(
                "phases requires 4 arguments: genesis, exposure, differentiation, crystallization"
                    .to_string(),
                Some(ctx.span),
            ));
        }

        let genesis = extract_int(&args[0], "phases", "genesis")? as u32;
        let exposure = extract_int(&args[1], "phases", "exposure")? as u32;
        let differentiation = extract_int(&args[2], "phases", "differentiation")? as u32;
        let crystallization = extract_int(&args[3], "phases", "crystallization")? as u32;

        let span = ctx.span;
        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("phases requires NeurogenBuilder host".to_string(), None)
        })?;
        let region = builder.active_region_mut().ok_or_else(|| {
            RuneError::type_error(
                "phases must be called inside a region block".to_string(),
                Some(span),
            )
        })?;

        region.phase_durations = PhaseDurations {
            genesis,
            exposure,
            differentiation,
            crystallization,
        };
        Ok(Value::Nil)
    }
}

// ---------------------------------------------------------------------------
// Disc configuration verbs
// ---------------------------------------------------------------------------

struct NearVerb;

impl Verb for NearVerb {
    fn name(&self) -> &str {
        "near"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "near requires 1 argument: gradient name".to_string(),
                Some(ctx.span),
            ));
        }

        let name = extract_symbol(&args[0], ctx, "near", "gradient")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("near requires NeurogenBuilder host".to_string(), None)
        })?;
        let disc = builder.active_disc_mut().ok_or_else(|| {
            RuneError::type_error(
                "near must be called inside a differentiate block".to_string(),
                Some(span),
            )
        })?;

        disc.near_gradient = Some(name);
        Ok(Value::Nil)
    }
}

struct TargetVerb;

impl Verb for TargetVerb {
    fn name(&self) -> &str {
        "target"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "target requires 1 argument: type symbol".to_string(),
                Some(ctx.span),
            ));
        }

        let sym = extract_symbol(&args[0], ctx, "target", "type")?;
        let span = ctx.span;

        let disc_target = DiscTarget::from_symbol(&sym).ok_or_else(|| {
            RuneError::argument(
                format!(
                    "unknown target type '{}', expected: pyramidal, interneuron, gate, relay, \
                     oscillator, memory, sensory, motor, computational",
                    sym
                ),
                Some(span),
            )
        })?;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("target requires NeurogenBuilder host".to_string(), None)
        })?;
        let disc = builder.active_disc_mut().ok_or_else(|| {
            RuneError::type_error(
                "target must be called inside a differentiate block".to_string(),
                Some(span),
            )
        })?;

        disc.target = disc_target;
        Ok(Value::Nil)
    }
}

struct ThresholdVerb;

impl Verb for ThresholdVerb {
    fn name(&self) -> &str {
        "threshold"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "threshold requires 1 argument".to_string(),
                Some(ctx.span),
            ));
        }

        let val = extract_int(&args[0], "threshold", "value")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("threshold requires NeurogenBuilder host".to_string(), None)
        })?;
        let disc = builder.active_disc_mut().ok_or_else(|| {
            RuneError::type_error(
                "threshold must be called inside a differentiate block".to_string(),
                Some(span),
            )
        })?;

        disc.threshold = val;
        Ok(Value::Nil)
    }
}

struct PopulationCapVerb;

impl Verb for PopulationCapVerb {
    fn name(&self) -> &str {
        "population_cap"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "population_cap requires 1 argument".to_string(),
                Some(ctx.span),
            ));
        }

        let val = extract_int(&args[0], "population_cap", "percentage")?;
        let span = ctx.span;

        if val < 0 || val > 100 {
            return Err(RuneError::argument(
                format!("population_cap must be 0-100, got {}", val),
                Some(span),
            ));
        }

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("population_cap requires NeurogenBuilder host".to_string(), None)
        })?;
        let disc = builder.active_disc_mut().ok_or_else(|| {
            RuneError::type_error(
                "population_cap must be called inside a differentiate block".to_string(),
                Some(span),
            )
        })?;

        disc.population_cap = val;
        Ok(Value::Nil)
    }
}

struct BindProgramVerb;

impl Verb for BindProgramVerb {
    fn name(&self) -> &str {
        "bind_program"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "bind_program requires 1 argument: program name".to_string(),
                Some(ctx.span),
            ));
        }

        let name = extract_symbol(&args[0], ctx, "bind_program", "name")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("bind_program requires NeurogenBuilder host".to_string(), None)
        })?;
        let disc = builder.active_disc_mut().ok_or_else(|| {
            RuneError::type_error(
                "bind_program must be called inside a differentiate block".to_string(),
                Some(span),
            )
        })?;

        disc.bind_program = Some(name);
        Ok(Value::Nil)
    }
}

struct PeriodRangeVerb;

impl Verb for PeriodRangeVerb {
    fn name(&self) -> &str {
        "period_range"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.len() < 2 {
            return Err(RuneError::argument(
                "period_range requires 2 arguments: min_us, max_us".to_string(),
                Some(ctx.span),
            ));
        }

        let min = extract_int(&args[0], "period_range", "min_us")?;
        let max = extract_int(&args[1], "period_range", "max_us")?;
        let span = ctx.span;

        if min <= 0 || max <= 0 || min > max {
            return Err(RuneError::argument(
                format!("period_range: invalid range [{}, {}]", min, max),
                Some(span),
            ));
        }

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("period_range requires NeurogenBuilder host".to_string(), None)
        })?;
        let disc = builder.active_disc_mut().ok_or_else(|| {
            RuneError::type_error(
                "period_range must be called inside a differentiate block".to_string(),
                Some(span),
            )
        })?;

        disc.period_range = Some((min as u32, max as u32));
        Ok(Value::Nil)
    }
}

struct SpreadVerb;

impl Verb for SpreadVerb {
    fn name(&self) -> &str {
        "spread"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.is_empty() {
            return Err(RuneError::argument(
                "spread requires 1 argument".to_string(),
                Some(ctx.span),
            ));
        }

        let dist = extract_symbol(&args[0], ctx, "spread", "distribution")?;
        let span = ctx.span;

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("spread requires NeurogenBuilder host".to_string(), None)
        })?;
        let disc = builder.active_disc_mut().ok_or_else(|| {
            RuneError::type_error(
                "spread must be called inside a differentiate block".to_string(),
                Some(span),
            )
        })?;

        disc.spread = Some(dist);
        Ok(Value::Nil)
    }
}

// ---------------------------------------------------------------------------
// Tract verb
// ---------------------------------------------------------------------------

struct WireVerb;

impl Verb for WireVerb {
    fn name(&self) -> &str {
        "wire"
    }

    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        if args.len() < 4 {
            return Err(RuneError::argument(
                "wire requires 4 arguments: name, from, to, type".to_string(),
                Some(ctx.span),
            ));
        }

        let name = extract_symbol(&args[0], ctx, "wire", "name")?;
        let from = extract_symbol(&args[1], ctx, "wire", "from")?;
        let to = extract_symbol(&args[2], ctx, "wire", "to")?;
        let type_sym = extract_symbol(&args[3], ctx, "wire", "type")?;
        let span = ctx.span;

        let tract_type = TractType::from_symbol(&type_sym).ok_or_else(|| {
            RuneError::argument(
                format!(
                    "unknown tract type '{}', expected: association, projection, commissural",
                    type_sym
                ),
                Some(span),
            )
        })?;

        let fiber_count = if args.len() > 4 {
            Some(extract_int(&args[4], "wire", "fiber_count")? as u32)
        } else {
            None
        };

        let builder = ctx.host_mut::<NeurogenBuilder>().map_err(|_| {
            RuneError::type_error("wire requires NeurogenBuilder host".to_string(), None)
        })?;
        builder.tracts.push(TractSpec {
            name,
            from,
            to,
            tract_type,
            fiber_count,
        });

        Ok(Value::Nil)
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

pub struct DevelopModule {
    verbs: Vec<Box<dyn Verb>>,
    constructs: Vec<Box<dyn Construct>>,
}

impl DevelopModule {
    pub fn new() -> Self {
        Self {
            verbs: vec![
                Box::new(ArchetypeVerb),
                Box::new(NeuronCountVerb),
                Box::new(NeighborsVerb),
                Box::new(PhasesVerb),
                Box::new(NearVerb),
                Box::new(TargetVerb),
                Box::new(ThresholdVerb),
                Box::new(PopulationCapVerb),
                Box::new(BindProgramVerb),
                Box::new(PeriodRangeVerb),
                Box::new(SpreadVerb),
                Box::new(WireVerb),
            ],
            constructs: vec![
                Box::new(RegionConstruct),
                Box::new(DifferentiateConstruct),
            ],
        }
    }
}

impl Default for DevelopModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for DevelopModule {
    fn name(&self) -> &str {
        "develop"
    }

    fn version(&self) -> ModuleVersion {
        ModuleVersion::new(1, 0, 0)
    }

    fn verbs(&self) -> &[Box<dyn Verb>] {
        &self.verbs
    }

    fn constructs(&self) -> &[Box<dyn Construct>] {
        &self.constructs
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

    fn sym(s: &str) -> Value {
        Value::String(s.into())
    }

    #[test]
    fn region_construct_lifecycle() {
        let mut builder = NeurogenBuilder::new();
        let construct = RegionConstruct;

        let args = vec![sym("brainstem")];
        let mut ctx = make_ctx(&mut builder);
        let enter_val = construct.enter(&args, &mut ctx).unwrap();
        assert_eq!(enter_val, Value::Integer(0));
        assert!(builder.in_region());

        let mut ctx = make_ctx(&mut builder);
        construct.exit(enter_val, Value::Nil, &mut ctx).unwrap();
        assert!(!builder.in_region());
        assert_eq!(builder.regions.len(), 1);
        assert_eq!(builder.regions[0].name, "brainstem");
    }

    #[test]
    fn region_construct_no_nesting() {
        let mut builder = NeurogenBuilder::new();
        let construct = RegionConstruct;

        let args = vec![sym("a")];
        let mut ctx = make_ctx(&mut builder);
        construct.enter(&args, &mut ctx).unwrap();

        let args2 = vec![sym("b")];
        let mut ctx = make_ctx(&mut builder);
        assert!(construct.enter(&args2, &mut ctx).is_err());
    }

    #[test]
    fn differentiate_lifecycle() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());

        let construct = DifferentiateConstruct;
        let args = vec![sym("osc1")];
        let mut ctx = make_ctx(&mut builder);
        construct.enter(&args, &mut ctx).unwrap();
        assert!(builder.in_disc());

        let mut ctx = make_ctx(&mut builder);
        construct.exit(Value::Nil, Value::Nil, &mut ctx).unwrap();
        assert!(!builder.in_disc());

        let region = builder.active_region_mut().unwrap();
        assert_eq!(region.discs.len(), 1);
    }

    #[test]
    fn differentiate_outside_region_fails() {
        let mut builder = NeurogenBuilder::new();
        let construct = DifferentiateConstruct;

        let args = vec![sym("osc")];
        let mut ctx = make_ctx(&mut builder);
        assert!(construct.enter(&args, &mut ctx).is_err());
    }

    #[test]
    fn archetype_verb() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());

        let verb = ArchetypeVerb;
        let args = vec![sym("brainstem")];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        assert_eq!(
            builder.active_region_mut().unwrap().archetype,
            RegionArchetype::Brainstem
        );
    }

    #[test]
    fn archetype_unknown_fails() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());

        let verb = ArchetypeVerb;
        let args = vec![sym("alien")];
        let mut ctx = make_ctx(&mut builder);
        assert!(verb.call(&args, &mut ctx).is_err());
    }

    #[test]
    fn neuron_count_verb() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());

        let verb = NeuronCountVerb;
        let args = vec![Value::Integer(200)];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        assert_eq!(builder.active_region_mut().unwrap().neuron_count, 200);
    }

    #[test]
    fn target_verb() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());
        builder.begin_disc("d1".to_string());

        let verb = TargetVerb;
        let args = vec![sym("oscillator")];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        assert_eq!(builder.active_disc_mut().unwrap().target, DiscTarget::Oscillator);
    }

    #[test]
    fn near_verb() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());
        builder.begin_disc("d1".to_string());

        let verb = NearVerb;
        let args = vec![sym("respiratory_center")];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        assert_eq!(
            builder.active_disc_mut().unwrap().near_gradient.as_deref(),
            Some("respiratory_center")
        );
    }

    #[test]
    fn threshold_verb() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());
        builder.begin_disc("d1".to_string());

        let verb = ThresholdVerb;
        let args = vec![Value::Integer(25)];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        assert_eq!(builder.active_disc_mut().unwrap().threshold, 25);
    }

    #[test]
    fn population_cap_out_of_range_fails() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());
        builder.begin_disc("d1".to_string());

        let verb = PopulationCapVerb;
        let args = vec![Value::Integer(101)];
        let mut ctx = make_ctx(&mut builder);
        assert!(verb.call(&args, &mut ctx).is_err());
    }

    #[test]
    fn period_range_verb() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());
        builder.begin_disc("d1".to_string());

        let verb = PeriodRangeVerb;
        let args = vec![Value::Integer(500_000), Value::Integer(2_000_000)];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        assert_eq!(
            builder.active_disc_mut().unwrap().period_range,
            Some((500_000, 2_000_000))
        );
    }

    #[test]
    fn wire_verb() {
        let mut builder = NeurogenBuilder::new();

        let verb = WireVerb;
        let args = vec![
            sym("thalamocortical"),
            sym("thalamus"),
            sym("temporal"),
            sym("projection"),
        ];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        assert_eq!(builder.tracts.len(), 1);
        assert_eq!(builder.tracts[0].name, "thalamocortical");
        assert_eq!(builder.tracts[0].tract_type, TractType::Projection);
    }

    #[test]
    fn phases_verb() {
        let mut builder = NeurogenBuilder::new();
        builder.begin_region("test".to_string());

        let verb = PhasesVerb;
        let args = vec![
            Value::Integer(300),
            Value::Integer(1500),
            Value::Integer(800),
            Value::Integer(700),
        ];
        let mut ctx = make_ctx(&mut builder);
        verb.call(&args, &mut ctx).unwrap();

        let phases = builder.active_region_mut().unwrap().phase_durations;
        assert_eq!(phases.genesis, 300);
        assert_eq!(phases.exposure, 1500);
    }

    #[test]
    fn module_metadata() {
        let m = DevelopModule::new();
        assert_eq!(m.name(), "develop");
        assert_eq!(m.verbs().len(), 12);
        assert_eq!(m.constructs().len(), 2);
    }

    #[test]
    fn brainstem_full_spec() {
        let mut builder = NeurogenBuilder::new();

        // Simulate brainstem.rune evaluation
        builder.begin_region("brainstem".to_string());
        builder.active_region_mut().unwrap().archetype = RegionArchetype::Brainstem;
        builder.active_region_mut().unwrap().neuron_count = 200;

        builder.active_region_mut().unwrap().gradients.push(
            super::super::builder::GradientSpec {
                name: "respiratory_center".to_string(),
                strength: 180,
                radius: 30,
            },
        );

        // 3 differentiation discs
        for (name, tgt, thr, cap) in &[
            ("respiratory_osc", DiscTarget::Oscillator, 25i64, 15i64),
            ("autonomic_out", DiscTarget::Motor, 30, 8),
            ("reticular_relay", DiscTarget::Relay, 35, 12),
        ] {
            builder.begin_disc(name.to_string());
            builder.active_disc_mut().unwrap().target = *tgt;
            builder.active_disc_mut().unwrap().threshold = *thr;
            builder.active_disc_mut().unwrap().population_cap = *cap;
            if *tgt == DiscTarget::Oscillator {
                builder.active_disc_mut().unwrap().period_range = Some((500_000, 2_000_000));
                builder.active_disc_mut().unwrap().near_gradient =
                    Some("respiratory_center".to_string());
            }
            builder.end_disc();
        }

        builder.end_region();

        assert_eq!(builder.regions.len(), 1);
        let r = &builder.regions[0];
        assert_eq!(r.name, "brainstem");
        assert_eq!(r.archetype, RegionArchetype::Brainstem);
        assert_eq!(r.neuron_count, 200);
        assert_eq!(r.discs.len(), 3);
        assert_eq!(r.discs[0].target, DiscTarget::Oscillator);
        assert_eq!(r.discs[0].period_range, Some((500_000, 2_000_000)));
    }
}
