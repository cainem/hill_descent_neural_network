# hill_descent_lib: Integration Experience & Improvement Suggestions

## Context
This document captures the experience of integrating `hill_descent_lib` v0.1.0 into a neural network training system as an alternative to gradient-based backpropagation. The library was used to optimize ~50,890 parameters (784-64-10 network architecture).

## Overall Assessment
The library works well for its intended purpose, but the API has some confusing aspects around parameter passing that caused significant friction during integration. The core genetic algorithm functionality is solid once you get past the initial setup hurdles.

---

## Pain Points & Suggested Improvements

### 1. **Confusing Parameter Signatures for `training_run` and `get_best_organism`**

#### The Problem
The function signatures use `&[[f64]]` (slice of slices), but the actual usage pattern is unclear from documentation:

```rust
pub fn training_run(&mut self, inputs: &[[f64]], known_outputs: &[[f64]]) -> bool
pub fn get_best_organism(&mut self, training_data: &[&[f64]], known_outputs: &[&[f64]]) -> Arc<Organism>
```

**What caused confusion:**
1. Examples show `&[0.0]` for SingleValuedFunction but signature shows `&[[f64]]`
2. Different slice types between `training_run` (`&[[f64]]`) and `get_best_organism` (`&[&[f64]]`)
3. Error messages like "known_outputs must not be empty" appear even when passing `&[0.0]`
4. "Mismatch: input length 0 != output length 1" when trying `&[]` vs `&[0.0]`
5. "Training data cannot be empty" when using documented example `&[&[]]`

**Trial and error required:**
```rust
// These all failed with different errors:
world.training_run(&[], &[0.0])              // Wrong - type mismatch
world.training_run(&[], &[[0.0]])            // Wrong - type mismatch  
world.get_best_organism(&[&[]], &[&[]])      // Panics: "Training data cannot be empty"
world.get_best_organism(&[], &[0.0])         // Wrong - type mismatch

// Eventually discovered these work:
world.training_run(&[], &[0.0])              // This actually works!
let dummy = [0.0];
let floor = [0.0];
world.get_best_organism(&[&dummy], &[&floor]) // This works!
```

#### Suggested Improvements

**Option A: Simplify the API with Enums**
```rust
pub enum TrainingData<'a> {
    /// For SingleValuedFunction optimization (no external data needed)
    None { floor_value: f64 },
    /// For WorldFunction with external training data
    Supervised { 
        inputs: &'a [[f64]], 
        outputs: &'a [[f64]] 
    },
}

// Much clearer usage:
world.training_run(TrainingData::None { floor_value: 0.0 });
world.get_best_organism(TrainingData::None { floor_value: 0.0 });
```

**Option B: Separate Methods**
```rust
// For the common case
pub fn training_run_simple(&mut self, floor_value: f64) -> bool;
pub fn get_best_organism_simple(&mut self, floor_value: f64) -> Arc<Organism>;

// For advanced case
pub fn training_run_supervised(&mut self, inputs: &[[f64]], outputs: &[[f64]]) -> bool;
pub fn get_best_organism_supervised(&mut self, inputs: &[[f64]], outputs: &[[f64]]) -> Arc<Organism>;
```

**Option C: Better Documentation**
At minimum, add explicit examples in doc comments:
```rust
/// # Examples for SingleValuedFunction
/// ```
/// // CORRECT way - pass empty inputs, single-element known_outputs
/// world.training_run(&[], &[0.0]); // Note: &[0.0] not &[[0.0]]
/// 
/// // CORRECT way for get_best_organism - needs at least one element
/// let dummy_input = [0.0];
/// let floor_value = [0.0];  
/// let best = world.get_best_organism(&[&dummy_input], &[&floor_value]);
/// ```
```

---

### 2. **Inconsistent Type Requirements Between Similar Functions**

#### The Problem
`training_run` and `get_best_organism` have different expectations:
- `training_run(&[], &[0.0])` works fine with empty inputs
- `get_best_organism(&[], ...)` panics with "Training data cannot be empty"

This inconsistency is confusing - why does one function accept empty inputs while the other requires a dummy value?

#### Suggested Improvement
**Make the behavior consistent.** Either:
1. Both should accept truly empty data for SingleValuedFunction, OR
2. Both should require dummy data (document why)

If there's a technical reason `get_best_organism` needs data when `training_run` doesn't, **document it prominently** in the function docs.

---

### 3. **Unclear Initial Best Score**

#### The Problem
The initial best score displays as:
```
Initial best score: 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000
```

This is `f64::MAX` - technically correct but extremely confusing in output.

#### Suggested Improvement
```rust
// In the display logic:
if score >= f64::MAX * 0.99999 {
    println!("Initial best score: <not yet evaluated>");
    // or
    println!("Initial best score: f64::MAX (not yet evaluated)");
} else {
    println!("Initial best score: {:.6}", score);
}
```

---

### 4. **Missing Convenience Method for Parameter Extraction**

#### The Problem
Getting the best parameters requires this chain:
```rust
let best_organism = world.get_best_organism(&[&dummy], &[&floor]);
let best_params = best_organism.phenotype().expression_problem_values();
```

This is verbose and exposes internal structure (`phenotype`, `expression_problem_values`) that users shouldn't need to know about.

#### Suggested Improvement
```rust
impl World {
    /// Returns the parameter values of the best organism found so far.
    /// 
    /// This is a convenience method that combines get_best_organism() and 
    /// parameter extraction. Use this when you just need the parameter values.
    pub fn get_best_params(&self) -> &[f64] {
        // Returns parameters without needing another training run
    }
}

// Usage:
let best_params = world.get_best_params(); // Much simpler!
```

**Note:** `get_best_organism` runs an additional training epoch, which seems unnecessary if you just want to extract the current best solution. Consider adding a non-mutating accessor.

---

### 5. **Documentation Gaps for SingleValuedFunction Use Case**

#### The Problem
The documentation examples are excellent for mathematical optimization (Sphere, Rosenbrock, etc.), but don't cover the common machine learning use case where:
- You have training data but don't want to pass it all every epoch (performance)
- Your fitness function internally manages data subsets
- You're optimizing thousands of parameters

#### Suggested Improvement
Add a comprehensive example:

```rust
/// # Example: Machine Learning Parameter Optimization
/// 
/// This example shows how to optimize a large parameter space (like neural network
/// weights) where the fitness function internally manages training data access.
/// 
/// ```
/// use hill_descent_lib::{setup_world, GlobalConstants, SingleValuedFunction};
/// use std::sync::Arc;
/// 
/// struct ModelFitness {
///     training_data: Arc<Vec<Vec<f64>>>,
///     subset_size: usize,
/// }
/// 
/// impl SingleValuedFunction for ModelFitness {
///     fn single_run(&self, params: &[f64]) -> f64 {
///         // Randomly sample training data internally
///         // Evaluate model with these params on subset
///         // Return validation loss
///         // ... implementation ...
///         0.0
///     }
/// }
/// 
/// // Optimize 10,000+ parameters
/// let param_count = 10000;
/// let bounds = vec![-1.0..=1.0; param_count];
/// let constants = GlobalConstants::new(500, 50);
/// 
/// let fitness = ModelFitness { /* ... */ };
/// let mut world = setup_world(&bounds, constants, Box::new(fitness));
/// 
/// // Training loop for large-scale optimization
/// for generation in 0..100 {
///     world.training_run(&[], &[0.0]); // Empty inputs for SingleValuedFunction
///     
///     if generation % 10 == 0 {
///         println!("Generation {}: Loss = {:.6}", generation, world.get_best_score());
///     }
/// }
/// 
/// // Extract best parameters (requires dummy data currently)
/// let dummy = [0.0];
/// let best = world.get_best_organism(&[&dummy], &[&dummy]);
/// let best_params = best.phenotype().expression_problem_values();
/// ```

---

### 6. **Type Ergonomics: Slice of Slices**

#### The Problem
The `&[[f64]]` and `&[&[f64]]` types are difficult to construct correctly in Rust, especially for users who aren't Rust experts. The differences between:
- `&[f64]` - slice of f64
- `&[[f64]]` - slice of slices (Vec<Vec<f64>>)
- `&[&[f64]]` - slice of references to slices

...are subtle and the compiler errors aren't always helpful.

#### Suggested Improvement

**Option A: Accept more flexible types**
```rust
// Use impl trait to accept more types
pub fn training_run<I, O>(&mut self, inputs: I, known_outputs: O) -> bool
where
    I: AsRef<[impl AsRef<[f64]>]>,
    O: AsRef<[impl AsRef<[f64]>]>,
{
    // Can now accept:
    // - &[]
    // - &[0.0]
    // - &Vec<Vec<f64>>
    // - &[&[f64]]
    // etc.
}
```

**Option B: Provide builder helpers**
```rust
pub struct TrainingRunBuilder<'a> {
    world: &'a mut World,
}

impl<'a> TrainingRunBuilder<'a> {
    pub fn with_no_data(self) -> RunResult {
        self.world.training_run_impl(&[], &[])
    }
    
    pub fn with_floor_value(self, floor: f64) -> RunResult {
        self.world.training_run_impl(&[], &[floor])
    }
    
    pub fn with_supervised_data(self, inputs: &[[f64]], outputs: &[[f64]]) -> RunResult {
        self.world.training_run_impl(inputs, outputs)
    }
}

// Usage:
world.train().with_floor_value(0.0);
```

---

### 7. **Missing Guidance on Scaling**

#### The Problem
Documentation mentions "tested with 100+ dimensions" but doesn't provide guidance on:
- Recommended population sizes for different parameter counts
- Expected performance characteristics (evaluations/sec vs parameter count)
- When the algorithm is likely to struggle
- Memory usage scaling

For a 50,890-parameter problem, it's unclear if population=500 is too small, too large, or appropriate.

#### Suggested Improvement
Add a scaling guide section to the documentation:

```markdown
## Scaling Guidelines

### Parameter Count vs Population Size
- **< 10 dimensions**: 50-100 population
- **10-100 dimensions**: 200-500 population  
- **100-1000 dimensions**: 500-2000 population
- **1000-10000 dimensions**: 1000-5000 population
- **> 10000 dimensions**: 2000+ population (performance will degrade)

### Performance Expectations
- Evaluations are parallelized across CPU cores
- Typical: 10,000-100,000 evaluations/second (depends on fitness function)
- Memory: ~(population_size * parameter_count * 8) bytes minimum

### When to Use vs Not Use
**Good fit:**
- Non-differentiable fitness functions
- Discrete or combinatorial parameter spaces
- Multi-modal optimization (many local optima)
- Parameters < 5000

**Poor fit:**
- Smooth, convex problems (use gradient descent)
- Parameters > 50,000 (very slow)
- Real-time optimization (too many evaluations needed)
```

---

## What Worked Well

Despite the friction points, several aspects of the library were excellent:

1. **Core Algorithm**: Once configured correctly, it works reliably
2. **Parallelization**: Automatic parallel evaluation is transparent and effective
3. **Type Safety**: The type system prevents many logical errors (even if types are confusing)
4. **Deterministic**: Seeded RNG makes results reproducible
5. **Performance**: Evaluation throughput is good (170k+ evals/sec in our tests)
6. **Spatial Regions**: The adaptive region concept is clever and differentiates this from basic GAs

---

## Priority Ranking

If addressing these issues, I'd suggest this priority order:

1. **HIGH**: Fix `training_run`/`get_best_organism` parameter confusion (Issue #1)
2. **HIGH**: Add convenience method for parameter extraction (Issue #4)
3. **MEDIUM**: Document SingleValuedFunction ML use case (Issue #5)
4. **MEDIUM**: Make behavior consistent between functions (Issue #2)
5. **MEDIUM**: Add scaling guidance (Issue #7)
6. **LOW**: Improve initial score display (Issue #3)
7. **LOW**: Improve type ergonomics (Issue #6) - nice to have but not critical

---

## Conclusion

`hill_descent_lib` is a solid genetic algorithm implementation with good performance characteristics. The main barrier to adoption is API confusion around the `training_run`/`get_best_organism` parameter passing, especially for the common SingleValuedFunction use case.

With clearer documentation or simplified API surface for the simple case (Option B from Issue #1), this would be much more approachable for users trying to optimize non-trivial parameter spaces.

The library successfully optimized a 50,890-parameter neural network in our testing, demonstrating it can handle real-world problems beyond simple mathematical functions.
