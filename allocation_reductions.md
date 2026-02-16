# Allocation Reduction Plan for `hill_descent_lib`

## Goal

Reduce per-generation heap allocations from **~417 MB** to **~15 MB** (28× reduction) to eliminate memory allocator contention as the primary bottleneck limiting parallel CPU utilisation.

## Current State (baseline)

Measured with a counting allocator on the neural_network project (784-16-10 architecture, 500 population, 250 subset):

| Metric | Value |
|--------|-------|
| Per-generation allocation | ~417 MB |
| Per-generation allocations (count) | ~3,959 |
| CPU utilisation | ~50% (of 24 logical processors) |
| Speedup vs single-threaded | 4.8× |

## Benchmarking Strategy

### Before starting: establish baselines

Run **both** benchmarks and record results before any code changes:

1. **`hill_descent_lib` benchmark** (measures library-level epoch cost):
   ```powershell
   cd C:\Users\mickc\dev\hill_descent
   cargo bench -p hill_descent_lib
   ```
   This runs `hundred_parameter_benchmark` (100 params, 500 pop, 20 regions).

2. **Neural network benchmark** (measures real-world MNIST workload):
   ```powershell
   cd C:\Users\mickc\dev\hill_descent_neural_network
   cargo bench
   ```
   This runs `genetic_training_784_16_10_40gen` (12,730 params, 500 pop, 40 generations).

### After each change

1. Run `cargo test` in `hill_descent_lib` — all tests must pass
2. Run `cargo clippy` in `hill_descent_lib` — zero warnings
3. Run `cargo bench -p hill_descent_lib` — compare to baseline and previous step
4. Update `hill_descent_lib` version, then in `hill_descent_neural_network`:
   - Update `Cargo.toml` to point to local path: `hill_descent_lib = { path = "..." }`
   - Run `cargo test` — all tests must pass
   - Run `cargo bench` — compare to baseline and previous step
5. Record results in the table at the bottom of this document

### Decision rule

If a change does **not** improve the `hill_descent_lib` benchmark time by at least **2%**, revert it. The 100-param benchmark may not show gains for changes that scale with parameter count — in that case, use the neural network benchmark (12,730 params) as the deciding measurement.

---

## Changes (in priority order)

### Change 1: Buffer-pooled `Gamete::reproduce`

**Estimated saving: ~350 MB/generation (84% of total)**

#### Problem

`Gamete::reproduce` in `hill_descent_lib/src/gamete/reproduce.rs` allocates two new `Vec<Locus>` with `Vec::with_capacity(len)` on every call. Each `Locus` is ~72 bytes. With ~250 organism pairs doing 2 meioses each (4 reproduce calls per pair), this produces ~1,000 new `Vec<Locus>` of length 12,730 per generation.

```
1,000 vecs × 12,730 loci × 72 bytes = ~916 MB allocated (then ~half is freed as parents die)
```

#### Current code

```rust
// gamete/reproduce.rs
pub fn reproduce<R: Rng>(
    parent1: &Gamete,
    parent2: &Gamete,
    crossovers: usize,
    rng: &mut R,
    sys: &SystemParameters,
) -> (Gamete, Gamete) {
    // ...
    let mut offspring1 = Vec::with_capacity(len);
    let mut offspring2 = Vec::with_capacity(len);
    for i in 0..len {
        // ... offspring1.push(...); offspring2.push(...);
    }
    (Gamete::new(offspring1), Gamete::new(offspring2))
}
```

#### Approach

Modify `Gamete::reproduce` to accept pre-allocated output buffers instead of allocating new ones. The buffers are cleared and refilled each call.

**Option A — Pass mutable Gametes:**

```rust
pub fn reproduce_into<R: Rng>(
    parent1: &Gamete,
    parent2: &Gamete,
    offspring1: &mut Gamete,
    offspring2: &mut Gamete,
    crossovers: usize,
    rng: &mut R,
    sys: &SystemParameters,
) {
    offspring1.loci.clear();
    offspring2.loci.clear();
    // ... same loop, but push into existing vecs
}
```

**Option B — Thread-local buffer pool:**

Keep the existing signature but use `thread_local!` to cache and reuse the `Vec<Locus>` allocations internally. This avoids changing the public API.

```rust
thread_local! {
    static BUFFER1: RefCell<Vec<Locus>> = RefCell::new(Vec::new());
    static BUFFER2: RefCell<Vec<Locus>> = RefCell::new(Vec::new());
}

pub fn reproduce<R: Rng>(...) -> (Gamete, Gamete) {
    BUFFER1.with(|b1| BUFFER2.with(|b2| {
        let mut buf1 = b1.borrow_mut();
        let mut buf2 = b2.borrow_mut();
        buf1.clear();
        buf2.clear();
        // ... push into buf1, buf2 ...
        (Gamete::new(std::mem::take(&mut *buf1)),
         Gamete::new(std::mem::take(&mut *buf2)))
    }))
}
```

Note: Option B still creates new `Gamete` wrappers but reuses the backing `Vec` storage. The `std::mem::take` swaps in an empty vec (no alloc) and gives the filled one to `Gamete::new`. Next call, the `clear()` resets the taken-back empty vec (still no alloc because capacity is 0), but the thread-local retains... actually this doesn't work as intended.

**Recommended: Option A** — it's simpler, makes the reuse explicit, and avoids thread-local complexity. The callers in `sexual_reproduction.rs` and `perform_sexual_reproduction.rs` would need updating to maintain reusable `Gamete` buffers (either thread-local at that level, or passed through the call chain).

#### Files to modify

- `hill_descent_lib/src/gamete/mod.rs` — add `clear()` method or make `loci` accessible for clearing; or add `reproduce_into`
- `hill_descent_lib/src/gamete/reproduce.rs` — change allocation pattern
- `hill_descent_lib/src/phenotype/sexual_reproduction.rs` — use new API
- `hill_descent_lib/src/world/regions/region/perform_sexual_reproduction.rs` — thread-local buffers for reuse
- Tests in `reproduce.rs` — update to use new API

#### Verification

- All existing tests pass
- Benchmark shows reduction in epoch time
- Neural network benchmark confirms gains at scale (12,730 params)

---

### Change 2: Buffer-pooled `compute_expressed`

**Estimated saving: ~50 MB/generation (12% of total)**

#### Problem

`compute_expressed` in `hill_descent_lib/src/phenotype/compute_expressed.rs` allocates a new `Vec<f64>` with `Vec::with_capacity(loci_count)` for every new `Phenotype`. With ~500 offspring per generation at 12,730 parameters:

```
500 × 12,730 × 8 bytes = ~50 MB/generation
```

#### Current code

```rust
// phenotype/compute_expressed.rs
pub(super) fn compute_expressed<R: Rng>(g1: &Gamete, g2: &Gamete, rng: &mut R) -> Vec<f64> {
    let mut result = Vec::with_capacity(loci1.len());
    for (l1, l2) in loci1.iter().zip(loci2.iter()) {
        // ... dominance logic ...
        result.push(value);
    }
    result
}
```

The result is stored in `Phenotype.expressed` and used for:
- `expression_problem_values()` — returns `&expressed[NUM_SYSTEM_PARAMETERS..]`
- `expressed_values()` — returns `&expressed`
- Region key computation

#### Approach

Change `compute_expressed` to write into a caller-provided `&mut Vec<f64>` buffer instead of returning a new `Vec<f64>`.

```rust
pub(super) fn compute_expressed_into<R: Rng>(
    g1: &Gamete,
    g2: &Gamete,
    rng: &mut R,
    result: &mut Vec<f64>,
) {
    result.clear();
    // ... same loop, push into result ...
}
```

Since `Phenotype` stores the `expressed` field, the real savings come at the `Phenotype::new` call site. Each new phenotype still needs its own `Vec<f64>`, but the buffer can be pre-allocated once and then moved via `std::mem::replace` — so only the first allocation per thread actually hits the allocator.

Alternatively, store `expressed` as `Arc<Vec<f64>>` or use a pool, but this adds complexity. The simplest approach:

1. In `Phenotype::new`, accept an optional pre-sized `Vec<f64>` to reuse
2. Or, use `std::mem::take` from the old phenotype's `expressed` field before it dies (requires restructuring the lifecycle)

**Note:** This change is less impactful than Change 1 and may require more design thought. If Change 1 already brings CPU utilisation above 80%, consider whether the complexity is justified.

#### Files to modify

- `hill_descent_lib/src/phenotype/compute_expressed.rs` — accept `&mut Vec<f64>` buffer
- `hill_descent_lib/src/phenotype/mod.rs` — `Phenotype::new` to handle buffer reuse
- `hill_descent_lib/src/phenotype/sexual_reproduction.rs` — pass buffers
- Tests in `compute_expressed.rs`

#### Verification

- All existing tests pass
- Benchmark shows measurable improvement beyond Change 1

---

### Change 3: Return `f64` from `SingleValuedFunction` blanket impl

**Estimated saving: ~500 small allocations/generation (minor bytes, reduces allocator calls)**

#### Problem

The `WorldFunction` trait returns `Vec<f64>`:

```rust
// world/world_function.rs
pub trait WorldFunction: Debug + Sync {
    fn run(&self, phenotype_expressed_values: &[f64], inputs: &[f64]) -> Vec<f64>;
}
```

The blanket impl for `SingleValuedFunction` wraps the single `f64` result in a `vec![result]`:

```rust
fn run(&self, phenotype_expressed_values: &[f64], _inputs: &[f64]) -> Vec<f64> {
    let score = self.single_run(phenotype_expressed_values);
    vec![score]
}
```

This creates ~500 tiny heap allocations per generation (one per organism evaluation). Each is only 8 bytes but still requires allocator bookkeeping.

#### Approach

**Option A — SmallVec**: Replace `Vec<f64>` return with `SmallVec<[f64; 1]>` so single-valued returns stay on the stack. This avoids heap allocation for the common case.

**Option B — Separate scalar path**: Add a `run_scalar()` method with a default implementation that calls `run()` and takes the first element. Override it in the `SingleValuedFunction` blanket impl to return the `f64` directly. The organism evaluation code checks which path to use.

**Option C — Store score directly**: Since the organism only cares about a single score value, modify the evaluation pipeline to call `single_run` directly when the world function is a `SingleValuedFunction`, storing the `f64` without wrapping.

**Recommended: Option A** — least disruptive, single type change.

**Note:** This is a **breaking API change** if external code depends on the `Vec<f64>` return type. Consider whether consumers exist beyond this project.

#### Files to modify

- `hill_descent_lib/Cargo.toml` — add `smallvec` dependency (if Option A)
- `hill_descent_lib/src/world/world_function.rs` — change return type
- `hill_descent_lib/src/world/single_valued_function.rs` — update blanket impl
- All callers of `WorldFunction::run` — update to handle new return type
- `hill_descent_lib/src/world/organisms/run_all.rs` — where results are consumed

#### Verification

- All existing tests pass
- Benchmark shows reduced allocation count (may not show time improvement on 100-param benchmark; check neural network benchmark)

---

### Change 4: In-place `Locus::mutate`

**Estimated saving: improved cache performance (no allocation reduction — Locus is stack-sized)**

#### Problem

`Locus::mutate()` and `mutate_unbound()` each return a new `Locus` by value. `Locus` is a stack type (no heap allocation), so this doesn't cause allocator pressure. However, the current pattern:

```rust
offspring1.push(parent1.loci()[i].mutate(rng, &dists));
```

creates a temporary `Locus` on the stack, then copies it into the `Vec`. An in-place mutation variant could write directly into a destination slot, avoiding the copy.

#### Approach

Add `mutate_into` and `mutate_unbound_into` methods:

```rust
pub fn mutate_into<R: Rng>(&self, rng: &mut R, dists: &MutationDistributions, dest: &mut Locus) {
    // Write directly into dest fields instead of creating a new Locus
}
```

**Note:** This is the lowest-priority change. The compiler may already optimise the copy away. Only proceed if profiling shows the mutation loop as a hotspot after Changes 1-3.

#### Files to modify

- `hill_descent_lib/src/locus/mutate.rs` — add `mutate_into` variants
- `hill_descent_lib/src/gamete/reproduce.rs` — use in-place mutation (requires indexing into pre-allocated buffer from Change 1)
- Tests in `mutate.rs`

#### Verification

- All existing tests pass
- Benchmark shows measurable improvement beyond Changes 1-3

---

## Benchmark Results Tracking

Record results here after each change. Use the median time from criterion.

### `hill_descent_lib` benchmark: `hill_descent_train_epoch_100d`

| State | Median time | vs Baseline | Allocations (if measured) |
|-------|-------------|-------------|---------------------------|
| Baseline (before any changes) | | — | |
| After Change 1 (Gamete buffers) | | | |
| After Change 2 (compute_expressed) | | | |
| After Change 3 (scalar return) | | | |
| After Change 4 (in-place mutate) | | | |

### Neural network benchmark: `genetic_training_784_16_10_40gen`

| State | Median time | vs Baseline | Notes |
|-------|-------------|-------------|-------|
| Baseline (before any changes) | | — | |
| After Change 1 (Gamete buffers) | | | |
| After Change 2 (compute_expressed) | | | |
| After Change 3 (scalar return) | | | |
| After Change 4 (in-place mutate) | | | |

### CPU utilisation (Task Manager during `mnist_comparison`)

| State | CPU % | Evals/sec |
|-------|-------|-----------|
| Baseline | ~50% | ~295 |
| After Change 1 | | |
| After Change 2 | | |
| After Change 3 | | |
| After Change 4 | | |

---

## Setup: using local `hill_descent_lib` for benchmarking

To benchmark against your local changes, temporarily point the dependency to the local path:

```toml
# In hill_descent_neural_network/Cargo.toml
# Replace:
#   hill_descent_lib = "0.3.0"
# With:
hill_descent_lib = { path = "../hill_descent/hill_descent_lib" }
```

Remember to revert this before committing to the neural network repo.

---

## Summary

| Change | Est. saving | Complexity | Confidence |
|--------|------------|------------|------------|
| 1. Gamete buffer reuse | ~350 MB/gen | Medium | High |
| 2. compute_expressed buffer | ~50 MB/gen | Medium | Medium |
| 3. Scalar return path | ~500 allocs/gen | Low | Low-Medium |
| 4. In-place mutate | Cache perf only | Low | Low |

**Stop early if CPU utilisation exceeds 80% after any step.** The remaining changes add complexity with diminishing returns.
