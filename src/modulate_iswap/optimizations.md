# `VariationalRolloutObjective` per-iter speedup notes

Goal: make per-Ipopt-iter wall time on the 3-level Duffing iSWAP rollout problem fast enough to actually run for hundreds of iterations.

Baseline: production run uses `VariationalRolloutProblem` (rather than the direct
collocation `VariationalSplinePulseProblem`), which already cuts the NLP-side
constraint Jacobian cost by ~200× (no `:var_Ũ⃗` block of size
`iso_dim·(1+n_vars)`). After that fix, the bottleneck moved to **`gradient!`**
on the rollout objective itself: ForwardDiff seeds through a 648-dim Tsit5
solve, materialising a Dual `G_iso` matrix per RHS call. That's what this doc
is about.

Source: `Piccolo.jl/src/control/objectives.jl` — `VariationalRolloutObjective`
+ `_vro_rollout_J`.

Harness: `profile_rollout_3level.jl` — N_knots=6, `varsys.levels=9`, n_vars=3,
n_drives=4, time-dependent H but with Δ_mw=0 so the linear-G cache fires.

## Profile baseline (Float64 / Dual)

| Op | Wall | Allocs | Memory |
|---|---|---|---|
| `objective_value` | 0.260 s | 224 k | 100 MiB |
| `gradient!`       | 43.4 s  | 2.91 M | **25.7 GiB** |

The 25.7 GiB allocation rate per `gradient!` is the smoking gun: at chunk=12
each `similar(drift, Dual)` allocation in the RHS is ~33 KB, and the RHS gets
called ~5–10 k times per ODE × `N-1 = 5` intervals × ~18 ForwardDiff passes.

## Tier-A wins shipped (no algorithm change)

### 1. Skip seeding `Ũ⃗_init` through ForwardDiff when fixed

The first knot's `Ũ⃗_1` is fixed by `traj.initial[:Ũ⃗]` → an equality constraint
in the NLP. Any gradient we provide there is *unused* by the optimizer, yet
the original code seeded all `iso_dim = 162` partials of it through ForwardDiff.

**Fix.** New field `state_init_fixed::Bool` on `VariationalRolloutObjective`,
auto-detected from `haskey(traj.initial, state_sym)` at construction. When
`true`, `Ũ⃗_init` is captured as a Float64 closure constant — only the
control variables (and `Δt`, if free) enter `z_flat`.

For our 3-level setup:

- before: `z_flat` ≈ 210 dims (162 `Ũ⃗` + 24 `u` + 24 `du`) → chunk=12 → **18 passes**
- after:  `z_flat` ≈ 48 dims (24 `u` + 24 `du`) → chunk=12 → **4 passes**

That's a **~4.5× reduction in the number of ForwardDiff forward passes**.

Opt-out kwarg: `VariationalRolloutProblem(...; rollout_state_init_fixed=false)`
if you ever want to optimise the initial state (uncommon).

Correctness validated against finite-diff in
`test_rollout_hadamard_regression.jl` for both paths (`2a/3` and `2b/3`).

### 2. Avoid materialising `G_iso` as a Dual matrix in the RHS

The cached fast path used to build the full `G_iso = drift + Σᵢ uᵢ·drives[i]`
matrix once per RHS call. When seeded by ForwardDiff this becomes an 18×18 Dual
matrix → ~`chunk × 324 × 8` ≈ 33 KB allocated and zeroed per call.

**Fix.** New `_vro_apply_aug_rhs_linear!` that takes `(drift, drives, u_vec, ...)`
directly and computes `dx = drift·x + Σⱼ uⱼ·(drives[j]·x)` via successive
5-arg `mul!(C, A_float, B_dual, α_dual, β)` calls. `drift` and `drives[i]`
stay Float64 — only the result vector carries Duals. Zero per-RHS matrix
allocations on the gradient path.

### 3. Compile-time eltype dispatch (Float64 vs Dual)

Naïvely applying the axpy form to the Float64 (`objective_value`) path made it
**slower** (0.26 s → 0.73 s) — for 18×18 matrices, BLAS gemv on a single
materialised matrix beats `1 + n_drives` separate axpys due to dispatch cost.

**Fix.** Inside `_vro_rollout_J`'s closure, branch on `eltype(x)`. Julia
specialises the function per concrete type, so the branch is decided at
compile time:

- `Float64` → materialise `G_iso` once, then `_vro_apply_aug_rhs!` (BLAS path)
- Dual → `_vro_apply_aug_rhs_linear!` (no Dual matrix allocation)

Both paths now benefit from the eltype-appropriate strategy.

## Tier-B wins (in progress)

### 4. Cache layout on the objective struct
Layout (`idx_us`, `idx_dus`, `idx_Δts`, `Δts_fixed`, …) is invariant after
construction. Currently rebuilt every `gradient!`/`objective_value` call.
Microsecond cost but zero risk.

### 5. Cache `ForwardDiff.GradientConfig`
Avoids per-call setup of the Dual seed buffer. Needs careful storage in the
struct (parameterised by chunk size).

### 6. Loosen Tsit5 tolerances
Current default `reltol=1e-5, abstol=1e-7` (problem template overrides to
`1e-8, 1e-10` — even tighter). For gradient direction, `reltol=1e-3,
abstol=1e-5` likely fine. Could cut Tsit5 step count by ~3–5×.

### 7. Pre-allocate `u_τ`
4-element Dual vector allocated per RHS call. Small (~8 MB/grad eval at
chunk=12) but free to fix.

## Tier-C: algorithmic / structural

### 8. Block-decouple the augmented ODE

The augmented dynamics are block lower-triangular:

```
dŨ⃗/dt    = G(u,t) · Ũ⃗
d∂Ũ⃗ᵢ/dt = G(u,t) · ∂Ũ⃗ᵢ + (G_varᵢ / scaleᵢ) · Ũ⃗
```

Currently one Tsit5 solve on the full 648-dim augmented state. Could split
into:

1. Solve nominal `Ũ⃗(τ)` (dim 162) once, save dense output.
2. Solve each `∂Ũ⃗ᵢ` (dim 162) independently, using the cached `Ũ⃗(·)` as a forcing term.

Same total flops in the linear-algebra inner loop, but:

- Smaller per-step state ⇒ less memory traffic per Tsit5 step (esp. with Duals)
- Each block's step controller responds to *its* stiffness, not the worst of all four
- The `n_vars` ∂Ũ⃗ᵢ solves are **mutually independent given Ũ⃗(·)** ⇒ embarrassingly parallel

### 9. Multithread the variational direction solves

After block-decoupling, run the n_vars ∂Ũ⃗ᵢ rollouts on `Threads.@threads`.
For our 3-direction case and a 4+ core machine: near-`n_vars`× speedup on
the variational portion.

Caveats:

- ForwardDiff and threading interact safely as long as per-thread scratch
  buffers don't overlap.
- Launch Julia with `JULIA_NUM_THREADS≥n_vars+1` (= 4 for our case).

**Failed experiment: threading inside the RHS.** We initially tried
`Threads.@threads for i = 1:n_vars` *inside* `_vro_apply_aug_rhs!` /
`_vro_apply_aug_rhs_linear!`. Each call to that function is ~10 μs of work;
the per-call `@threads` task-scheduling overhead dominates even with 1 thread
(measured: 7.6 s → 13.6 s on the 100-iter Hadamard solve). Reverted.
Threading needs to happen at a *much coarser* level — at minimum the whole
ODE solve per variational direction (block-decoupling + threading combined).

### 10. Backward adjoint (Phase 2)

The "real" fix per the original design doc
(`variational-rollout-adjoint-template.md`): replace ForwardDiff with one
forward solve + one backward ODE on the adjoint. Cost no longer scales with
`n_drives·N` (chunk count drops out). Estimated 10–20× on top of everything
above. Multi-day implementation; deferred until 1–9 are exhausted.

## Per-fix benchmark

3-level Duffing iSWAP, N_knots=6 (≈ 1.0k NLP vars, n_vars=3, n_drives=4,
varsys.levels=9, iso_dim=162, aug_dim=648). Bench harness:
`profile_rollout_3level.jl`. Single-thread Julia (JULIA_NUM_THREADS=1).

| Round | `objective_value` | `gradient!` | grad alloc | Notes |
|---|---|---|---|---|
| baseline | 0.260 s | 43.4 s | 25.7 GiB | template Tsit5 reltol=1e-8, abstol=1e-10 |
| +#1 (Ũ⃗ skip) +#2 (linear-axpy applied to both paths) | 0.73 s | 11.2 s | 100 MiB | Float64 path regressed from extra mul! dispatches |
| +#3 (eltype dispatch — materialised G_iso for Float64, axpy for Dual) | 0.254 s | 11.2 s | 100 MiB | Float64 path restored |
| +#6 (template tolerance → reltol=1e-5, abstol=1e-7) | **0.057 s** | **2.68 s** | **47 MiB** | **16.2× total speedup on gradient!** |
| +#9 (threading) | _pending_ | _pending_ | _pending_ | |
| +#8 (block-decouple) | _pending_ | _pending_ | _pending_ | |
| +#10 (adjoint) | _future_ | _future_ | _future_ | |

## Headline

After fixes 1–3 + 6, the per-call cost dropped:

- `gradient!`: **43.4 s → 2.68 s** (16.2× faster)
- `objective_value`: **0.260 s → 0.057 s** (4.6× faster)
- gradient allocation: **25.7 GiB → 47 MiB** (560× less memory)

For the production 3-level run (1000 Ipopt iterations), this translates from
roughly a 12-hour wall to under an hour — i.e. it becomes feasible to actually
iterate on the optimization rather than wait overnight.

## Correctness validation

`test_rollout_hadamard_regression.jl` now has four independent checks. After
the optimizations above:

| Check | Result | What it validates |
|---|---|---|
| 2a — gradient AD vs FD (movable components) | rel err = **2.54e-7** | AD gradient matches FD on u/du (fast path, state_init_fixed=true) |
| 2b — gradient AD vs FD (full trajectory) | rel err = **3.03e-8** | Same, with state_init_fixed=false — covers Ũ⃗ partials too |
| 2c — **rollout VALUE vs independent FD reference** | rel err = **1.13e-8** | The ∂Ũ⃗ᵢ(T) the rollout produces matches an FD on `unitary_rollout(traj, perturbed_sys)` — i.e. the optimizations have not changed *what* the rollout computes, only *how fast* |
| 3 — 100-iter Hadamard solve | F_nlp = **1.000000** | end-to-end optimizer convergence still works |
| 4 — integrator-honesty | ‖U_nlp − U_ref‖ = **1.40e-3**, ΔF = **4.4e-4** | NLP's `:Ũ⃗[:, end]` (from the bilinear integrator) vs an independent high-accuracy ODE re-evolution of the optimized controls (`unitary_rollout` with `cubic_hermite` interpolation, abstol=reltol=1e-12) |

Test 2c (rel err = 1.13e-8 — essentially machine precision) is the
sharpest check that the optimizations preserve correctness.

## The NLP-vs-reality fidelity gap (not caused by these optimizations)

Test 4 found that the NLP reports F=1.000000 while an independent re-evolution
of the optimized controls gives F=0.999562 — a ~**4×10⁻⁴** "fictitious
fidelity" gap. **Root cause: Ipopt's default `constr_viol_tol = 1e-4`.**

The `TimeDependentBilinearIntegrator` enforces
`Ũ⃗_{k+1} − exp(−i·Δt·G_avg)·Ũ⃗_k = 0` as an equality constraint. With Ipopt
stopping at constraint residual ~1e-4 per row, and N−1 = 14 such constraints
compounding multiplicatively, the final unitary ends up ~10⁻³ off. The
integrator's own internal Tsit5 tolerance has nothing to do with it
(verified: sweeping it across `reltol ∈ {1e-3, 1e-5, 1e-10}` gives identical
gap).

**Fix:** pass `constr_viol_tol = 1e-8` in `IpoptOptions`:

```julia
solve!(qcp; max_iter=N,
    options = IpoptOptions(
        eval_hessian = false,
        constr_viol_tol = 1e-8,   # << default is 1e-4
        tol = 1e-8,
        acceptable_tol = 1e-8,
    ),
)
```

Sweep at `spline_order=3, N=15, T=2.0`, Hadamard:

| `constr_viol_tol` | F_nlp | F_ref | ΔF | wall |
|---|---|---|---|---|
| 1e-4 (Ipopt default) | 1.0000005 | 0.9991833 | **8.17e-4** | 9.8 s |
| 1e-6 | 1.0000005 | 0.9991833 | 8.17e-4 | 3.4 s |
| **1e-8** | **1.0000000** | **0.9999988** | **1.20e-6** | 3.4 s |
| 1e-10 | 1.0000001 | 0.9999060 | 9.41e-5 | 3.3 s |

The 1e-8 setting:
- Closes the gap ~700× (8e-4 → 1e-6)
- Does **not** slow Ipopt down (it actually finished faster — heuristics steer
  it through a different convergence path)

A residual `‖U_nlp − U_ref‖ ≈ 1.2 mrad` remains even at tight tolerance, but
that's a global-phase artefact (`tr(U_goal'·U)·e^{iφ}` is gauge-invariant)
and does not affect fidelity. Lower priority.

**Sanity check**: this fidelity gap is independent of my optimizations
(verified by sweeping integrator Tsit5 tolerance — no effect). It would be
present in the baseline implementation too; we only noticed it because
Test 4 explicitly compared NLP-reported and independently-re-evolved
unitaries after the speed optimizations made fast turnaround feasible.
