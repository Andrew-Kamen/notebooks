# `VariationalRolloutProblem` — indirect/adjoint formulation

## Goal

Cut 3-level Duffing optimization wall time by **10-20×** by removing `:var_Ũ⃗` from the NLP and computing its terminal value (and the gradient of `‖∂Ũ⃗(T)‖²` w.r.t. controls) via forward rollout + backward adjoint solve.

Equivalent in spirit to GRAPE/Krotov, but using the variational adjoint as the robustness objective. Keeps everything else about Piccolo's problem assembly (constraints, regularizers, fidelity objective, free-phase, embedded operators) untouched.

## Why this works

Per the scaling doc: per-iter matrix-exp cost is `O((iso_dim·(1+n_vars))³)` because `var_G` is a full `S × S` matrix. By rolling out instead, the variational state never enters the NLP — Ipopt only sees `(:Ũ⃗, :u, :du, globals)`. The forward rollout + adjoint backsolve is `2 × O(iso_dim²·N_steps)` per iter, replacing `N_knots × O(iso_dim³·(1+n_vars)³)` matrix-exps. For `n_vars = 3` this is the difference between minutes and hours per opt.

## Recommended: KEEP cubic Hermite splines

Reasoning:

| | Splines (cubic Hermite) | Piecewise-constant (classic GRAPE) |
|---|---|---|
| Hardware friendliness | C¹ smooth, naturally bandwidth-limited | Step functions, infinite BW until filtered |
| `R_du`/`R_ddu` knobs | Already wired in | Would need re-wiring on `:u` only |
| ODE accuracy | Tsit5 follows smooth spline → high-order | Tsit5 wastes effort on step discontinuities |
| Adjoint complexity | One extra linear basis pass per step (cheap) | Cleanest |
| Consistency with existing templates | Same trajectory layout (`:u, :du`) | Different trajectory layout |
| Filter distortion in post | Minimal — already smooth | Worse — step → filtered shape differs |

Splines win 5/6. The "adjoint complexity" cost is one extra `4 × n_drives` linear basis evaluation per ODE step — trivial. Adopt cubic Hermite throughout. `LinearSplinePulse` should also work but is a follow-up.

## Architecture (phased)

### Phase 1 — Indirect only on `:var_Ũ⃗`, keep `:Ũ⃗` direct (MVP)

This is the safe first step. The NLP still has `:Ũ⃗` as a decision variable with the existing `TimeDependentBilinearIntegrator` constraint — same machinery you trust. Only `:var_Ũ⃗` becomes indirect.

NLP variables per knot: `(:Ũ⃗, :u, :du, :t, :Δt)` plus globals. Removed: `:var_Ũ⃗` block of dim `iso_dim · (1 + n_vars)`.

### Phase 2 — Full GRAPE: roll out `:Ũ⃗` too

Optional follow-up. Decision vars become just `(:u, :du, :t, :Δt, globals)`. Fidelity objective rolls out `Ũ⃗(T)` from controls. Maximum speedup but bigger change to fidelity-objective code paths.

This plan covers Phase 1. Phase 2 is a follow-up after Phase 1 is benchmarked.

## New types

### `VariationalRolloutObjective <: AbstractObjective`

Defined in `Piccolo.jl/src/control/objectives.jl` (new file or new section). Constructor:

```julia
VariationalRolloutObjective(
    varsys::VariationalQuantumSystem,
    traj::NamedTrajectory;
    Q_r::Float64 = 1.0,
    state_sym::Symbol = :Ũ⃗,
    control_sym::Symbol = :u,
    du_sym::Union{Symbol,Nothing} = :du,
    spline_order::Int = 3,
    variational_scales::AbstractVector{Float64} = fill(1.0, length(varsys.G_vars)),
    solve_kwargs = (; reltol = 1e-8, abstol = 1e-10),
)
```

Stores:
- The system + variational directions
- The spline-order / du_sym for control reconstruction
- Pre-allocated buffers for forward/backward ODE state
- Trajectory component index maps for writing into the gradient vector

Implements:
- `objective_value(obj, traj)` — forward rollout, return `Q_r · ‖∂Ũ⃗(T)‖²/norm`
- `gradient!(∇, obj, traj)` — forward rollout, backward adjoint, scatter into `∇` at indices for `:u, :du, :Ũ⃗_1` (initial nominal state if applicable)
- `hessian_structure`, `hessian!` — return empty / no-op (rely on L-BFGS)

### `VariationalRolloutProblem` — problem-template constructor

Defined in `Piccolo.jl/src/control/templates/variational_rollout_problem.jl` (new file). Same external signature as `VariationalSplinePulseProblem`:

```julia
VariationalRolloutProblem(
    varsys, pulse, U_goal, N_or_times;
    Q, Q_r, R, R_u, R_du, R_ddu,
    du_bound, du_bounds, ddu_bound, ddu_bounds, Δt_bounds,
    variational_scales,
    constraints, piccolo_options,
    dynamics_spline_order, n_path_samples,
    free_phase, initial_phases, phase_name,
    global_bounds,
)
```

so users can swap `VariationalSplinePulseProblem` → `VariationalRolloutProblem` with a single name change.

Internally:
1. Build `nominal_sys`, `qtraj`, `base_traj` — identical to `VariationalSplinePulseProblem`
2. Add `:du` / `:ddu` derivatives if needed — identical
3. **Skip** the augmented `:var_Ũ⃗` state and its integrator
4. Add `TimeDependentBilinearIntegrator` for `:Ũ⃗` — identical to current nominal integrator
5. Build objective: `UnitaryInfidelityObjective + QuadraticRegularizer(:u) + QuadraticRegularizer(:du) + VariationalRolloutObjective(Q_r=...)`
6. Free-phase, embedded operator, leakage constraints: pass through unchanged

## Math

### Forward rollout

State: `x = [Ũ⃗; ∂Ũ⃗_1; …; ∂Ũ⃗_{n_vars}]`, total dim `iso_dim · (1 + n_vars)`.

ODE on `[0, T]`, with cubic-Hermite control reconstruction `u(τ; u_k, du_k, u_{k+1}, du_{k+1})` per knot interval `[t_k, t_{k+1}]`:

```
dŨ⃗/dt    =  G(u(t), t) · Ũ⃗
d∂Ũ⃗ᵢ/dt  =  G(u(t), t) · ∂Ũ⃗ᵢ  +  (G_varᵢ(u(t), t) / scaleᵢ) · Ũ⃗
```

Initial conditions: `Ũ⃗(0) = traj[:Ũ⃗][:, 1]` (typically iso(I)), `∂Ũ⃗ᵢ(0) = 0`.

Use Tsit5 from `OrdinaryDiffEqTsit5` (already a DirectTrajOpt dep). Solve from `t = 0` to `t = T` in one shot.

### Objective

```
J = Q_r · Σᵢ scale_i⁴ · ‖∂Ũ⃗ᵢ(T) / scaleᵢ‖² / (d · (Δt·N)² · n_vars)
  = (Q_r / (d · T² · n_vars)) · Σᵢ ‖∂Ũ⃗ᵢ(T) · scaleᵢ‖²
```

(matches current normalization at `variational_spline_problem.jl:380-382`)

### Backward adjoint

Set `λ(T)`:
- `λ_Ũ⃗(T) = 0`
- `λ_{∂Ũ⃗ᵢ}(T) = (2 Q_r · scaleᵢ² / (d · T² · n_vars)) · ∂Ũ⃗ᵢ(T)`

Solve backward `dλ/dt = −(∂f/∂x)ᵀ · λ` from `t = T → 0`:
- `∂f/∂Ũ⃗` block contributes to both `dλ_Ũ⃗/dt` and `dλ_{∂Ũ⃗ᵢ}/dt` (since `G_var · Ũ⃗` couples them)
- `∂f/∂(∂Ũ⃗ᵢ)` is just `G(u, t)` block-diag

In iso-vec form, with `Ũ⃗` and `∂Ũ⃗` viewed as `iso_dim`-vectors:

```
dλ_Ũ⃗/dt          =  -G(u, t)ᵀ · λ_Ũ⃗  -  Σᵢ G_varᵢ(u, t)ᵀ · λ_{∂Ũ⃗ᵢ}
dλ_{∂Ũ⃗ᵢ}/dt    =  -G(u, t)ᵀ · λ_{∂Ũ⃗ᵢ}
```

(Note `G` is real-antisymmetric in iso form, so `Gᵀ = -G` and signs simplify; double-check with finite difference.)

### Gradient assembly

For each knot pair `(t_k, t_{k+1})`, the control reconstruction `u(τ) = h_basis(τ) · [u_k; du_k; u_{k+1}; du_{k+1}]` depends linearly on the four NLP variables. So:

```
∂J/∂u_k    =  Σ_steps in adjacent intervals  ⟨λ(t),  (∂G/∂u_k(t)) · Ũ⃗(t) + (∂G_var/∂u_k(t)) · ∂Ũ⃗(t)⟩
```

(written schematically; in iso form involves the correct iso-G derivative). The 4 basis functions tell you the weight at each substep.

**Practical implementation**:
- Use the Tsit5 solution object's interpolant or a saved dense output to evaluate `(Ũ⃗(t), ∂Ũ⃗(t), λ(t))` at the same sub-grid
- Loop over knot intervals; per interval, accumulate `∂J/∂u_k`, `∂J/∂du_k`, `∂J/∂u_{k+1}`, `∂J/∂du_{k+1}` by quadrature
- Quadrature points: use the ODE solver's saved time grid (no extra cost) or a fixed Simpson grid

### Free-phase compatibility

The variational rollout is independent of `(cosθ, sinθ)`. So `∂J_var/∂globals = 0` and globals don't appear in the variational gradient. They only enter `UnitaryFreePhaseInfidelityObjective` — which is unchanged. **No work needed for free-phase compat.**

### Embedded operator

The augmented variational state's `subsystem_levels` already match the embedded `U_goal`. Rollout doesn't care — the variational ODE doesn't project. The same `varsys` is reused. **No work needed.**

## Files

| File | Action | Lines |
|---|---|---|
| `Piccolo.jl/src/control/objectives.jl` | Add `VariationalRolloutObjective` struct + interface impls | ~250 |
| `Piccolo.jl/src/control/templates/variational_rollout_problem.jl` | New file, mirrors `variational_spline_problem.jl` minus the augmented state | ~250 |
| `Piccolo.jl/src/control/templates/_problem_templates.jl` | Register new template (include statement) | ~5 |
| `Piccolo.jl/src/Piccolo.jl` | Export `VariationalRolloutProblem` | ~2 |
| `Piccolo.jl/test/test_variational_rollout.jl` | New test file: gradient finite-diff check, matches direct version on small case | ~200 |
| `DirectTrajOpt.jl/src/objectives/_objectives.jl` | (probably no change — should plug in as `AbstractObjective`) | 0 |

## Phased implementation

### Step 1 — Forward rollout only, finite-diff gradient (1 day)

Smallest possible verification:

- Build `VariationalRolloutObjective` with only `objective_value` implemented
- `gradient!` falls through to ForwardDiff or finite difference for now
- Build a `VariationalRolloutProblem` test with `N_knots = 4`, single qubit, single `H_var`
- Solve the same problem with `VariationalSplinePulseProblem` → assert same converged controls (to within Ipopt tolerance)
- This validates the rollout direction without committing to adjoint correctness

**Acceptance**: solver converges to ≥99.9% fidelity, same `Q_r·‖∂Ũ⃗(T)‖²` value as the direct version.

### Step 2 — Adjoint gradient implementation (3-4 days)

- Implement backward `λ` ODE
- Implement gradient quadrature for `:u`, `:du`
- Add finite-difference check in tests: for small problem with `n_drives=2`, `N_knots=4`, verify each `∂J/∂u_k`, `∂J/∂du_k` matches FD to `rtol=1e-5`
- This is the highest-risk step. Sign errors in `Gᵀ` vs `G`, missed cross-terms in `∂f/∂x`, basis-function indexing.

**Acceptance**: all 32 gradient components match FD to `rtol=1e-5` on the test problem.

### Step 3 — Multi-direction `n_vars > 1` support (1 day)

- Rollout already handles arbitrary `n_vars` (just longer state vector)
- Verify adjoint handles multiple `G_varᵢ` correctly (each contributes a coupling term to `dλ_Ũ⃗/dt`)
- FD gradient check with `n_vars = 3`

### Step 4 — Free-phase + embedded-operator integration test (1 day)

- Build a small `EmbeddedOperator` goal, qutrit, `free_phase = true`
- Verify constraints satisfied, finite-diff still matches

### Step 5 — Performance benchmark (1 day)

- Run your existing `robust_iswap_detuned_2MHz_150ns_5nsbuf.jl` with both:
  - `VariationalSplinePulseProblem` (current)
  - `VariationalRolloutProblem`
- Compare wall time, iteration count, final infidelity
- Run 3-level version. Same comparison.

**Acceptance**: same converged infidelity (within Ipopt tolerance), 3-level wall time ≥ 5× faster.

### Step 6 — Replace direct version (optional, 0.5 day)

Once `VariationalRolloutProblem` is stable and faster, the direct `VariationalSplinePulseProblem` can be marked deprecated or removed. Don't do this until step 5 has run for a while in real use.

**Total: 7-9 days of focused work**, plus debugging overhead. Realistic calendar time **1.5-2 weeks**.

## Risks

1. **Adjoint sign errors** — easy to make, hard to spot, only show as slow/no convergence in Ipopt. Mitigation: FD check in Step 2 is mandatory.
2. **Multiple variational directions** — coupling terms in `dλ_Ũ⃗/dt` are easy to drop. Mitigation: FD check with `n_vars=3`.
3. **Iteration count** — indirect methods often need more iterations than direct. Net speedup may be smaller than per-iter speedup suggests. Mitigation: benchmark in Step 5.
4. **ODE solver tolerances** — too loose → noisy gradients; too tight → slow. Defaults `reltol=1e-8, abstol=1e-10` should work but may need tuning per problem size.
5. **Memory for dense output** — saving `Ũ⃗(t), ∂Ũ⃗(t), λ(t)` at every ODE step over 150 ns may be tens of MB per evaluation. Acceptable, but careful with allocation in hot loops.

## Open questions

1. **Should `:Ũ⃗` also be rolled out (Phase 2 / full GRAPE)?**  
   Adds ~5× speedup on top of Phase 1 by removing the only remaining state-block from NLP. But touches fidelity-objective code paths. Defer until Phase 1 is stable.

2. **Use `SciMLSensitivity.jl` or hand-rolled adjoint?**  
   Hand-rolled is recommended. SciMLSensitivity is powerful but finicky around piecewise-defined controls (Tsit5 callbacks at knot boundaries). Hand-rolled is ~100 more lines but you understand every step.

3. **Should the rollout solver be shared with the existing `TimeDependentBilinearIntegrator`?**  
   No. Different problem (full-time-horizon vs single knot interval), different state dim, different solve_kwargs reasonable defaults. Keep them independent.

4. **What about `R_u`/`R_du`/`R_ddu` regularizers?**  
   Unchanged — these act on NLP variables directly, no rollout involved. Keep as-is.

5. **Cron / cluster runs?**  
   Once stable, the 3-level optimization that currently takes 1-3 hours could run in 5-15 min. Means you can do parameter sweeps interactively rather than overnight.

## Recommendation

**Do Phase 1.** The math is well-understood (it's the same adjoint method GRAPE has used for 30 years; you're just applying it to the variational adjoint as an objective). The code architecture is clean (single new `AbstractObjective` + one new problem-template wrapper). Biggest risk is gradient-correctness bugs, mitigated by the FD check in Step 2.

**Keep splines** — cubic Hermite is the right control parameterization for both hardware and ODE accuracy. Don't switch to piecewise-constant.

Defer Phase 2 (full GRAPE on `:Ũ⃗`) until Phase 1 is benchmarked and you understand whether the remaining `:Ũ⃗` NLP block is the bottleneck or not.
