# SplinePulseProblem + VariationalSplinePulseProblem edits

Summary of the uncommitted changes on Piccolo.jl branch `modulate` that make
`SplinePulseProblem` and the new `VariationalSplinePulseProblem` work with
cubic-Hermite dynamics and a path-constrained control.

Branch: `modulate` · status as of 2026-04-16 (uncommitted).

---

## 1. Integrator was changed

Yes. Both problems now thread a `spline_order` / `du_name` pair into
`BilinearIntegrator` / `TimeDependentBilinearIntegrator`, so the NLP dynamics
use the same cubic-Hermite interpolation as the pulse parameterization.

### `src/control/integrators.jl`

- `BilinearIntegrator(::UnitaryTrajectory, N; spline_order=1, du_name=nothing)`
  — forwards the two kwargs into the underlying `TimeDependentBilinearIntegrator`.
- New single-symbol `VariationalUnitaryIntegrator(sys, traj, var_state_sym, u; scales, spline_order, du_name)`.
  Uses the block-triangular `var_G` over one contiguous state block
  `[Ũ⃗; ∂Ũ⃗₁; …; ∂Ũ⃗ₙ]`. Dispatches to `TimeDependentBilinearIntegrator` when
  `sys.time_dependent`, otherwise `BilinearIntegrator`.
- Legacy multi-symbol `VariationalUnitaryIntegrator(sys, traj, Ũ⃗, Ũ⃗_variations, u; …)`
  now delegates to the single-symbol version (backwards-compatible shim).

## 2. `src/control/templates/spline_pulse_problem.jl`

Two new kwargs on `SplinePulseProblem`:

- `dynamics_spline_order::Union{Int, Nothing} = nothing` — auto-resolves to
  `1` for `LinearSplinePulse`, `3` for `CubicSplinePulse`; can be overridden.
- `n_path_samples::Int = 0` — when `> 0` and the pulse is cubic, adds a
  `CubicHermitePathConstraint` that bounds the interpolated control between
  knots at `n_path_samples` intermediate samples per interval. Bound is taken
  from `sys.drive_bounds`.

The `BilinearIntegrator(qtraj, N)` call became:

```julia
BilinearIntegrator(qtraj, N; spline_order = _dyn_order, du_name = _dyn_du_name)
```

where `_dyn_du_name = du_sym` whenever `_dyn_order >= 2`.

## 3. New `src/control/templates/variational_spline_problem.jl`

Builds a nominal `QuantumSystem` from the `VariationalQuantumSystem` so
`UnitaryTrajectory` has something to ODE-integrate for the warm start, then:

1. Adds `:du` (and optionally `:ddu`) components to the trajectory.
   `LinearSplinePulse` gets them via `add_control_derivatives`; `CubicSplinePulse`
   already exposes `:du` as Hermite tangents.
2. Packs everything variational into one contiguous state symbol:

   ```
   :var_Ũ⃗ = [ Ũ⃗ | ∂Ũ⃗₁ | … | ∂Ũ⃗ₙ ]    total dim = iso_dim * (1 + n_vars)
   ```

3. Runs two integrators:
   - Nominal — `I⊗G(u,t)` on `:Ũ⃗` (so the infidelity objective has a
     physical gradient w.r.t. `:u`; without it, `:Ũ⃗` is a free NLP variable).
   - Variational — the new single-symbol `VariationalUnitaryIntegrator` on
     `:var_Ũ⃗` with the block-triangular `var_G`.

   Both honor `_dyn_order` / `_dyn_du_name` (same auto-resolution as
   `SplinePulseProblem`).

4. `DerivativeIntegrator(:u → :du)` is added only for `LinearSplinePulse`.
   A second `DerivativeIntegrator(:du → :ddu)` is added whenever `ddu` is in play.

5. Objective:
   - `UnitaryInfidelityObjective` on nominal `:Ũ⃗`.
   - `QuadraticRegularizer` on `:u`, `:du`, optionally `:ddu`.
   - **Robustness**: one `KnotPointObjective` per error channel on the
     corresponding block of `:var_Ũ⃗`. Uses the same normalization as
     `UnitarySensitivityObjective`:
     `scale⁴ · ‖∂Ũ⃗/scale‖² / (d · (Δt₀·N)² · n_vars)`.

   **Robustness is evaluated at the terminal knot only** (`times = [N]`).
   If you want to penalize throughout the pulse, widen `times` — but note
   the `(Δt·N)²` factor in the normalization was chosen for terminal-sensitivity
   scaling, so the weighting semantics change.

6. Optional `CubicHermitePathConstraint` (same `n_path_samples` kwarg as
   `SplinePulseProblem`).

## 4. Supporting changes

- `src/quantum/systems/variational_quantum_systems.jl`
  - Added `time_dependent::Bool` field.
  - Constructor now accepts `time_dependent = true` and, in that branch,
    builds `G`, `G_vars` as `(a, t) -> …` and probes `levels` with `H(zeros(…), 0.0)`.

- `src/quantum/dynamics.jl`
  - `rollout*`, `unitary_rollout*`, `ket_rollout*` gained a new
    `interpolation = :cubic_hermite` option that uses `CubicHermiteSpline(traj, control_name)`
    from the optimized `:du` tangents — matches `spline_order = 3` in the NLP.
  - Existing `:cubic` switched to `DataInterpolations.CubicSpline` from knot
    values only (no tangents) — useful as an independent cross-check.

- `src/control/templates/_problem_templates.jl` — includes the new file.
- `Project.toml` — `SciMLBase` compat loosened from `"2.148"` to `"2"`.

---

## Quick reference — calling the new APIs

```julia
# Cubic-Hermite NLP dynamics + path-bounded control
qcp = SplinePulseProblem(
    qtraj, N;
    dynamics_spline_order = 3,      # default for CubicSplinePulse
    n_path_samples        = 8,      # bound u between knots at 8 samples/interval
    ...,
)

# Variational robustness with cubic dynamics
qcp = VariationalSplinePulseProblem(
    varsys, pulse, U_goal, times;
    Q_r                   = 100.0,
    variational_scales    = [scale_1, scale_2, ...],
    dynamics_spline_order = 3,
    n_path_samples        = 8,
    ...,
)

# Rollout cross-check consistent with spline_order=3
fid = unitary_rollout_fidelity(traj, sys; interpolation = :cubic_hermite)
```
