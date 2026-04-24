# Cubic Hermite Splines in Piccolo

## What is a cubic Hermite spline?

A cubic Hermite spline is a piecewise cubic polynomial that interpolates both the **values** and **derivatives (tangents)** at each knot point. Between knots $k$ and $k+1$, the curve is determined by four quantities:

$$u_k, \quad \dot{u}_k, \quad u_{k+1}, \quad \dot{u}_{k+1}$$

where $u_k$ is the control amplitude and $\dot{u}_k$ is the tangent (slope) at knot $k$.

## Hermite basis polynomials

On the normalized interval $\tau = (t - t_k) / \Delta t_k \in [0, 1]$, the interpolant is:

$$u(\tau) = h_{00}(\tau)\, u_k + h_{10}(\tau)\, \Delta t_k\, \dot{u}_k + h_{01}(\tau)\, u_{k+1} + h_{11}(\tau)\, \Delta t_k\, \dot{u}_{k+1}$$

The four Hermite basis polynomials are:

| Basis | Formula | Role |
|-------|---------|------|
| $h_{00}(\tau)$ | $2\tau^3 - 3\tau^2 + 1$ | Blends $u_k$ (1 at $\tau=0$, 0 at $\tau=1$) |
| $h_{10}(\tau)$ | $\tau^3 - 2\tau^2 + \tau$ | Blends $\dot{u}_k$ (slope 1 at $\tau=0$, 0 at $\tau=1$) |
| $h_{01}(\tau)$ | $-2\tau^3 + 3\tau^2$ | Blends $u_{k+1}$ (0 at $\tau=0$, 1 at $\tau=1$) |
| $h_{11}(\tau)$ | $\tau^3 - \tau^2$ | Blends $\dot{u}_{k+1}$ (0 at $\tau=0$, slope 1 at $\tau=1$) |

The $\Delta t_k$ factors ensure the tangents are in physical units (amplitude per time), not per unit $\tau$.

## Key properties

- **Local support**: Each segment $[t_k, t_{k+1}]$ depends only on its two endpoint values and tangents. Changing $u_k$ or $\dot{u}_k$ affects only the adjacent segments.
- **$C^1$ continuity**: The curve is continuous in both value and first derivative at every knot.
- **Independent tangents**: Unlike natural cubic splines (which compute tangents globally from knot values only), the tangents $\dot{u}_k$ are **free variables** that the optimizer controls independently.

## How it is used in this NLP

`CubicSplinePulse` parameterizes the control as a cubic Hermite spline with NLP decision variables:

```
traj[:u]   — knot values     u_k        (bounded: |u_k| ≤ a_bound)
traj[:du]  — knot tangents   dot_u_k    (bounded: |du_k| ≤ du_bound)
traj[:t]   — knot times      t_k
traj[:Δt]  — timestep sizes  Δt_k
```

During integration, the `TimeDependentBilinearIntegrator` evaluates $u(\tau)$ at internal ODE sub-steps using the Hermite formula above, so the dynamics see the smooth cubic curve — not just the knot values.

## What spline integrator was implemented?

**Short answer: no brand-new integrator type was written.** The existing `TimeDependentBilinearIntegrator` in `DirectTrajOpt.jl` was **extended**, and the Piccolo.jl wrappers were taught to forward the new kwargs into it. The change is best described as "the same integrator, but now it knows how to interpolate with Hermite cubics instead of only linears."

### Layer 1 — `DirectTrajOpt.jl/src/integrators/time_dependent_bilinear_integrator.jl`

Before, `TimeDependentBilinearIntegrator` supported only:

- `spline_order = 0` — zero-order hold, `u(τ) = uₖ`
- `spline_order = 1` — linear, `u(τ) = (1−τ)·uₖ + τ·uₖ₊₁`

It was extended with two more orders, gated on a new `du_name::Union{Symbol,Nothing}` kwarg that points to the tangent variable in the trajectory:

- `spline_order = 2` — quadratic Hermite: matches $u_k$, $\dot u_k$, $u_{k+1}$.
  $u(\tau) = a\tau^2 + \Delta t\, \dot u_k\, \tau + u_k,\ a = u_{k+1} - u_k - \Delta t\, \dot u_k$.
- `spline_order = 3` — **cubic Hermite**: matches $u_k$, $\dot u_k$, $u_{k+1}$, $\dot u_{k+1}$.
  Same basis as above:
  $u(\tau) = h_{00}(\tau)\, u_k + h_{10}(\tau)\,\Delta t\, \dot u_k + h_{01}(\tau)\, u_{k+1} + h_{11}(\tau)\,\Delta t\, \dot u_{k+1}$.

The integrator's internal ODE (Tsit5 on $\tau \in [0,1]$) calls `u_fn(τ, pₖ, Δtₖ)` at each sub-step, so orders 2/3 make the ODE see the full Hermite curve rather than a straight line between knots. Crucially, each step still depends only on $(u_k, \dot u_k, u_{k+1}, \dot u_{k+1})$ — the **Jacobian stays banded**.

### Layer 2 — `Piccolo.jl/src/control/integrators.jl`

`BilinearIntegrator(::UnitaryTrajectory, N)` and `VariationalUnitaryIntegrator` gained `spline_order` / `du_name` kwargs and forward them into `TimeDependentBilinearIntegrator`. Before, they hardcoded `spline_order = 1`.

### Layer 3 — `Piccolo.jl/src/control/templates/spline_pulse_problem.jl` (+ new `variational_spline_problem.jl`)

Auto-resolves the order at problem construction:

- `LinearSplinePulse`  → `spline_order = 1`, `du_name = nothing`
- `CubicSplinePulse`   → `spline_order = 3`, `du_name = :du`

User can override via the `dynamics_spline_order` kwarg.

## Distinction from the default setup

Before this work, the default setup with a `CubicSplinePulse` was **inconsistent**:

| Component               | Before                                            | After (with `dynamics_spline_order = 3`) |
|-------------------------|---------------------------------------------------|------------------------------------------|
| Pulse parameterization  | Cubic Hermite in $u_k, \dot u_k$                  | Cubic Hermite in $u_k, \dot u_k$         |
| Rollout interpolation   | Linear (or `:cubic_hermite` if asked)             | `:cubic_hermite` (matches NLP)           |
| **NLP integrator**      | **Linear** between knots (`spline_order = 1`)     | **Cubic Hermite** (`spline_order = 3`)   |
| NLP dynamics ≟ rollout  | **No** — different interpolations                 | **Yes** — identical polynomial           |

The subtle pre-edit failure mode: the optimizer had `:du` as decision variables (via `CubicSplinePulse`), could regularize and bound them, and the rollout could use them — but the NLP's own dynamics constraint treated the control as **piecewise linear**. So $\tilde U_k$ in the NLP was the evolution under a line through the knots, while the rollout's $\tilde U$ was the evolution under the cubic. Converged NLP fidelity and rollout fidelity could disagree, and `:du` had no direct dynamical effect — it only entered the objective through regularizers and bounds.

After the edits, both paths evaluate the same polynomial on every sub-step, so:

- NLP fidelity ≈ rollout fidelity (modulo ODE tolerance).
- $\dot u_k$ now has **first-class dynamical meaning** inside the NLP — changing a tangent directly alters $\tilde U_{k+1}$, so the optimizer can shape the curve between knots instead of just the values.
- Coarser grids become viable: fewer knots with well-chosen tangents can match what previously required a finer linear grid.

## Why not natural cubic splines?

A natural cubic spline computes tangents by solving a global tridiagonal system from the knot values alone. This means:

1. **Dense Jacobian**: changing any single knot value changes tangents everywhere → the constraint Jacobian is dense → IPOPT becomes slow.
2. **No tangent DOFs**: the optimizer cannot independently tune the curve shape between knots.

The Hermite parameterization keeps the Jacobian **banded and sparse** because each constraint only touches $(u_k, \dot{u}_k, u_{k+1}, \dot{u}_{k+1})$.

## Rollout consistency

The rollout uses `interpolation = :cubic_hermite`, which constructs a `DataInterpolations.CubicHermiteSpline` from the optimized `traj[:u]` and `traj[:du]`. This is mathematically identical to the hand-coded Hermite basis in the NLP integrator, so rollout fidelity ≈ NLP fidelity.

Using `interpolation = :cubic` instead constructs a **natural cubic spline** from `traj[:u]` only (ignores `traj[:du]`), producing a different curve and a lower/different fidelity.

## Overshoot and path constraints

Cubic splines can overshoot between knots even when all knot values are within bounds. The `n_path_samples` option adds inequality constraints at $n$ uniformly-spaced interior points per interval:

$$-a_\text{bound} \leq u(\tau_j) \leq a_\text{bound}, \quad \tau_j = \frac{j}{n+1}, \quad j = 1, \ldots, n$$

Each constraint is:

$$u(\tau_j) = h_{00}(\tau_j)\, u_k + h_{10}(\tau_j)\, \Delta t_k\, \dot{u}_k + h_{01}(\tau_j)\, u_{k+1} + h_{11}(\tau_j)\, \Delta t_k\, \dot{u}_{k+1}$$

When timesteps are fixed (`timesteps_all_equal = true`), $\Delta t_k$ is a constant and these are **linear constraints** in $(u, \dot{u})$. With `n_path_samples = 10`, each interval is checked at 10 interior points, reliably catching overshoot regardless of where the spline extremum falls.

## Quick reference

| Parameter | Effect |
|-----------|--------|
| `du_bound` | Limits tangent magnitude; indirectly reduces overshoot |
| `n_path_samples` | Directly enforces amplitude bounds at interior sample points |
| `dynamics_spline_order = 3` | Makes NLP integrator use cubic Hermite (default for `CubicSplinePulse`) |
| `interpolation = :cubic_hermite` | Makes rollout use the same Hermite curve as the NLP |
| `interpolation = :cubic` | Uses natural cubic spline (cross-check; different from NLP) |
