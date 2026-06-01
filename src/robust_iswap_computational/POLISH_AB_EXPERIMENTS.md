# 3-lvl polish A/B experiments

Methodology validation for the spectral-notch / 2-lvl-warm-start / 3-lvl-polish
pipeline applied to the 2Q resonant-exchange iSWAP.

For program-level context, design decisions, and the warm-spline result
already in hand, see [`ROBUST_ISWAP_PROGRAM.md`](ROBUST_ISWAP_PROGRAM.md).
This document covers the *next* two A/B experiments scoped to answer specific
methodological questions, and is intentionally narrower.

## What we already established

One reference point exists: a 300-iter `VariationalSplinePulseProblem` (collocation)
solve in 9-dim Hilbert space (3 lvls per qubit), warm-started from a 4-dim
2-lvl + spectral-notch optimization. Result: three-nines fidelity, visible
σ_z-channel robustness vs the calibrated Gaussian-square baseline, leakage
suppressed to ~10⁻⁴–10⁻³ range. Per-iter wall time ~150 s.

Two methodological questions remain unresolved by that single data point:

1. **Is the warm-start load-bearing?** Maybe the polish lands in the same
   neighborhood from random initialization, just slower. If so, the 2-lvl
   spectral-notch optimization is unnecessary work — you'd skip directly
   to 9-dim with random init.
2. **Is collocation the right solver?** The historical DRAG warm-start work
   in `../modulate_iswap/` used `VariationalRolloutProblem` (rollout-based)
   at ~20 s/iter — 7× faster than collocation per iter. The rollout DRAG
   run hit Max_Iterations at 200 without converging robustness; possibly
   that was a per-iter weakness of rollout for tight constraints, possibly
   it was Q_r being too small. Need to disentangle.

## Experimental design

Two-axis factorial, three cells (warm × {collocation, rollout} and cold × collocation):

|                | Warm-start (4-dim spectral-notch traj) | Cold (random, seed=42) |
|---|---|---|
| **Collocation** (VariationalSplinePulseProblem) | **W-Coll** (existing reference) | **C-Coll** (this experiment) |
| **Rollout** (VariationalRolloutProblem) | **W-Roll** (this experiment) | — |

The fourth cell (C-Roll) is not required to answer the two questions:

- **W-Coll vs C-Coll** isolates the contribution of the warm-start at fixed
  solver, fixed init scheme, fixed everything else.
- **W-Coll vs W-Roll** isolates the contribution of the solver template at
  fixed warm-start, fixed everything else.

All other dimensions — Hilbert space, Hamiltonian, error generators, control
spline, constraints, regularization, iteration budget — are held identical
across the three runs. This is the cleanest possible factorial isolation
of the two factors of interest.

## What's held constant (the controlled axes)

These match across all three scripts. Verified by code inspection:

- **Hilbert space**: 9-dim, 3 lvls per qubit.
- **Drift**: `H = H_anh_9 + g_flat · (X₁X₂ + Y₁Y₂)`. `g_flat = 2π·2 MHz`,
  η = −2π·170 MHz on each qubit.
- **Controls**: 4 MW channels `(u_X1, u_Y1, u_X2, u_Y2)` driving
  `XI_9, YI_9, IX_9, IY_9`, on-resonance with each qubit.
- **Variational generators**: `H_vars = [n̂_1, n̂_2, n̂_1·n̂_2]` —
  physical transmon dephasing on each qubit plus dispersive cross-Kerr.
- **Spline**: 15 cubic-Hermite knots over the 180 ns flat region, Δt = 180/14 ≈
  12.857 ns.
- **Targets**: same `U_goal = V_fall† · U_iSWAP · V_rise†` (4×4 in qubit
  subspace), lifted via `EmbeddedOperator(U_goal, [1,2,4,5], 9)`.
- **Constraints**:
  - `FinalUnitaryFidelityConstraint(embedded_target, :Ũ⃗, 0.9999)`
  - `SpectralLeakageConstraintIQ(:u, 1, 2, 2π·170 MHz, 1e-2)` (qubit-1 IQ pair)
  - `SpectralLeakageConstraintIQ(:u, 3, 4, 2π·170 MHz, 1e-2)` (qubit-2 IQ pair)
  - `du_bound = Inf`, amplitude path constraint via `n_path_samples = 3` on the
    combined 4-channel MW envelope.
- **Objective weights**: `Q = 0` (fidelity is a constraint, not penalty),
  `Q_r = 1e2`, `R = 1e-3`.
- **IPOPT options**: `max_iter = 300`, `tol = constr_viol_tol = acceptable_tol = 1e-8`,
  `eval_hessian = false` (L-BFGS quasi-Newton).
- **Integrator tolerances**: Tsit5 defaults of `reltol = 1e-5, abstol = 1e-7`
  per Piccolo's variational rollout settings.
- **Random seed**: `42` (only relevant for C-Coll's random init).

The g_eff envelope used in the post-solve full-gate verification — filtered
200 ns square, η-killer Gaussian (σ_f = 100 MHz), AWG Gaussian
(σ_f = B_AWG/√(ln 2)) — is also identical across scripts.

## What varies (the design axes)

| Factor | W-Coll | W-Roll | C-Coll |
|---|---|---|---|
| Solver template | `VariationalSplinePulseProblem` | `VariationalRolloutProblem` | `VariationalSplinePulseProblem` |
| Initialization | Loaded `traj_robust` (2-lvl spectral-notch traj, 15 knots) | Same | Random pulse, 300 samples, seed = 42 |
| `:var_Ũ⃗` state | Collocation variable | Not in state (rolled out in objective) | Collocation variable |
| Expected per-iter wall time | ~150 s | ~20 s | ~150 s |
| Expected 300-iter wall time | ~12 h | ~1.5–3 h | ~12 h |

## What we measure

For each run, three classes of output:

1. **Optimizer trace** (IPOPT log):
   - Constraint violation `inf_pr` vs iteration
   - Dual infeasibility `inf_du` vs iteration
   - Objective value vs iteration
   - Iteration at which `inf_pr` first drops below 1e-3, 1e-5, 1e-7
   - Terminal status (`Solve_Succeeded` vs `Maximum_Iterations_Exceeded`)
2. **Gate quality at ε = 0** (3-lvl full-gate honest verification):
   - `F_3 = |Tr(U_iSWAP† · U_sub)|² / 16`
   - `L_3 = 1 − Tr(U_sub†·U_sub) / 4` (population that left the qubit subspace)
   - These are the "did we get a usable gate" numbers.
3. **Robustness** (3-lvl full-gate ε-sweep):
   - F_3(ε) and L_3(ε) for each of n̂_1, n̂_2, n̂_1·n̂_2 channels,
     ε ∈ [−5, +5] MHz at 101 points
   - All raw (no virtual-Z post-correction). Compared to the calibrated π/4
     Gaussian-square default in the same plot.
   - "Flat" F(ε) over a reasonable window is the success criterion for
     robustness.

## Hypotheses and what each outcome would mean

### Q1: Does warm-start help? (W-Coll vs C-Coll, same wall-time budget)

| Outcome | Interpretation |
|---|---|
| W-Coll wins on F, L, and robustness; C-Coll fails feasibility | Warm-start is *load-bearing*. The 2-lvl spectral-notch optimization meaningfully constrains the basin of attraction for the 3-lvl polish. Sticking with the full 2-lvl → 3-lvl pipeline is justified. |
| W-Coll and C-Coll converge to similar gates | Warm-start saves iterations but doesn't change the basin. The 2-lvl pass is then optional — could be skipped for simplicity, or kept as a cheap pre-conditioner. |
| C-Coll wins | Surprising; would mean random init reaches a *better* local minimum than the warm-start. Suggests the warm-start funnels into a sub-optimal manifold. |
| Both fail at 300 iter | Neither setup is enough; the problem is harder than the budget. Need higher Q_r, longer iter, or restructured problem. |

Expected: W-Coll dominates on feasibility-by-iter-N for small N; outcomes
at iter 300 likely close, with W-Coll having an edge.

### Q2: Does collocation help? (W-Coll vs W-Roll, same warm-start)

| Outcome | Interpretation |
|---|---|
| W-Coll achieves better feasibility (`inf_pr` lower) at iter 300 | Collocation's explicit Lagrange-multiplier treatment of hard constraints pays off; rollout's adjoint-through-rollout gradients can't drive `F ≥ 0.9999` to feasibility as tightly. |
| W-Roll achieves equal feasibility in less wall time | Rollout's 7× per-iter speedup wins for this problem class; methodology should default to rollout. |
| W-Coll achieves better robustness, W-Roll better feasibility | The two solvers occupy different points on the optimality–feasibility trade. Use rollout for exploration sweeps (e.g. Q_r choices), collocation for final polish. |
| W-Roll converges to dramatically different gate | Different basins. Each solver's local minimum structure depends on the gradient calculation — interesting but harder to interpret. |

Expected: W-Coll achieves tighter constraint satisfaction (lower `inf_pr`).
W-Roll may complete more iterations within a fixed wall-time budget but with
less per-iter progress on tight constraints. Historical evidence from
`../modulate_iswap/`'s rollout DRAG run supports this hypothesis: that run
hit `Max_Iterations_Exceeded` with `inf_pr ~ 3e-5` (good but not at tol) and
robustness barely improved over the 2-lvl baseline.

## Decision criteria

Define "success" upfront so the post-hoc analysis isn't biased:

- **A run succeeds methodologically** if it terminates with `inf_pr < 1e-4`
  (constraints near-feasible), and the 3-lvl full-gate F_3 at ε = 0
  exceeds 0.999 (three-nines minimum) under raw verification.
- **The warm-start is justified** (Q1) if W-Coll succeeds AND C-Coll either
  fails or matches W-Coll only at iter 300 (i.e., warm-start provides
  speed-up).
- **Collocation is preferred** (Q2) if W-Coll succeeds AND W-Roll fails OR if
  W-Roll succeeds but at substantially worse robustness (`χ_n̂` at termination
  > 2× the W-Coll value). Otherwise rollout is preferred (faster).

## Files and run instructions

Three scripts; outputs are tagged so they don't collide:

| Script | Tag | Outputs |
|---|---|---|
| `robust_iswap_polish_3lvl.jl` | `polish_3lvl_…_profile` | `traj_*`, `ipopt_*.log`, plus plots |
| `robust_iswap_polish_3lvl_rollout.jl` | `polish_3lvl_rollout_…` | same shape |
| `robust_iswap_polish_3lvl_cold.jl` | `polish_3lvl_cold_…` | same shape |

Each script self-contained: activates the local project, `Pkg.develop`s the
sibling-layout Piccolo and DirectTrajOpt, runs the 300-iter solve, saves
trajectory + IPOPT log + ε-sweep numerical data (`eps_sweep_*.jld2`) + four
PNGs (full-gate ε-sweep, MW time domain, MW frequency domain, fidelity panels).

Run (from `src/robust_iswap_computational/`):

```bash
julia --project=. robust_iswap_polish_3lvl_rollout.jl   # ~1.5–3 hr
julia --project=. robust_iswap_polish_3lvl_cold.jl       # ~12 hr
```

The existing W-Coll run does not need to be re-run; its outputs are already
saved with the `_profile` suffix.

## Analysis plan once all three runs complete

1. **IPOPT log comparison**: parse the three `ipopt_*.log` files, plot
   `(iter, inf_pr)`, `(iter, inf_du)`, `(iter, objective)` overlaid for the
   three runs. Tells us *trajectories* through the optimization, not just
   endpoints.
2. **Endpoint comparison table**: at termination, report F_3 (ε=0), L_3 (ε=0),
   `χ_n̂_i` per generator, dual infeasibility, total wall time, and IPOPT exit
   status for each run.
3. **Robustness curves**: overlay F_3 vs ε and 1−F_3 vs ε for the three runs
   on the same axes (one per channel, three panels per figure). Visible
   flatness is the robustness signature.
4. **Postprocess notebook** (`postprocess_polish.ipynb`) loads each
   `eps_sweep_*.jld2` and produces the side-by-side plots.

## What this doesn't address (out of scope)

These experiments isolate the two factors above and nothing else.
Methodological questions left open and that should be subsequent A/Bs once
the warm-start / solver question is settled:

- **Q_r magnitude**: is 1e2 the right weight on variational sensitivity? The
  historical DRAG run also used 1e2 and barely buys robustness. A separate
  Q_r sweep (1e2 / 1e3 / 1e4) at fixed warm-start and solver is the natural
  next experiment.
- **`ε_max` tightness**: 1e-2 is loose; tighter spectral leakage might enable
  more 3-lvl convergence but at the cost of more constraint pressure. A
  separate ε_max sweep at fixed Q_r is also a natural follow-up.
- **`n_path_samples`**: held at 3 for all three runs (same as W-Coll baseline).
  Whether 5 or 7 sub-knot amplitude path checks would materially change the
  result is unstudied here.
- **Polish system dimension**: 9-dim (3 lvls each) is the minimum that
  captures the leakage channels. 16-dim (4 lvls each) would catch higher-
  order |3⟩ effects but at much higher per-iter cost. Out of scope.

The first two follow-ups (Q_r and ε_max) are the natural sweeps once we know
*which solver / init combination* to use them with. That's what these
A/B experiments determine.
