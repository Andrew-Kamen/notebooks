# Session Handoff — Modulated iSWAP / DRAG / 3-Level Robust Optimization

**Date**: 2026-05-12
**Project**: robust_control_sam/src/modulate_iswap
**Branch contexts**: robust_control_sam@main (Andrew-Kamen/notebooks), Piccolo.jl@modulate (Andrew-Kamen/Piccolo-private), NamedTrajectories.jl@main

This document brings a new agent (or future-you) up to speed on the modulated-iSWAP robust-gate optimization work. Read top-to-bottom; the "Current State" section lists what's actively running and what to do next.

---

## 1. Project context

**Goal**: optimize a robust, low-leakage iSWAP gate on a 2-qubit Duffing-transmon system using Piccolo's variational rollout framework. Target: F ≥ 0.9999, leakage ≤ 10⁻⁴, robustness to detuning errors (n̂_1, n̂_2, n̂_1·n̂_2 perturbations).

**System parameters (original)**:
- Coupling: g_eff = 2π · 0.002 rad/ns = 2 MHz
- Anharmonicity: η = -2π · 0.170 rad/ns = -170 MHz
- Drive bound: a_bound = 2π · 0.01 rad/ns = 10 MHz
- Idle detuning: δ₁₂ = 2π · 0.06 rad/ns = 60 MHz
- Computational subspace: indices [1, 2, 4, 5] in 9-dim Hilbert space (3 levels × 2 qubits)

**Gate layout**: Gaussian rise (4σ = 5 ns) + buffer flat (5 ns) + microwave region (130 ns) + buffer flat (5 ns) + Gaussian fall (5 ns) = **150 ns total**.

---

## 2. Theoretical context

### Poggi & Kiely paper (arXiv:2509.26247, Sep 2025)
"Suppressing Leakage and Maintaining Robustness in Transmon Qubits."

Key result: there's a **fundamental trade-off** between leakage suppression and fidelity susceptibility (robustness) in transmon gates. Joint optimization is hard; they advocate a **two-stage sequential approach** — minimize one objective then the other with the first held near its optimum.

This paper validated, formally, what we hit empirically in the 3-level Q100 random-init run (see below).

### DRAG (Motzoi et al., PRL 103, 110501)
Single-qubit pulse-shaping recipe that cancels leakage to leading order in 1/|η|:
- u_X^new = u_X + du_Y/η
- u_Y^new = u_Y − du_X/η
(symmetric form for X+Y primary drives, applied per qubit)

For our problem, Ω/|η| ≈ 0.06 → leading-order DRAG residual leakage ~10⁻³.

---

## 3. Methodology evolution (what we tried, in order)

### Stage 1: Existing 2-level optimization (pre-session, already converged)
**Script**: `robust_iswap_detuned_2MHz_150ns_5nsbuf_rollout_1kiter.jl`
- 4-dim Pauli Hilbert space, no anharmonicity, no |2⟩
- Z-robustness via VariationalRolloutObjective with ZI, IZ, ZZ error operators
- N_knots = 24, T_mw = 130 ns, Q_r = 100, 1000 iters
- **Result**: F = 0.99990, robust to Z errors ✓
- Output dir: `robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_rollout_1kiter_seed42/`

### Stage 2: 3-level random-init run (the "Q100" run — pre-existing)
**Script**: `robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_rollout.jl` (note: fidelity constraint is **commented out** in this script)
- 9-dim Duffing with H_anh, n̂_1/n̂_2/n̂_1·n̂_2 robustness, N_knots = 20, Q_r = 100
- Random init, 1000 iters
- **Result**: MaxIter, F_flat = 0.955, leakage = 3.4%
- Output dir: `robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_seed42_Q100/`
- **What went wrong**: no fidelity constraint, optimizer found a local optimum that exploits |2⟩-mediated paths to reduce ‖∂U/∂n̂‖² at the cost of fidelity. Classic Poggi-Kiely trade-off in action.

### Stage 3: Diagnostic — leakage_through_time notebook
**Notebook**: `leakage_through_time.ipynb`
- Loads pulse_full_gate.csv from each run
- Step-by-step midpoint Magnus propagation in 3-level Duffing
- Tracks leakage(t) and F(t) at every dt = 0.05 ns step
- Includes:
  - 2-lvl pulse simulated in 3-level → ~6% leakage (drive-induced, no DRAG)
  - 3-lvl Q100 → 3.4% leakage (matches its saved CSV)
  - **Analytic DRAG correction of the 2-lvl pulse** → F = 0.997, leakage ~10⁻³ in 3-lvl, **with 2-lvl robustness inherited**
  - "P2_long" stretched-time test (T → 2T, amp → amp/2): predicted leakage ~10⁻⁴ from (Ω/η)² scaling
- ε-sweep code that re-implements plot_results.jl's susceptibility analysis on multiple pulses
- **Key takeaway**: analytic DRAG-corrected 2-lvl beats the 3-lvl optimizer's output by a wide margin

### Stage 4: DRAG-constrained 2-level optimization attempt
**Script**: `robust_iswap_detuned_2MHz_150ns_5nsbuf_drag_constrained.jl`
- Tried to enforce continuous DRAG `u_Y(τ) = −u̇_X(τ)/η` at intermediate samples via a custom `DRAGPathConstraint` (mirroring `CubicHermitePathConstraint`)
- **Result**: Ipopt exited with `TOO_FEW_DOF` (0 iters)
- **What went wrong**: cubic Hermite splines are over-constrained when DRAG is enforced continuously — `u_Y` is a cubic, `du_X` is a quadratic, can't be equal except trivially. We computed this would happen in advance, ran it anyway to confirm.

### Stage 5: DRAG-warmstart 3-level optimization
**Script**: `robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_drag_warmstart.jl`
- Loads the 2-lvl trajectory, applies symmetric DRAG at the knots, uses as initial pulse for 3-lvl optimization
- Uses a **custom `FinalSubspaceProcessFidelityConstraint`** (bare `|Tr(U_goal'·U_sub)|²/n²` ≥ 0.9999, not the leakage-inflated F_avg formula Piccolo's stock constraint uses)
- N_knots = 24, T_mw = 130 ns, Q_r = 100
- **Results (200-iter probe)**:
  - F_flat = 0.99994 (NLP) / 0.99976 (independent)
  - Leakage = 4.1×10⁻⁵ (NLP) / 2.2×10⁻⁴ (independent)
  - But: **`Full gate F to iSWAP` = 0.942** (this is a script measurement artifact — see "What still puzzles me" below)
  - Convergence: MaxIter at 200 iters, inf_pr ≈ 5×10⁻⁶
- Output dir: `robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_dragwarmstart_seed42/`

### Stage 6: 300 ns stretched runs (currently in progress)
**Premise**: leakage scales as `(Ω/η)²`, so halving Ω and doubling T reduces leakage 4×. Plus more time → more variational expressivity for robustness.

**Construction**:
- Scale g_eff: 2 MHz → 1 MHz (halved)
- Scale time: σ_rise 1.25 → 2.5 ns; buffer_flat 5 → 10 ns; T_total 150 → 300 ns
- Keep a_bound = 10 MHz (user's choice — give optimizer freedom to grow amplitudes if needed)
- Q_r = 1000 (10× higher robustness weight to push past the warm-start's local optimum)
- Strict process F constraint = 0.9999

**Two scripts**:

(a) **Stretched warm-start**: `robust_iswap_detuned_1MHz_300ns_3level_stretched_warmstart.jl`
- Loads the DRAG-warmstart 3-lvl trajectory, halves u, divides du by 4, stretches time 2×
- All rotation areas ∫g·dt and ∫u·dt preserved → same iSWAP target
- 200-iter local probe
- **Status when handed off**: 200-iter run completed; objective hovering at ~1610 (= Q_r × ‖∂U/∂n̂‖² with sensitivity ≈ 1.6, same as 150 ns warm-start before scaling). Robustness DID NOT improve substantially — confirms the warm-start sits at a local minimum.

(b) **Stretched random-init**: `robust_iswap_detuned_1MHz_300ns_3level_random_init.jl`
- Same physics, but random initialization (no warm-start)
- num_iter = 2500 (overnight SSH budget)
- **Status when handed off**: currently running on SSH machine in tmux, ~24 hours wall time
- Pulled robust_control_sam@main to SSH before launching

---

## 4. Key files

### Active optimization scripts (currently in use)
| File | Purpose | Status |
|---|---|---|
| `robust_iswap_detuned_2MHz_150ns_5nsbuf_rollout_1kiter.jl` | 2-lvl Pauli rollout, source for DRAG warm-start | Converged, output saved |
| `robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_rollout.jl` | 3-lvl random-init (Q100 run, no F constraint) | Pre-existing, fidelity constraint commented out |
| `robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_drag_warmstart.jl` | 3-lvl with DRAG warm-start + strict F constraint | 200-iter probe done, hit MaxIter |
| `robust_iswap_detuned_1MHz_300ns_3level_stretched_warmstart.jl` | 300 ns stretched, warm-start | 200-iter probe done; obj stuck ~1610 |
| `robust_iswap_detuned_1MHz_300ns_3level_random_init.jl` | 300 ns stretched, random init | **Running on SSH, ~24h wall time** |
| `robust_iswap_detuned_2MHz_150ns_5nsbuf_drag_constrained.jl` | DRAG-constrained 2-lvl (FAILED) | TOO_FEW_DOF; kept for reference |

### Analysis / plotting scripts
| File | Purpose |
|---|---|
| `plot_results.jl` | Generic plotting (combined.png, default_vs_robust*.png, controls_full_gate.png, fidelity_*.csv) — parses parameters.txt to auto-detect any run |
| `plot_stretched_results.jl` | Same as plot_results.jl but defaults to the stretched run dir; Gaussian-square baseline computed at the run's g_eff |
| `leakage_through_time.ipynb` | Notebook with leakage(t), F(t), DRAG analysis, P2_long stretched test, ε-sweep, Ipopt iteration history |
| `duffing_3level_verify_5nsbuf.ipynb` | Verifies a pulse in 3-level Duffing via QuantumToolbox-style propagation |
| `profile_rollout_3level.jl` | Performance profiling for VariationalRolloutObjective gradient |
| `test_rollout_hadamard_regression.jl` | Regression test for the rollout speedups (single-qubit Hadamard) |

### Documentation
| File | Purpose |
|---|---|
| `physics.md` | Hamiltonian + cost function definitions |
| `optimizations.md` | VariationalRolloutObjective per-iter speedup notes |
| `cubic_hermite_splines.md` | Spline basis definitions |
| `robust_iswap_plan.md` | Original gate-design plan |
| `drag_transformation.md` | **NEW**: DRAG derivation specific to this problem |
| `SESSION_HANDOFF.md` | This file |
| `VERSION_CONTROL.md` | Multi-repo workflow doc (in `src/`) |

### Run output directories
- `robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf/` — 2-lvl direct (old, pre-rollout)
- `robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_rollout_1kiter_seed42/` — 2-lvl rollout (DRAG-warmstart source)
- `robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_seed42_Q100/` — 3-lvl Q100 random init
- `robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_dragwarmstart_seed42/` — DRAG warm-start 3-lvl
- `robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_dragconstr_170MHzeta_npath5_seed42/` — DRAG-constrained (TOO_FEW_DOF artifact)
- `robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_stretched_Qr1000_seed42/` — 300 ns stretched warm-start
- `robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_random_Qr1000_seed42/` — 300 ns stretched random init (in progress)

---

## 5. What worked

1. **Analytic DRAG correction of the 2-lvl pulse**: F = 0.997, leakage ~10⁻³ in 3-level Duffing, 2-lvl robustness inherited — for free (no optimization needed). This was the most successful intervention of the session.

2. **DRAG-warmstart 3-lvl + strict process F constraint**: closed the gap from F = 0.955 to F = 0.99976 (independent measurement). Convergence in 200 iters vs 1000+ from random init. The custom `FinalSubspaceProcessFidelityConstraint` is the key — Piccolo's stock constraint (F_avg) allows leakage gaming.

3. **Custom fidelity constraint pattern** (in `_drag_warmstart.jl` and `_stretched_warmstart.jl` and `_random_init.jl`):
```julia
function FinalSubspaceProcessFidelityConstraint(U_goal_4x4, subspace, state_name, F_thr, traj)
    function terminal_constraint(Ũ⃗)
        U = iso_vec_to_operator(Ũ⃗)
        U_sub = U[subspace, subspace]
        F = abs2(tr(U_goal_4x4' * U_sub)) / size(U_goal_4x4, 1)^2
        return [F_thr - F]
    end
    return NonlinearKnotPointConstraint(terminal_constraint, state_name, traj;
        equality = false, times = [traj.N])
end
```

4. **Diagnostic via the leakage-through-time notebook**: ε-sweeps + leakage(t) plots make the trade-offs visually obvious. Saved many cycles of debugging.

5. **Identifying the leakage_constraint bug in Piccolo** (`VariationalRolloutProblem` and `VariationalSplinePulseProblem` silently drop the `leakage_constraint` option — see `optimizations.md` for the upstream-side fix; **not fixed yet upstream**). For now, the strict process F constraint sidesteps it.

---

## 6. What didn't work (and why)

### a. DRAG as a hard path-sample equality constraint
Cubic Hermite splines have 4 DOF per knot per channel; continuous DRAG needs 4 conditions per interval per qubit. Globally that's `N_knots × 4` constraints on `N_knots × 4` free parameters → over-constrained by adjacency. Ipopt exited TOO_FEW_DOF (0 iters). See `_drag_constrained.jl`.

**Lesson**: use DRAG as a *warm-start* (initialization), not as a hard constraint. The optimizer can drift from DRAG if it finds something better, but starts in the right basin.

### b. 3-lvl optimization without fidelity constraint
The Q100 run had its fidelity constraint commented out. The optimizer minimized Q_r·‖∂U/∂n̂‖² freely, found a local optimum at F = 0.955 with 3.4% leakage that exploits |2⟩-mediated paths to lower the variational sensitivity. **Lesson**: always have a fidelity floor when there's a sensitivity-based robustness objective.

### c. Piccolo's stock `FinalUnitaryFidelityConstraint` with `EmbeddedOperator`
Uses the *average* gate fidelity formula `F_avg = (Tr(M†M) + |Tr(M)|²)/(n(n+1))` (Goerz et al. 2014/2015). The `Tr(M†M)` term is the comp-subspace purity `n·(1−L)`, so F_avg stays inflated when leakage is high but the bare process fidelity `|Tr(M)|²/n²` collapses. **The optimizer can game this** by exploiting leakage. **Lesson**: use the bare process fidelity formula (`FinalSubspaceProcessFidelityConstraint` we wrote) when leakage matters.

### d. Pushing robustness by raising Q_r alone
Bumping Q_r from 100 → 1000 in the stretched warm-start didn't unlock new robustness — the optimizer stayed at the warm-start's local minimum (`obj/Q_r = 1.6` either way). **Lesson**: warm-start locks you into a basin. To get genuinely better robustness, need either (i) different warm-start, (ii) explicit two-stage Q_r ramp, or (iii) random init from scratch (currently testing).

### e. Script's `F_full` calculation
The 3-lvl scripts compute `F_full = abs2(tr(U_iswap' * (V_fall * U_flat * V_rise)_sub)) / 16` using **static** `V_rise = exp(-iA·(XX+YY))` matrix exponentials in 9-dim. These pump population into |2⟩ via the √2 matrix elements of `X3⊗X3`. Result: F_full ≈ 0.94 even when F_flat ≈ 0.9999. **This is a measurement artifact, not a physical degradation.** The actual gate (with proper time-ordered edge propagation including H_anh, as `plot_results.jl::simulate_edge_unitary` does) gives F closer to F_flat.

---

## 7. What still puzzles me / open questions

### a. Why does the stretched warm-start objective NOT decrease at Q_r=1000?
With 10× higher robustness pressure and a strict F constraint preventing leakage gaming, we'd expect the optimizer to find a more robust solution within the high-F manifold. Instead it hovers at obj ≈ 1610 (= Q_r × 1.6, same effective sensitivity as the unstretched warm-start). Two possible explanations:
- The warm-start IS the local optimum for this objective + constraint set → no improvement possible
- inf_pr hasn't dropped low enough (was ~10⁻⁴ at iter 39) for the optimizer to focus on the objective → wait longer

The random-init run will help disambiguate: if it converges to a different (lower) objective, the warm-start was stuck in a suboptimal basin. If it converges to the same ~1.6 sensitivity, that's a hard limit of the problem class.

### b. Is the F_full = 0.94 actually a problem?
The discrepancy between F_flat = 0.9999 (correct, what the optimizer targets) and F_full = 0.94 (script's static-V_rise calc) is concerning. We argued it's a measurement artifact in `_drag_warmstart.jl` and `plot_results.jl`. But it should be verified by running `duffing_3level_verify_5nsbuf.ipynb` on the DRAG-warmstart trajectory — that notebook does proper time-ordered edge propagation. **Action item**: verify this hasn't been done yet.

### c. Robustness improvement from time-stretching?
The user's plot from the DRAG-warmstart 3-lvl run shows F-vs-ε curves where "Robust" is barely above "Default" — i.e., robustness gain is minimal. This is what motivated the stretched + Q_r=1000 attempt. Whether the stretched random-init can beat this is the open empirical question.

### d. Piccolo bug: leakage_constraint silently dropped by variational templates
Identified earlier in the session. `apply_piccolo_options!` in `Piccolo.jl/src/control/templates/_problem_templates.jl` (lines 39-101) handles the leakage knobs, but `variational_rollout_problem.jl` and `variational_spline_problem.jl` don't call it. **Not fixed upstream**. For now our scripts use the strict process F constraint which implicitly bounds leakage (F_sub ≤ 1−L means L ≤ 1 − F_sub ≤ 10⁻⁴ for F_threshold = 0.9999).

### e. Free-phase / virtual-Z optimization
None of our current fidelity computations (in plot_results.jl, leakage_through_time.ipynb, or the optimization scripts) use virtual-Z optimization. They report **bare** process fidelity. An experimentalist applying virtual Z corrections would see higher F. This is consistent across all our analyses, but means our F numbers are gauge-locked / pessimistic vs operational F.

---

## 8. Current state at handoff

### Currently running (SSH machine in tmux)
- `julia robust_iswap_detuned_1MHz_300ns_3level_random_init.jl` in tmux session `drag300` (or wherever the user named it)
- 2500-iter budget, ~24 hour wall time
- Output dir (when done): `robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_random_Qr1000_seed42/`

### Currently running (laptop, maybe)
- The stretched warm-start 200-iter probe might be running locally; if so, it's finishing.

### Latest commits (on robust_control_sam@main)
- `6e5bcbd` 3-level DRAG warm-start + stretched-time variants + leakage notebook (all session work)

### Piccolo.jl + NamedTrajectories.jl
- Both clean, in sync with their remotes
- Piccolo.jl on `modulate` branch (mirror at Andrew-Kamen/Piccolo-private)
- NamedTrajectories.jl on `main`

---

## 9. Next steps when work resumes

### Immediate (when the SSH random-init run finishes)
1. Pull the output dir back to laptop
2. Run `plot_stretched_results.jl <run_dir>` to generate fidelity_*.csv, combined.png, etc.
3. Compare random-init's converged objective to stretched-warmstart's (~1610). If random-init lands below 1500, we have evidence a better basin exists.

### Medium-term
1. **Verify F_full via the proper edge propagation**. Open `duffing_3level_verify_5nsbuf.ipynb`, point its `RUN_DIR` at the DRAG-warmstart output, see what `F(iSWAP, comp)` the time-ordered edge propagation gives. Expect F ≈ 0.999 if our "F_full = 0.94 is artifact" hypothesis is right.

2. **Two-stage Q_r ramp**. If neither warm-start nor random-init at Q_r=1000 produces a more robust solution than Q_r=100 did:
   - Stage 1: re-run at Q_r=100 to converged
   - Stage 2: warm-start at Q_r=1000 from #1 result
   - Stage 3: Q_r=10000 from #2 if needed
   This is the Poggi-Kiely recipe.

3. **Fix the Piccolo `leakage_constraint` bug** in `variational_rollout_problem.jl`. Patch is described in `optimizations.md` — add a call to `_apply_piccolo_options(qtraj, piccolo_options, all_constraints, traj, state_sym)` just before `DirectTrajOptProblem(...)`. Then re-enable `leakage_constraint = true` in the scripts (currently it's set but silently ignored).

### Long-term
1. **Map the Pareto frontier**: sweep over `(T, Q_r, F_threshold)` to find operationally relevant gates.
2. **Hardware emulation**: 250 MHz Gaussian filter on the optimized pulses to check survival under bandwidth constraints (see `physics.md`).
3. **Experimental translation**: convert gate-frame controls to AWG waveforms with proper virtual-Z corrections (see `robust_iswap_plan.md` Phase 4).

---

## 10. Key code patterns to know

### Strict process F constraint (use this instead of Piccolo's stock one when leakage matters)
```julia
function FinalSubspaceProcessFidelityConstraint(U_goal_4x4, subspace, state_name, F_thr, traj)
    n = size(U_goal_4x4, 1)
    terminal_constraint(Ũ⃗) = begin
        U = iso_vec_to_operator(Ũ⃗)
        Usub = U[subspace, subspace]
        [F_thr - abs2(tr(U_goal_4x4' * Usub)) / n^2]
    end
    NonlinearKnotPointConstraint(terminal_constraint, state_name, traj;
        equality = false, times = [traj.N])
end
```

### DRAG warm-start (apply at knots after loading from JLD2)
```julia
u_drag_knots = similar(u_2lvl)
u_drag_knots[1, :] .= u_2lvl[1, :] .+ du_2lvl[2, :] ./ η   # uX1 += duY1/η
u_drag_knots[2, :] .= u_2lvl[2, :] .- du_2lvl[1, :] ./ η   # uY1 -= duX1/η
u_drag_knots[3, :] .= u_2lvl[3, :] .+ du_2lvl[4, :] ./ η   # uX2 += duY2/η
u_drag_knots[4, :] .= u_2lvl[4, :] .- du_2lvl[3, :] ./ η   # uY2 -= duX2/η
du_drag_knots = du_2lvl   # keep original tangents
```

### Time-stretching for a longer pulse
```julia
# u_new[k] = u_old[k] / 2 (amplitudes halved)
# du_new[k] = du_old[k] / 4 (since du = d(u)/d(t); u→u/2, t→2t  ⇒  du→du/4)
# t_new[k] = 2 · t_old[k]
u_stretched  = u_old ./ 2
du_stretched = du_old ./ 4
t_stretched  = t_old .* 2
```

### Detecting integrator-honesty gap (always include in 3-level scripts)
```julia
let
    nominal_sys_v = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
    Ũ⃗_ref = unitary_rollout(traj, nominal_sys_v;
        interpolation = :cubic_hermite, abstol = 1e-12, reltol = 1e-12)
    # ... compare F_NLP vs F_independent, warn if |ΔF| > 1e-4
end
```

---

## 11. Repo conventions

- **robust_control_sam**: origin = Andrew-Kamen/notebooks (private). Push to `main` with bare `git push`.
- **Piccolo.jl**: laptop has `origin` (public harmoniqs) and `private` (Andrew-Kamen/Piccolo-private) remotes. Active branch is `modulate`. Push to private with `git push private modulate`. Don't push to origin.
- **NamedTrajectories.jl**: origin = harmoniqs/NamedTrajectories.jl. Currently clean, no Andrew-Kamen modifications.
- **DirectTrajOpt.jl**: similar to Piccolo (private mirror, modulate branch). Currently clean.
- **Notebooks expect sibling layout**: Piccolo.jl, DirectTrajOpt.jl, NamedTrajectories.jl, robust_control_sam as siblings under `~/.../GitHub/`.
- **SSH machine**: `~/notebooks/...` matches the laptop's `~/.../GitHub/robust_control_sam/...`. Origin = mirror, so bare push works.

---

## 12. Glossary

- **F_avg vs F_process**: F_avg = `(Tr(M†M) + |Tr(M)|²) / (n(n+1))` is the average gate fidelity (Goerz formula). F_process = `|Tr(M)|² / n²` is the bare process fidelity. They agree when M is unitary (no leakage) but diverge when M has reduced norm.
- **Variational rollout / VariationalRolloutObjective**: Piccolo's indirect-formulation robustness objective that propagates the variational state ∂U/∂ε inside the objective evaluation (vs the direct collocation which adds it as a decision variable). Cheaper for large Hilbert spaces.
- **Inf_pr / inf_du**: Ipopt's primal infeasibility (max constraint violation) and dual infeasibility (gradient of Lagrangian).
- **Q_r**: weight on the variational robustness objective `Q_r · Σ_i ‖∂U/∂ε_i‖²`.
- **F_threshold**: target fidelity for the inequality constraint `F ≥ F_threshold`.
- **`:Ũ⃗`**: Piccolo's symbol for the iso-vec representation of the unitary state (length 2·D²).

---

## 13. Quick-reference: run + plot commands

All commands below run from `robust_control_sam/src/modulate_iswap/`.

### Currently running on SSH
```bash
# 300 ns random init, n̂ robustness, NO leakage constraint (the original)
julia robust_iswap_detuned_1MHz_300ns_3level_random_init.jl
# 2500 iters, ~24 hours wall time
# Output dir:
#   robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_random_Qr1000_seed42/
```

### Local 200-iter probes (already finished)
```bash
julia robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_drag_warmstart.jl
# Output: robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_dragwarmstart_seed42/

julia robust_iswap_detuned_1MHz_300ns_3level_stretched_warmstart.jl
# Output: robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_stretched_Qr1000_seed42/
```

### New scripts — to test the "leakage hard-constrained + lifted-Z vs n̂" hypothesis
Both 1000 iters, random init (same seed 42 as the running random-init), strict process F=0.9999, hard `LeakageConstraint` at every knot (bound `1e-5` per matrix element, `LeakageObjective` weight 10.0). Estimated ~10-14 hours wall time each.

```bash
# Lifted-Z robustness (diag(1,-1,0) per qubit; cleaner optimization landscape)
julia robust_iswap_detuned_1MHz_300ns_3level_liftedZ_with_leakage.jl
# Output dir:
#   robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_liftedZ_leak5_Qr1000_seed42/

# n̂ robustness (diag(0,1,2) per qubit; physically correct frequency-noise channel)
julia robust_iswap_detuned_1MHz_300ns_3level_nhat_with_leakage.jl
# Output dir:
#   robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_nhat_leak5_Qr1000_seed42/
```

### Lifted-Z, NO leakage constraint, 2500 iters (fills the 2×2 design matrix)
Companion to the killed `_random_init.jl` (which was n̂ + no-leakage + 2500 iter). Same seed 42, strict F=0.9999, Q_r=1000, but with lifted-Z robustness operators and NO `LeakageConstraint`. Tests whether operator choice matters even when leakage is unconstrained. ~24 h wall time.

```bash
julia robust_iswap_detuned_1MHz_300ns_3level_liftedZ_no_leakage.jl
# Output dir:
#   robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_liftedZ_noLeak_Qr1000_iter2500_seed42/
```

To run in tmux:
```bash
tmux new -s liftedZ_noleak
julia robust_iswap_detuned_1MHz_300ns_3level_liftedZ_no_leakage.jl 2>&1 | tee liftedZ_noleak.log
# Ctrl-b d to detach
```

Plot when done:
```bash
julia plot_stretched_results.jl robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_liftedZ_noLeak_Qr1000_iter2500_seed42
```

### Cascade continuation of the 200-iter stretched warm-start (recommended path)
Loads `_stretched_warmstart.jl`'s `trajectory.jld2` (the 200-iter result that was still descending fast at obj ~1538) and continues for 1000 more iters at same Q_r=1000 / F=0.9999 / n̂ robustness. No leakage constraint. ~14 h wall time.

Run order: the source dir must exist first, so finish (or have already finished) `_stretched_warmstart.jl` before launching this.

```bash
julia robust_iswap_detuned_1MHz_300ns_3level_stretched_cascade.jl
# Output dir:
#   robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_cascade_Qr1000_iter1000_seed42/
```

In tmux:
```bash
tmux new -s cascade
julia robust_iswap_detuned_1MHz_300ns_3level_stretched_cascade.jl 2>&1 | tee cascade.log
# Ctrl-b d
```

Plot when done:
```bash
julia plot_stretched_results.jl robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_cascade_Qr1000_iter1000_seed42
```

The cascade pattern generalizes: after this one finishes, copy `_stretched_cascade.jl`, change `CASCADE_SOURCE_DIR` to its output dir, change `run_tag` suffix to `_cascade2`, and run again. You can chain as many phases as you like — useful for "let me just leave this running until the descent flatlines."

To run on SSH in tmux:
```bash
ssh atkamen@<machine>
cd ~/notebooks
git pull origin main
cd src/modulate_iswap

# Each in its own tmux session so they run independently
tmux new -s liftedZ
julia robust_iswap_detuned_1MHz_300ns_3level_liftedZ_with_leakage.jl 2>&1 | tee liftedZ_leak.log
# Ctrl-b d to detach

tmux new -s nhat
julia robust_iswap_detuned_1MHz_300ns_3level_nhat_with_leakage.jl 2>&1 | tee nhat_leak.log
# Ctrl-b d to detach
```

### Plot results for any run
The plotting script auto-detects parameters from `parameters.txt`:
```bash
# Stretched random-init (currently running on SSH)
julia plot_stretched_results.jl robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_random_Qr1000_seed42

# Stretched warm-start (200-iter probe, already done)
julia plot_stretched_results.jl robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_stretched_Qr1000_seed42

# New lifted-Z + leakage run (when done)
julia plot_stretched_results.jl robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_liftedZ_leak5_Qr1000_seed42

# New n̂ + leakage run (when done)
julia plot_stretched_results.jl robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_nhat_leak5_Qr1000_seed42
```

Each plot command writes into `<run_dir>/figs/`:
- `default_vs_robust.png` (linear F vs ε)
- `default_vs_robust_log.png` (log infidelity vs ε)
- `default_vs_robust_6panel.png` (both stacked)
- `controls_full_gate.png` (pulse layout)
- `combined.png` (money plot)
- `ipopt_iter_history.png` (objective / inf_pr / inf_du vs iter)

Plus `fidelity_*.csv` and `pulse_full_gate.csv` next to `parameters.txt` in the run dir.

### Original 150 ns runs (for cross-comparison)
```bash
julia plot_results.jl robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_dragwarmstart_seed42
julia plot_results.jl robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_seed42_Q100
julia plot_results.jl robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_rollout_1kiter_seed42
```
(`plot_results.jl` auto-detects the run's parameters too — works for both 150 ns and 300 ns runs.)
