# Robust iSWAP — computational subspace warm-start

Successor folder to `../hadamard_leakage_tradeoff/`. The Hadamard testbed
validated the spectral-notch + 2-lvl-optimization + 3-lvl-warm-start-polish
recipe at the single-qubit level. This folder applies that recipe to the
research target: a robust iSWAP gate.

## What we learned from the Hadamard testbed

(Full details in `../hadamard_leakage_tradeoff/HADAMARD_LEAKAGE_TESTBED.md`.)

- Spectral notch as a hard constraint pushes 3-lvl leakage to 10⁻⁶ from a
  2-lvl-only optimization.
- Variational σ_z robustness composes cleanly with the notch.
- 2-lvl is capped at ~3-nines by AC Stark + commutator residual.
  Warm-started 3-lvl polish closes the gap (default case: F₃ → 0.99999 in 18 s).
- Methodology transfers to 2Q with: (i) multiple spectral notches at the
  relevant 2Q leakage frequencies, (ii) embedded-operator goal in a
  higher-dim Hilbert space, (iii) per-qubit n̂ robustness in the polish step.

## Current scope (2026-05-27)

**Today's focus: solve the 4-dim 2Q problem with spectral leakage constraint.**

- 4-dim Hilbert space (two 2-lvl qubits)
- Resonant exchange drift: `H = g_eff(t)·(XX+YY)` — exchange Hamiltonian
- g_eff envelope: 200 ns square, filtered by two cascaded Gaussians
  (η-killer at σ_f = 100 MHz, AWG at σ_f = 250/√(ln 2) MHz)
- MW controls active only in the flat region [T_RISE+10, T_FALL−10] = 180 ns
- Spectral leakage constraint `|Ω̂_i(2π·170 MHz)|² ≤ 1e-2` per qubit
- Variational σ_z robustness on each qubit (ZI, IZ generators), `Q_r = 1e2`
- Target: `U_goal = V_fall† · U_iSWAP · V_rise†` (V†·U·V† method, mirrors
  modulate_iswap convention)
- 15 spline knots, cubic Hermite
- Save IPOPT log + final trajectory to JLD2 for downstream warm-start polish

**Baseline ("default") = filtered Gaussian-square g_eff with NO microwaves**,
evaluated with the H4 virtual-Z + ZZ corrections from
*Modular Quantum Processor with an All-to-All Reconfigurable Router*:

  `U_iSWAP_paper = diag(1, 0, 0, 0; 0, 0, −i·e^{i α_2}, 0; 0, −i·e^{i α_1}, 0, 0; 0, 0, 0, e^{i(θ_zz+α_1+α_2)})`

The 3 virtual parameters are scanned (2D grid over α₁, α₂ with analytic
θ_zz) to find the best post-hoc-corrected fidelity. This is what an
experimentalist would do with no MW-shaping work, only frame changes.

## Architecture decisions made today

- **g_max = 2 MHz, T_pulse = 200 ns.** The pulse over-rotates: total drift
  area ≈ 3.2·(π/4) = 3.2 iSWAPs. The V†·U·V† machinery accounts for this
  — the optimizer doesn't need a clean single-iSWAP envelope; it shapes
  the MW to land on `U_goal` exactly.
- **Edge fractions** (each side): ~32% of one full iSWAP per edge
  (~0.32·(π/4)). Flat-region drift carries the remaining 256% — large,
  but the MW navigates it.
- **iSWAP convention** (matches modulate_iswap): `H = g·(XX+YY)`, so on
  the {|01⟩,|10⟩} subspace `(XX+YY)` has eigenvalue ±2 and full iSWAP
  requires `∫g dt = π/4` (NOT π/2 — earlier off-by-2 fixed).
- **On-resonant drives** (Δ_mw = 0). User-confirmed simplification — both
  qubits at the same frequency for the gate. Detuning can be added later
  if the architecture demands it.
- **Default-case calibration** uses the H4 virtual-Z + ZZ form. Implementable
  as frame changes; no additional hardware burden.

## What lives in this notebook (`iswap_pulse_construction.ipynb`)

1. Pulse infrastructure: filtered g_eff envelope, flat-region demarcation,
   iSWAP-angle accounting.
2. Edge unitaries `V_rise`, `V_fall` and the flat-region target `U_goal`.
3. `SpectralLeakageConstraintIQ` — 2Q generalization of the 1Q
   `SpectralLeakageConstraint`. Selects index pairs (1,2) and (3,4) from
   the 4-channel `:u` for per-qubit IQ.
4. `H_flat_gate` (4-dim drift+MW), `VariationalQuantumSystem` with ZI/IZ
   variational generators, `VariationalSplinePulseProblem`.
5. Constraints: fidelity ≥ 0.9999, two SpectralLeakageConstraintIQ at
   2π·170 MHz with ε_max = 1e-2.
6. Solve cell: IPOPT 1000 iter, log to `ipopt_<RUN_TAG>.log`, trajectory
   to `traj_<RUN_TAG>.jld2` along with U_goal, V_rise/V_fall, edge areas,
   params.
7. Default baseline: drift-only propagator + virtual Z+ZZ correction.
8. Optimized result: V_fall · U_flat · V_rise + virtual Z+ZZ correction.
9. Full-pulse plot: g_eff + 4 MW channels (knots + fine sample), same
   visual style as modulate_iswap.
10. 2-lvl ε-sweep: F vs ε for ZI, IZ, ZZ channels (raw + virtual-corrected).
11. 3-lvl honest verification: 9-dim Hilbert (3-lvl per qubit), report
    F_3, L_3 at ε=0 and ε-sweep against ZI₍3₎, IZ₍3₎, n̂_1, n̂_2.

## Files

- `ROBUST_ISWAP_PROGRAM.md` (this file) — running plan and progress log.
- `iswap_pulse_construction.ipynb` — main notebook (see above).
- `Project.toml`, `Manifest.toml` — environment.
- Outputs (created on first run): `ipopt_<RUN_TAG>.log`,
  `traj_<RUN_TAG>.jld2`, `full_pulse_<RUN_TAG>.png`,
  `eps_sweep_2lvl_<RUN_TAG>.png`, `eps_sweep_3lvl_<RUN_TAG>.png`,
  `g_eff_filtered.png`, `g_eff_iswap_fraction.png`.

## Open / next

- Run the solve cell. Inspect convergence; confirm both spectral
  constraints active and feasible at the end.
- Compare optimized F (raw and virtual-corrected) vs default F (virtual-corrected).
  Expectation: default F should be ≤ optimized F — the optimization buys
  robustness at minimum and possibly raw fidelity too.
- Read the 3-lvl table at ε=0: if F₃ ≥ 0.9999 raw or with virtual
  correction, the 4-dim optimization is sufficient. If F₃ stalls at
  3-nines like 1Q did, plan a 3-lvl polish step (warm-started from this
  trajectory).
- Decide whether the next iteration drops g_max to ~1.25 MHz so the
  drift area cleanly equals π/4 (one iSWAP from drift alone), or keeps
  the over-rotated regime and lets the optimizer handle it.
- Eventually: warm-start a 9-dim 3-lvl-per-qubit polish using
  `traj_<RUN_TAG>.jld2` as the initial controls. Embedded-operator goal
  on the QSUB = [1,2,4,5] indices.
