# Hadamard Leakage Testbed

Single-qubit Hadamard gate used as a methodology testbed for a leakage-aware,
robust optimal-control pipeline that we intend to scale up to a two-qubit
iSWAP gate.

## Goal

Develop a 2-lvl-only optimization recipe that, when honestly verified in 3 lvl,
delivers:

- High in-subspace fidelity (≥ four nines on F₂; high three-nines on F₃)
- Low leakage to |2⟩ (L₃ ≲ 10⁻⁵)
- Robustness against static σ_z dephasing (variational susceptibility χ_z)

The 2-lvl restriction is a compute-budget constraint: full 3-lvl optimization
is too expensive to run on the eventual 2-qubit problem.

## Physics

In the rotating frame at ω₁₀, the drive Ω(t) = u_X(t) + i·u_Y(t) couples
the qubit to the leakage level |2⟩ with matrix element √2·Ω(t), off-resonant
by η = ω₂₁ − ω₁₀ (negative for transmon).

First-order time-dependent perturbation theory gives the leakage amplitude
at gate end as a Fourier coefficient of the envelope evaluated at ω = −η:

    c₂(T) ≈ −i·√2·∫₀ᵀ Ω(t)·c₁(t)·exp(+iη·t) dt

To leading order (c₁ ≈ const), this reduces to the **bare spectral proxy**:

    P_leak ≈ 2·|Ω̂(η)|²

The optimization strategy is to suppress |Ω̂(η)|² inside the optimizer
(2-lvl only) so that 3-lvl honest verification yields low leakage. Full
derivation in `SPECTRAL_LEAKAGE_DERIVATION.md`.

## Approaches tried

### Discarded

- **Direct 3-lvl optimization.** Compute-prohibitive at the 2-qubit
  problem scale. Used early as a ground-truth check; the per-T sweep
  directories were not informative beyond confirming feasibility.
- **Two-tone / single-tone modulated drive.** Putting a counter-tone at
  +|η| produces a spectral hole at the leakage frequency, equivalent to
  ALC. Useful for intuition but not the primary line we pursued.
- **Soft spectral barrier objective only.** Adding only a quadratic
  penalty R·|Ω̂(η)|² to the objective did not give tight enough leakage
  control; the constraint form (hard inequality) was needed.

### Kept, actively used

- **Spectral leakage constraint** (`SpectralLeakageConstraint`): hard
  inequality |Ω̂(η)|² ≤ ε_max. Analytic Jacobian and Hessian of the
  Lagrangian. Drop-in nonlinear constraint in the spline NLP.
- **Variational σ_z robustness penalty** (Q_r): adjoint-based sensitivity
  ‖∂U/∂ε‖² evaluated at gate end via `VariationalSplinePulseProblem`.
- **Combined-amplitude path constraint** at sub-knot samples: ensures
  |Ω(t)|² ≤ a_bound² between the spline knots, not just at them.
- **Post-hoc virtual-Z correction**: best 2×2 diagonal phase correction
  to recover the AC-Stark-induced Z phase that the 2-lvl optimizer
  cannot see. Analytic: F_corr = (|A₁₁| + |A₂₂|)²/4, A = U_target†·U_sub.
- **ALC-style modulated drive** (`hadamard_alc_modulated.ipynb`): 4-channel
  drive with a tone modulated at +|η|, equivalent (post-hoc) to a notched
  envelope. Status: in-progress comparison line.

## Current methodology

The "winning recipe" for 2-lvl optimization is:

1. `VariationalSplinePulseProblem` over (u_X, u_Y), cubic Hermite spline
   parameterization, N_knots ~ 15.
2. Smoothness regularization R·du² + R_ddu·ddu² with ddu_bound.
3. Combined-amplitude path constraint on |u_X² + u_Y²|.
4. `FinalUnitaryFidelityConstraint` against U_target = Hadamard.
5. `SpectralLeakageConstraint(:u, |η|, ε_max, traj)`.
6. Optional `SpectralLeakageObjective` as a soft barrier for gradient
   smoothness near the active constraint.
7. Variational generator: σ_z, with Q_r ∈ {0, Q_R_ROBUST} to compare
   non-robust vs robust pulses.

Honest 3-lvl verification done post-hoc with a fine cubic-Hermite
propagator over a 4000-step grid.

## Current best results (T_NS = 50, η/2π = −170 MHz)

|                  pulse              | F₂        | F₃ raw    | F₃ + vZ   | L₃         |
|-------------------------------------|-----------|-----------|-----------|------------|
| Default + spectral constraint       | 0.999998  | 0.988585  | 0.988583  | 8.52e-06   |
| Robust  + spectral constraint       | 0.999998  | 0.995866  | 0.997437  | 3.28e-06   |

Post-hoc spectral notch reduces L₃ by an additional ~10× without breaking
F₂ at T = 50 ns; effect disappears at T = 200 ns where |Ω̂(η)|² is already
intrinsically tiny.

The ~1% (default) / ~0.25% (robust) F₃ residual that *isn't* removable by
virtual-Z is identified as **higher-order off-resonant coupling**: the
Schrieffer-Wolff 2nd-order correction from |2⟩ produces a non-σ_z piece in
the effective qubit Hamiltonian (drive-amplitude renormalization, commutator
residual between Stark σ_z and the σ_xy drive). This is intrinsic to the
2-lvl approximation and cannot be removed by spectral methods alone.

## Open questions / next steps

- **3-lvl warm-started polish** (in development). Initialize 3-lvl optimizer
  from the 2-lvl-converged pulse, use embedded operator goal, n̂ robustness
  in `H_vars_fn`, same spectral leakage constraint. Compare same-iteration
  warm-start vs cold-start to decide whether the recipe is mature enough
  to commit 2Q compute.
- **Banded spectral notch.** Promote the single-frequency constraint at
  ω = |η| to multiple constraints across a band around |η|. Bounds
  transient leakage during the gate (not just at t = T), keeping the
  qubit-subspace assumption valid throughout the trajectory.
- **Time-distributed robustness.** Move `robust_times = [[T]]` to
  intermediate times so the σ_z susceptibility penalty is active during
  the gate, not just at the end.
- **Two-qubit iSWAP.** The methodology generalizes via:
  - Multiple `SpectralLeakageConstraint` instances at the relevant 2Q
    leakage frequencies (|η_1|, |η_2|, |11⟩↔|20⟩, |11⟩↔|02⟩, etc.)
  - Embedded operator goal in a truncated 3-lvl-per-qubit Hilbert space
    (9-dim minimum)
  - Variational generators: n̂_1, n̂_2, and possibly n̂_1·n̂_2 for ZZ-correlated
    phase robustness on |11⟩.
  - Same warm-start recipe: 4-dim 2-lvl pre-pass → 9- or 16-dim 3-lvl polish.

## File index

Active:

- `hadamard_spectral_proxy_constraint.ipynb` — main notebook, current
  methodology. Contains `SpectralLeakageObjective` + `SpectralLeakageConstraint`
  struct definitions, `optimize_2lvl()` builder, ε-sweep verification,
  post-hoc virtual-Z code, and (in progress) the 3-lvl warm/cold polish
  comparison.
- `SPECTRAL_LEAKAGE_DERIVATION.md` — formal derivation of the first-order
  leakage susceptibility and the spectral proxy.
- `DRAG_ALC_DOCUMENTATION.md` — earlier write-up on the DRAG/ALC
  perspective, kept for cross-reference.
- `hadamard_alc_modulated.ipynb` — in-progress 4-channel ALC implementation,
  parallel line to the constraint approach.
- `hadamard_alc_comparison.ipynb` — in-progress comparison notebook
  between ALC and notch/constraint approaches.
- `spectral_sweep_robust.jl` — Pareto sweep: ε_max ∈ [1, 1e-5] log-spaced,
  warm-chained, writes to `spectral_sweep_results/`.
- `plot_sweep_fidelity.jl` — aggregate plot script over the sweep results.
- `check_fidelity_tightest.jl` — honest 3-lvl verification of the tightest
  ε_max trajectory in the sweep.
- `robust_hadamard_2lvl.ipynb` — early starting-point notebook for the
  2-lvl variational robustness (before leakage work).
- `hadamard_spectral_proxy.ipynb` — earlier soft-barrier-only version of
  the spectral approach; superseded by the constraint version but kept
  for reference.

Output figures:

- `spectral_F_vs_eps_2lvl.png`, `spectral_F_L_vs_eps_3lvl.png`,
  `spectral_F_L_vs_eps_3lvl_virtualZ.png` — fidelity / leakage vs ε
  (the σ_z dephasing axis).
- `spectral_freq_domain.png`, `spectral_time_domain.png`,
  `spectral_notch_freq.png`, `spectral_notch_time.png` — diagnostic
  spectra and time-domain views of the optimized pulses, raw vs notched.
- `alc_*` — ALC-line figures.

Data:

- `spectral_sweep_results/` — `.jld2` trajectories from the Pareto sweep,
  plus aggregated `fidelity_sweep.jld2` and `sweep_summary.jld2`.
- `Project.toml`, `Manifest.toml` — environment for this subfolder.

## Design notes

- η_anh = −2π·0.170 rad/ns (transmon-typical anharmonicity).
- a_bound = 2π·0.01 rad/ns (drive amplitude cap).
- T_NS ∈ {50, 200} ns explored. Shorter T is harder (more spectral mass
  near |η|); longer T eases the leakage constraint but increases
  exposure to dephasing.
- Q_R_ROBUST and ε_MAX are the two knobs that produce the robustness ↔
  leakage Pareto. With the current setup the variational susceptibility
  is mostly flat across the leakage Pareto — no strong tradeoff observed
  at fixed T.
