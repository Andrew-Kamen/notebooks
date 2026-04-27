# Robust iSWAP QSL Sweep — Experiment Plan

## Goal

Empirically map the practical "robust quantum speed limit" for the detuned-frame
iSWAP gate as a function of coupling strength `g_eff` and gate time `T`, using
fixed knot count `N`. The headline deliverable is a family of Pareto fronts in
`(T, robustness)` space, one per `g_eff`, that quantify when robust control
adds value vs. the bare Gaussian-square baseline.

## Sweep variables

Three axes, only two swept simultaneously per experiment:

| Variable | Role | Range |
|---|---|---|
| `g_eff` | coupling strength (drift) | 2, 5, 10 MHz |
| `T_total_ns` | total flat-top duration | 60–250 ns (warm-start ladder, log-spaced) |
| `N_knots` | spline knot count | fixed at saturated value (probably 24 or 48) |

`a_bound = 10 MHz` and `F_threshold = 0.999` (lowered from 0.9999 to give
optimizer trade-room) held constant across all sweeps.

**Why fixed `N` during T sweep:**
- Same NLP problem size keeps optimizer behavior comparable.
- Knots/ns naturally rises as T shrinks, providing the higher control
  bandwidth that short gates need.
- Stays valid as long as `knots/ns ≤ B_hw_Nyquist ≈ 0.5` — for `N = 24` this
  means `T ≥ 48 ns`. Below that, fixed `N` represents pulses the hardware
  can't reproduce.

## Phase 1 — `N` saturation calibration

Establish the saturated knot count so subsequent sweeps aren't biased by
discretization choice. Single calibration point.

- Fix `(g_eff, T) = (5 MHz, 100 ns)` — middle of intended range.
- Sweep `N ∈ {12, 24, 48, 96}`.
- Plot achieved robustness (variational objective + ε-sweep width) vs `N`.
- Pick the smallest `N` past which improvement is < 5%.

Estimated wall time: ~1–2 h sequential.

## Phase 2 — main `T` sweep at each `g_eff`

The scientific core. Three Pareto fronts.

```
For g_eff in [2, 5, 10] MHz:
    For T in [250, 180, 130, 90, 60] ns (warm-start ladder, descending):
        Build qcp at this (g_eff, T) with N from Phase 1
        If T == 250 ns: random init
        Else:           warm-start from previous T's traj (squish + slope rescale)
        Solve (1000 iters, F_threshold = 0.999)
        Save trajectory + metrics
```

Each run produces:
- `traj.jld2` (knots, tangents, times, components)
- `metrics.csv` row: `g_eff, T, F_robust, E_V, F_at_eps_max, robust_width, exit_status`

Estimated wall time: 5 T × 3 g_eff × ~40 min = ~10 h sequential, ~3.5 h in
3 parallel processes.

## Phase 3 — multi-seed verification at elbow points

Hedge against single-chain basin trapping. The shortest T per g_eff (where
warm-start is most likely to be in the wrong basin) gets multiple seeds.

```
For g_eff in [2, 5, 10] MHz:
    T_elbow = shortest T from Phase 2 that still gave a robust solution
    For seed in [1, 2, 3]:
        Random init, solve at (g_eff, T_elbow, N)
    Compare to warm-start result; keep best
```

Estimated wall time: 3 g_eff × 3 seeds × ~40 min = ~6 h sequential.

## Phase 4 (optional) — `N` confounder check at sweep endpoints

For the shortest and longest T per g_eff, redo with `N = 96` (over-saturated)
to confirm the Phase 2 results aren't knot-limited.

- If results unchanged: clean data, fixed `N` was justified.
- If results improve: revise fixed `N` upward, re-run affected points.

Estimated wall time: ~4 h.

## Robustness metric — pick one and stick with it

Avoid post-hoc metric shopping. Decide before running:

- **`E_V`** (variational objective): terminal-only first-order sensitivity.
  Lower = flatter top of `F(ε)` curve. Direct optimizer target.
- **`robust_width`**: largest `|ε|` for which `F(ε) > F_threshold`.
  Practical experimental quantity.
- **`F_at_eps_max`**: `F(ε = 0.02 rad/ns)` (or whatever your worst-case ε is).

Recommendation: use `robust_width` as the primary scalar, `E_V` as the
diagnostic. Save all three per run so post-hoc analysis isn't gated on
one choice.

## Headline deliverable

Three curves, one per `g_eff`, of `robust_width` (or `1 − F_at_eps_max`) vs `T`.

```
robustness (e.g. width or improvement)
 ↑
 |  ╲╲              ← 2 MHz curve: robust dominates everywhere
 |   ╲ ╲
 |    ╲ ╲╲
 |     ╲  ╲╲
 |      ╲   ╲╲      ← 5 MHz curve: still wins broadly
 |       ╲    ╲╲
 |        ╲     ╲╲
 |         ╲      ╲╲ ← 10 MHz curve: thin advantage at long T
 |          ╲       ╲╲     possibly negative at short T
 |─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╲╲─ ── 0 (no improvement)
 |
 |───────────────────────→ T (ns)
   shorter ←        → longer
```

Crossover behavior — where the 10 MHz curve dips below the 2 MHz curve, or
where any curve crosses zero — is the physics story: **robust control adds
value precisely when the GS baseline is slow enough to accumulate coherent
error.** At high coupling, "fast and clean" beats "slow and robust."

## Theoretical context (what to expect)

For first-order canceled robust pulse: `1 − F_rob ~ ε⁴ T_rob⁴`.
For GS default: `1 − F_GS ~ ε² T_GS²` where `T_GS = 2·buf + θ_goal/g_eff`.

Two floors stack to set the practical robust QSL:

- **Coupling floor**: `T_iSWAP = θ_goal / g_eff` (linear in `1/g_eff`).
- **Cancellation floor**: `~ N_passes · π / a_bound` ≈ 50 ns (independent of `g_eff`).

Total robust floor `≈ T_iSWAP + T_cancellation`:

| `g_eff` | Coupling floor | Cancellation floor | Predicted T_min |
|---|---|---|---|
| 2 MHz | 60 ns | 50 ns | ~110 ns |
| 5 MHz | 24 ns | 50 ns | ~75 ns |
| 10 MHz | 12 ns | 50 ns | ~62 ns |

If the data shows the cancellation floor (~50 ns) is invariant across `g_eff`,
that's evidence the floor is set by microwave bandwidth, not coupling.

## Practical notes

### Hamiltonian
Use `H_gate_frame` from `robust_iswap_detuned*.ipynb` (no static `δ₁₂/2 IZ`
drift; on-resonance microwaves; constant `g_eff (XX+YY)` coupling). This is
the internally consistent rotating frame.

### Initialization
Use `CubicSplinePulse(us, dus, ts)` for the initial Ũ⃗ rollout (cell 13
pattern). Linear init causes O(10) initial `inf_pr` and wastes iterations on
feasibility recovery. Cubic init typically halves the iters-to-convergence.

### Warm-start (squish-and-rescale)
For T ladder, warm-start from previous T:
```julia
α = T_new / T_old
times_new = α .* vec(traj_old[:t])
u_new     = traj_old[:u]                 # values unchanged
du_new    = traj_old[:du] ./ α           # slopes scale inversely
pulse_warm = CubicSplinePulse(u_new, du_new, times_new)
```
For ≤ 20% T changes, basin transfer is reliable. Larger jumps need random
restart.

### Verification (cell 24 pattern)
ε sweep uses `CubicSplinePulse(us, dus, ts)` to match NLP integrator.
Default GS sweep uses `LinearSplinePulse` (matches its construction).
Don't mix. The robust curves verified with linear interp are plain wrong
(this was the bug we fixed in `robust_iswap_detuned.ipynb` cell 24).

### Output organization
```
results/
  qsl_sweep/
    geff2MHz_T250ns_N24_seed42/
      traj.jld2
      metrics.csv
      ipopt.log
    geff2MHz_T180ns_N24_seed42/
      ...
    ...
```
Dynamic dir naming by `(g_eff, T, N, seed)` so reruns don't overwrite and
post-hoc analysis is straightforward.

## Compute budget summary

| Phase | Time (sequential) | Time (3-way parallel) |
|---|---|---|
| 1 — N calibration | 1–2 h | 1–2 h (single point) |
| 2 — main T sweep | ~10 h | ~3.5 h |
| 3 — multi-seed verify | ~6 h | ~2 h |
| 4 — N confounder | ~4 h | ~1.5 h |
| **Total** | **~22 h** | **~9 h** |

Doable in a long weekend or two overnight runs.

## Open questions / risks

- **`g_eff = 10 MHz` may have very narrow robust regime** (predicted floor
  ~62 ns; current 240 ns runs show modest improvement). If 10 MHz never
  beats GS for any T, that's still a result — but worth checking that
  it's not a multi-seed / convergence artifact before publishing.

- **Fixed `N` may be too coarse at long T.** If `N = 24` and `T = 250 ns`,
  knot spacing is 10 ns, well under-resolved relative to hardware
  bandwidth. Phase 4 catches this.

- **The terminal-only `E_V` metric may favor "narrow-but-flat" over
  "wide-and-good."** If results disagree with `robust_width`, prefer the
  practical metric. (Source modification to penalize multiple times is a
  separate project.)

- **Hardware bandwidth assumption.** Numbers above assume `B_hw = 250 MHz`
  Gaussian filter. If actual hardware is faster/slower, the cancellation
  floor moves.
