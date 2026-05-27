# DRAG + ALC: paper math and notebook implementation

Reference: Ben Chiaro & Yaxing Zhang, *Active Leakage Cancellation in Single Qubit Gates*, PRL (2025), doi:10.1103/4kz9-w97h.

This document explains:
1. **What problem DRAG and ALC solve** (leakage from |1⟩→|2⟩)
2. **The two correction formulas** (paper Eq. 2 and Eq. 4)
3. **The complex-envelope generalization** to optimized pulses with both X and Y quadratures
4. **What's implemented** in `hadamard_alc_comparison.ipynb`
5. **Smoothness analysis** — why `R_ddu` matters
6. **Caveats**

---

## 1. The leakage problem

In a transmon (weakly anharmonic oscillator), the two lowest levels |0⟩, |1⟩ form the qubit. The third level |2⟩ is at energy
- $\omega_{12} = \omega_{01} + \eta$
- with $\eta < 0$ (transmon convention) and $|\eta| \ll \omega_{01}$.

When you drive the qubit at $\omega_{01}$ to do a gate like Hadamard, the drive is **off-resonant** with the |1⟩↔|2⟩ transition by amount $\eta$. But $|\eta|$ is small (~170 MHz vs ~5 GHz qubit freq), so the off-resonant excitation of |1⟩→|2⟩ is **not negligible**.

Result: gates incur **leakage** into |2⟩, especially for fast (short-T) pulses where the Fourier spectrum of the drive envelope has significant weight at $f_{21}$.

### Why leakage is bad

- |2⟩ has higher decoherence rates than |1⟩
- Crosstalk via |2⟩-mediated interactions with neighbors
- Quantum error correction can't handle leakage errors (codes assume comp subspace)

---

## 2. The two corrections (Chiaro-Zhang Eqs. 2 and 4)

The paper uses an analytical raised-cosine envelope $F(t)$ for the main pulse, then adds two corrections.

### Eq. 2 — DRAG quadrature on the main drive

> $\Omega_\text{main}(t) = A_\text{main}\,\left[F(t) + i\,\frac{\alpha}{2\pi\Delta}\,\frac{dF(t)}{dt}\right]\,e^{i 2\pi\delta_\text{main} t}$

Components:
- $F(t)$ — real envelope shape (raised cosine in the paper)
- $A_\text{main}$ — amplitude (rad/s)
- $\alpha = 1$ — DRAG parameter (fixed)
- $\Delta = -\eta - \delta_\text{main}$ — sets the spectral notch position
- $\delta_\text{main} = f_\text{main} - f_{10}$ — main drive detuning (compensates ac Stark)

**What it does:** the imaginary part (DRAG quadrature) adds a Y-component to the drive proportional to $dF/dt$. This places a **spectral notch in the drive at $f_{21}$**, killing the Fourier weight that would have excited |1⟩→|2⟩.

### Eq. 4 — ALC drive (additional tone)

> $\Omega_\text{ALC}(t) = i\,A_\text{ALC}\,\frac{1}{2\pi\delta_\text{ALC}}\,\frac{dF(t)}{dt}\,e^{i\,2\pi\delta_\text{ALC}\,t + i\phi_\text{ALC}}$

Components:
- $\delta_\text{ALC} = f_\text{ALC} - f_{10} \approx -\eta$ — second tone is at frequency $\approx f_{21}$
- $A_\text{ALC} \approx 0.15\,A_\text{main}$ with opposite sign (calibrated)
- $\phi_\text{ALC} = 0$ (held fixed)

**What it does:** adds a second, weaker drive *resonant* with the |1⟩↔|2⟩ transition. Its time-dependent amplitude is shaped like $dF/dt$. This creates destructive interference between:
- The leakage amplitude in |2⟩ caused by the (DRAG-modified) main drive's residual spectral weight near $f_{21}$
- The leakage amplitude in |2⟩ caused by the ALC drive itself

The two cancel at the gate end, leaving |2⟩ population ≈ 0.

### Combined effect

In the paper:
- DRAG alone reduces leakage by ~order of magnitude
- DRAG + ALC reduces leakage by an *additional* 10-20× compared to DRAG only

---

## 3. Complex-envelope generalization

The paper uses a **purely real** $F(t)$. For our optimization-based pipeline, the optimizer produces both $u_X(t)$ and $u_Y(t)$ as independent free controls. We generalize.

### Define the complex envelope

> $\Omega(t) \;\equiv\; u_X(t) + i\,u_Y(t)$

DRAG and ALC are both **linear operators** acting on $\Omega(t)$:

> $\Omega_\text{DRAG}(t) = \Omega(t) + i\,\frac{\alpha}{\eta}\,\frac{d\Omega}{dt}$
>
> $\Omega_\text{ALC env}(t) = i\,\frac{A_\text{ALC}}{\delta_\text{ALC}}\,\frac{d\Omega}{dt}$

(I have dropped factors of $2\pi$ since we use **angular frequencies** in rad/ns throughout. In our convention, $\eta = -2\pi\cdot 0.170 \approx -1.07$ rad/ns and $\delta_\text{ALC} = \eta$.)

### Decomposing into X, Y components

Using $d\Omega/dt = du_X/dt + i\,du_Y/dt$:

> $i\,\frac{\alpha}{\eta}\,\frac{d\Omega}{dt} = -\frac{\alpha}{\eta}\,\frac{du_Y}{dt} + i\,\frac{\alpha}{\eta}\,\frac{du_X}{dt}$

So the DRAG correction adds:
- $u_X^\text{DRAG} = u_X - (\alpha/\eta)\,du_Y/dt$
- $u_Y^\text{DRAG} = u_Y + (\alpha/\eta)\,du_X/dt$

Both quadratures get corrected. The single-quadrature Chiaro-Zhang formula is the special case $u_Y = 0$: the new X is unchanged, and only $+(\alpha/\eta)\,du_X/dt$ is added to Y.

### ALC envelope, same structure

> $\Omega_\text{ALC env} = i\,(A_\text{ALC}/\delta_\text{ALC})\,d\Omega/dt$
> $= -(A_\text{ALC}/\delta_\text{ALC})\,du_Y/dt + i\,(A_\text{ALC}/\delta_\text{ALC})\,du_X/dt$

Components:
- $\text{ALC}_X^\text{env}(t) = -(A_\text{ALC}/\delta_\text{ALC})\,du_Y/dt$
- $\text{ALC}_Y^\text{env}(t) = +(A_\text{ALC}/\delta_\text{ALC})\,du_X/dt$

### Lab-frame composite (in the rotating frame at $\omega_{01}$)

The ALC envelope sits on a fast carrier at frequency $\delta_\text{ALC}$. In the rotating frame at $\omega_{01}$, the carrier shows up explicitly:

> $u_X^\text{total}(t) = u_X^\text{DRAG}(t) + \text{ALC}_X^\text{env}(t)\,\cos(\delta_\text{ALC} t) - \text{ALC}_Y^\text{env}(t)\,\sin(\delta_\text{ALC} t)$
>
> $u_Y^\text{total}(t) = u_Y^\text{DRAG}(t) + \text{ALC}_X^\text{env}(t)\,\sin(\delta_\text{ALC} t) + \text{ALC}_Y^\text{env}(t)\,\cos(\delta_\text{ALC} t)$

This composite waveform is what we propagate through the 3-level Duffing Hamiltonian.

---

## 4. What's implemented in `hadamard_alc_comparison.ipynb`

### Section 1 — Optimize 2-level Hadamard pulses (default + robust)

Both pulses use the **same** `VariationalSplinePulseProblem` machinery:
- 2-level Hilbert space ($\sigma_x, \sigma_y$ drives, $\sigma_z$ as error operator for robustness)
- Cubic Hermite spline controls with 50 knots over T = 50 ns
- `Q_r = 0` for default (target only); `Q_r = 100` for robust
- Hard constraint $F \geq 0.9999$ via `FinalUnitaryFidelityConstraint`
- **Smoothness enforcement**: `R_ddu = 10` + `ddu_bound = 1.0` → triggers Piccolo's `:ddu` infrastructure, which ensures `du/dt` is a smooth waveform (not knot-rate noise)

### Section 2 — Honest verification with `unitary_rollout`

Re-propagates each saved trajectory using Piccolo's cubic-Hermite spline integrator at `abstol = reltol = 1e-12`. Reports honest F at gate end.

### Section 3 — 2-level ε-sweep with σ_z

For 51 ε values in ±10 MHz, builds a perturbed system $H_0 + \varepsilon\sigma_z$ and rolls the optimized pulse through it. Plots F vs ε on linear y, infidelity vs ε on log y.

### Section 4 — QuantumToolbox cross-check

Independent ODE-based propagation using `sesolve`, applied to each basis vector to build $U_T$ column-by-column. Compares against Piccolo's rollout — `max|ΔF|` should be < 1e-6.

### Section 5 — DRAG + ALC construction

The `build_composite(traj)` function:
1. Loads saved cubic Hermite spline (`u`, `du` from trajectory)
2. Reconstructs `sp_X(t)`, `sp_Y(t)` and computes `duXdt(t)`, `duYdt(t)` via central differences on the spline
3. Computes DRAG-corrected envelope:
   - `uX_drag = uX - (α/η)·duYdt`
   - `uY_drag = uY + (α/η)·duXdt`
4. Computes ALC envelope (slow part):
   - `alc_X_env = -(A_ALC/δ_ALC)·duYdt`
   - `alc_Y_env = +(A_ALC/δ_ALC)·duXdt`
5. Composites with carrier modulation at $\delta_\text{ALC}$ to give `uX_total`, `uY_total`

Returns a callable that emits all 6 pieces — raw / DRAG-only / DRAG+ALC — at any time t.

**Constants used (in our angular-frequency convention):**
- $\alpha = 1.0$ (`ALPHA_DRAG`)
- $\eta = -2\pi \cdot 0.170$ rad/ns ($\approx -1.07$ rad/ns, `η_anh`)
- $\delta_\text{ALC} = \eta$ (resonant with |1⟩↔|2⟩)
- `A_ALC_SCALE = 1.0` (full Chiaro-Zhang formula; tune up/down to match leakage suppression)

### Section 6 — Fig-1a-style plot

Two stacked panels (default, robust). Each shows three pairs of (u_X, u_Y):
- **Raw main** (solid + dotted)
- **+DRAG** (solid + dotted)
- **+DRAG +ALC** (solid + dotted, lighter)

Shows how the corrections layer onto the raw pulse.

### Section 7 — 3-level ε-sweep

For each (default, robust) × (raw, +DRAG+ALC), propagates the composite waveform in 3-lvl Duffing with $H_\text{drift} = (\eta/2)\hat n(\hat n - 1)$ and perturbation $V = \hat n$. Uses 1000 sub-step midpoint rule.

Final plot: 4 curves on infidelity panel + 4 curves on leakage panel.

---

## 5. Smoothness analysis — why R_ddu matters

### The chain

```
R_ddu + finite ddu_bound    →  Piccolo enforces :ddu = slope of :du
                            →  consecutive tangents :du[k] vary smoothly
                            →  du/dt (as a function of t) is a smooth waveform
                            →  DRAG correction (∝ du/dt) is smooth
                            →  ALC envelope (∝ du/dt) is smooth
                            →  composite waveform = sum of smooth functions = smooth
```

### Without R_ddu

The cubic Hermite spline lets the optimizer pick the tangents `:du[k]` **independently** at each knot. So `du[k]` and `du[k+1]` can be arbitrarily different. The resulting du(t) waveform is smooth WITHIN each inter-knot interval but has slope discontinuities AT each knot — meaning the SECOND derivative jumps at knots.

When you build DRAG or ALC from this jagged `du/dt`, the corrections have spikes at every knot. The lab-frame composite waveform looks like a spike train.

This is what we saw in the old modulate_iswap DRAG-warmstart experiments. Adding `R_ddu` was always the fix.

### With R_ddu

`Piccolo` adds `:ddu` as a trajectory variable, enforces `:ddu = (du[k+1] - du[k]) / Δt` via `DerivativeIntegrator`, regularizes it via `QuadraticRegularizer(ddu_sym, traj, R_ddu)`, and bounds it via `ddu_bound`. Net effect: consecutive tangents are close → `du/dt` is smooth → all derived corrections are smooth.

---

## 6. Caveats

### 6.1 ALC suppresses **final-time** leakage, not intermediate-time

The destructive interference in the |2⟩ amplitude happens at gate end. During the gate, |2⟩ population still builds up; it just cancels by t = T. So:
- ✅ Coherent gate fidelity at end is preserved
- ❌ T1 decay of transient |2⟩ during the gate is NOT suppressed
- ❌ Crosstalk via transient |2⟩ during the gate is NOT suppressed
- ❌ Mid-gate measurement of leakage would still show high |2⟩

For QEC contexts where transient |2⟩ also matters, ALC is insufficient — you'd want a time-integrated leakage cost (the J_L approach from Poggi & Kiely).

### 6.2 The complex DRAG formula assumes the optimizer's $u_X, u_Y$ are "well-behaved"

In Chiaro-Zhang, $F(t)$ is a clean raised cosine. In our pipeline, the optimizer chooses $u_X, u_Y$ to satisfy F + σ_z robustness in 2-lvl — the optimizer is free to use both quadratures however it wants. The DRAG/ALC formulas treat $\Omega(t) = u_X + i u_Y$ as the complex envelope of the main drive and correct based on its derivative. This is the natural generalization, but it's not guaranteed to give the same leakage suppression as the analytical formula would on a raised cosine.

If leakage suppression isn't as good as the paper reports, the likely cause is the optimized pulse having a *different* off-resonant Fourier spectrum than a raised cosine.

### 6.3 The 2-lvl optimization doesn't know about |2⟩

The optimizer never sees the 3-lvl Hamiltonian. It picks $u_X, u_Y$ that are optimal for 2-lvl F + σ_z-robustness only. There's no reason a priori for it to produce a pulse with a clean spectral notch at $f_{21}$. The analytical DRAG correction adds the notch; if the optimizer's main pulse has its own residual spectral content near $f_{21}$, the ALC drive may not fully cancel it.

A future refinement would be to **3-lvl-optimize the corrections themselves**: i.e., let the optimizer choose $A_\text{ALC}$, $\delta_\text{ALC}$, $\alpha$ (and possibly the main pulse shape) to minimize 3-lvl leakage directly. That's a different problem, more like the all-in-one variational approach we tried earlier.

### 6.4 `A_ALC_SCALE = 1.0` is the "natural" choice but may need tuning

The Chiaro-Zhang formula naturally produces ALC amplitudes of order $|du/dt| / |\delta_\text{ALC}|$ which, for typical optimized pulses at T ≈ 50 ns and $\eta = -170$ MHz, gives ALC amplitudes around 10–20% of the main drive — matching the paper's empirical calibration. If our notebook's `A_ALC_SCALE = 1.0` gives sub-optimal cancellation, scan it (or do a small grid search over A_ALC × δ_ALC, like the paper's Fig. 2c).

### 6.5 Optimization-time vs verification-time accuracy

- **Optimization** uses Piccolo's single-exp-per-knot BilinearIntegrator — fast but only first-order in time-averaging the cubic spline u(t).
- **Verification** (Section 2 and Section 7) uses either `unitary_rollout` with MagnusAdapt4 (much more accurate) or fine-grained midpoint rule with 1000 sub-steps (~0.05 ns/step at T = 50 ns).
- These can disagree on F by up to ~1e-4 in poorly-converged cases. The honest rollout is the truth.

---

## 7. Pipeline diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE A — 2-level optimization (no |2⟩ in the model)              │
│                                                                     │
│  inputs:  Hadamard target U_H, T = 50 ns, σ_z error op             │
│  knobs:   Q_r (0 for default, 100 for robust),                     │
│           F_THRESHOLD = 0.9999, R_ddu = 10, ddu_bound = 1.0        │
│  output:  optimized (u_X(t), u_Y(t)) at 50 cubic-Hermite knots     │
│                                                                     │
│  Result: smooth pulse satisfying F ≥ 0.9999 in 2-lvl Pauli.        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE B — analytical DRAG + ALC construction                       │
│                                                                     │
│  inputs:  saved (u_X, u_Y) trajectory                              │
│  formula: complex-envelope DRAG + ALC, both ∝ du/dt                │
│  outputs: u_X^total(t), u_Y^total(t) (lab-frame composite)         │
│                                                                     │
│  No optimization. Pure analytical correction.                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE C — 3-level Duffing verification                             │
│                                                                     │
│  inputs:  composite waveform, 3-lvl H_anh = (η/2) n̂(n̂-1)           │
│  method:  fine-grained midpoint propagation                        │
│  outputs: F(T), leakage(T), F vs ε, leakage vs ε                   │
│                                                                     │
│  Compares: raw main only  vs  main + DRAG + ALC.                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Summary

We implement the Chiaro-Zhang DRAG + ALC scheme as a **post-processing layer** on top of a 2-lvl-optimized robust pulse. Both DRAG (Eq. 2) and ALC (Eq. 4) are linear operators on the complex envelope $\Omega(t) = u_X + i u_Y$, with the form $i(\beta/\omega)\,d\Omega/dt$ for some prefactor $\beta$ and characteristic frequency $\omega$. Generalizing to both quadratures: both X and Y get corrected by $\pm (\beta/\omega)$ times the other channel's derivative.

Smoothness of `du/dt` (enforced by `R_ddu` + finite `ddu_bound` in the 2-lvl optimization) propagates cleanly through both corrections — no spike trains. The pipeline produces a smooth lab-frame waveform that should:
- Hit F ≥ 0.9999 in 2-lvl (from the optimization)
- Be robust to σ_z perturbations in 2-lvl (from `Q_r`)
- Suppress 3-lvl leakage at gate end via destructive interference (from DRAG + ALC)
- Preserve 3-lvl fidelity comparable to 2-lvl (since |2⟩ population at end ≈ 0)

The final plot in Section 7 of the notebook is the test: do the +DRAG+ALC infidelity curves sit visibly below the raw curves, and does the leakage drop by an order of magnitude or more? If so, the pipeline works as intended.
