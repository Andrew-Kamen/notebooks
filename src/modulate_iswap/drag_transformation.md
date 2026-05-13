# The DRAG transformation for the modulated-iSWAP problem

## Context

The 2-level optimization (`robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_rollout_1kiter_seed42`) found a robust iSWAP at $F = 0.99990$ under bare 4-dim Pauli dynamics. Re-simulated in the 3-level Duffing model (9-dim), the same controls leak ~6% into $|2\rangle$ — because the microwave drives off-resonantly couple $|1\rangle\leftrightarrow|2\rangle$ on each qubit.

DRAG (Derivative Removal by Adiabatic Gate; Motzoi, Gambetta, Rebentrost, Wilhelm, PRL 103, 110501 (2009)) is the analytic recipe for shaping the controls so that the 3-level evolution, projected onto the computational subspace, matches the 2-level intended evolution to leading order in $1/|\eta|$.

This document specifies the DRAG transformation for **this specific problem** — 2 qubits, each with X and Y drives, anharmonicity $\eta = -2\pi \cdot 170$ MHz, drive bound $\Omega_\text{max} = 2\pi \cdot 10$ MHz.

## Physical system parameters

| Symbol | Value | Description |
|---|---|---|
| $\eta$ | $-2\pi \cdot 0.170$ rad/ns | Anharmonicity (transmon, $\eta < 0$) |
| $\Omega_\text{max}$ | $2\pi \cdot 0.01$ rad/ns | Microwave drive amplitude bound |
| $\Omega_\text{max}/|\eta|$ | $\approx 0.06$ | DRAG small parameter |
| $g_\text{eff}$ | $2\pi \cdot 0.002$ rad/ns | Coupling during flat-top |
| $T_\text{mw}$ | 130 ns | Microwave-active region |

## 3-level Duffing Hamiltonian for one qubit

In the rotating frame at the qubit's $|0\rangle\leftrightarrow|1\rangle$ frequency, with $b$ the annihilation operator on 3 levels:

$$H_1(t) = \frac{\eta}{2}\, \hat n(\hat n - I) + u_X(t)\,(b + b^\dagger) + u_Y(t)\, i(b^\dagger - b)$$

where $\hat n = b^\dagger b = \mathrm{diag}(0,1,2)$. In the qubit-basis ordering, $X = b + b^\dagger$ couples both $|0\rangle\leftrightarrow|1\rangle$ (amplitude 1) and $|1\rangle\leftrightarrow|2\rangle$ (amplitude $\sqrt{2}$). The latter is detuned by $\eta$ from resonance.

## The leakage and Stark-shift problem

Drive $u_X$ at the qubit frequency. To leading order in $1/|\eta|$, two unwanted effects appear in the qubit subspace:

1. **Off-resonant excitation of $|2\rangle$**: the state picks up a small $|2\rangle$ amplitude of order
   $$c_2(t) \approx \frac{\sqrt{2}\, u_X(t)}{\eta}$$

2. **AC Stark shift on $|1\rangle$**: the virtual $|1\rangle\to|2\rangle\to|1\rangle$ process shifts the $|1\rangle$ energy by $\sim u_X^2 / \eta$, distorting the gate.

Both arise from the same $1/\eta$ expansion. Adding a real Y-quadrature drive of magnitude $-\dot u_X / \eta$ cancels the leading effect at every instant: the rate of change of the $|2\rangle$-ghost amplitude is what feeds back into $|1\rangle$, and a derivative-shaped Y drive cancels that feedback. After cancellation, $|2\rangle$ population stays bounded by the residual $(\Omega/\eta)^4$, and the qubit-subspace evolution matches what a bare 2-level Pauli Hamiltonian would have produced.

## Symmetric DRAG for X+Y drives

When **both** quadratures are used as primary drives (this problem), the DRAG transformation acts symmetrically: each primary drive needs the derivative correction from the *other* quadrature added back.

For each qubit independently:

$$\boxed{\;u_X^\text{new}(t) = u_X(t) + \frac{\dot u_Y(t)}{\eta}, \qquad u_Y^\text{new}(t) = u_Y(t) - \frac{\dot u_X(t)}{\eta}\;}$$

Equivalently, with the complex envelope $\Omega = u_X + i u_Y$:

$$\Omega^\text{new}(t) = \Omega(t) + i\,\frac{\dot \Omega(t)}{\eta}$$

This is the "first-order full DRAG" form. The $u_X$ correction comes from the $u_Y$ derivative because a Y-direction primary drive has its own $|2\rangle$-ghost that re-radiates into $|1\rangle$ along the X axis.

## Application to this problem

The 2-qubit iSWAP optimization stores four microwave channels: $u_{X1}, u_{Y1}, u_{X2}, u_{Y2}$. DRAG acts per qubit, so we apply the same transformation independently to each qubit's $(u_X, u_Y)$ pair using the **same** $\eta$ (the qubits are identical transmons in this model):

$$
\begin{aligned}
u_{X1}^\text{new}(t) &= u_{X1}(t) + \dot u_{Y1}(t)/\eta \\
u_{Y1}^\text{new}(t) &= u_{Y1}(t) - \dot u_{X1}(t)/\eta \\
u_{X2}^\text{new}(t) &= u_{X2}(t) + \dot u_{Y2}(t)/\eta \\
u_{Y2}^\text{new}(t) &= u_{Y2}(t) - \dot u_{X2}(t)/\eta
\end{aligned}
$$

The coupling envelope $g_\text{eff}(t)$ is **not** modified by DRAG. DRAG addresses single-qubit physics (each transmon's own $|2\rangle$ state); it has no handle on the two-qubit leakage channel $|11\rangle\leftrightarrow|02\rangle, |20\rangle$ driven by the $g_\text{eff}(XX+YY)$ coupling.

## Sign conventions for this code

- $\eta = -2\pi \cdot 0.170$ rad/ns (negative for transmons).
- $\dot u/\eta$ is therefore *negative* times $\dot u$ for positive $\dot u$ — the formulas above are written so that with $\eta < 0$, the corrections have the right physical sign to cancel the parasitic terms.
- This matches Eq. (4) of Motzoi et al. and the standard convention in the Piccolo codebase. If you switch to a different rotating-frame convention (e.g., $e^{+i\omega t}$ instead of $e^{-i\omega t}$), the signs flip.

## What DRAG guarantees

For the 3-level Duffing unitary $U_\text{3lvl}(T)$ generated by the DRAG-corrected controls, and the 2-level reference unitary $U_\text{2lvl}(T)$ generated by the original $(u_X, u_Y)$ on bare Pauli operators:

$$P_\text{comp}\, U_\text{3lvl}(T)\, P_\text{comp} \;=\; U_\text{2lvl}(T) \;+\; \mathcal{O}\!\left(\frac{\Omega^2}{\eta^2}\right)$$

and the leakage out of the computational subspace scales as

$$\bigl\lVert (I - P_\text{comp})\, U_\text{3lvl}(T)\, P_\text{comp} \bigr\rVert^2 \;=\; \mathcal{O}\!\left(\frac{\Omega^4}{\eta^4}\right)$$

instead of the uncorrected $\mathcal{O}(\Omega^2/\eta^2)$.

## Predictions for this specific pulse

With $\Omega_\text{max}/|\eta| \approx 0.06$:

| Quantity | Uncorrected 2-lvl pulse | 2-lvl + DRAG | Source |
|---|---|---|---|
| Comp-subspace infidelity $1 - F$ | $\sim 6 \times 10^{-2}$ | $\sim 4 \times 10^{-3}$ | $\mathcal{O}(\Omega^2/\eta^2)$ vs $\mathcal{O}(\Omega^4/\eta^4)$ |
| Leakage at $T$ | $\sim 6\%$ (measured) | $\sim 10^{-3}$ | $(\Omega/\eta)^2 \to (\Omega/\eta)^4$ + coupling channel |
| Z-robustness | $\sigma_F(\varepsilon=1\text{ MHz}) \approx 5\%$ | Same to leading order | DRAG preserves qubit-subspace dynamics |

The coupling channel $|11\rangle \leftrightarrow |02\rangle, |20\rangle$ (driven by $g_\text{eff}(XX+YY)$, detuning $|\eta|$, matrix element $\sqrt{2}\,g$) contributes

$$P_\text{coup} \;\sim\; \left(\frac{g_\text{eff}\, T_\text{mw}}{\eta}\right)^2 \;\approx\; \left(\frac{2\,\mathrm{MHz} \cdot 130\,\mathrm{ns}}{170\,\mathrm{MHz}}\right)^2 \approx 1 \times 10^{-4}$$

at the final time. This sets the floor on what DRAG alone can achieve here.

## Limitations

1. **Leading-order only.** Residual infidelity scales as $(\Omega/\eta)^2 \approx 4 \times 10^{-3}$ for this problem. Higher-order DRAG (second-order Stark detuning, 5-parameter ansatz) can drop this further.
2. **Requires smooth derivatives.** If the controls saturate rails with sharp transitions, $\dot u$ is large and discontinuous; the leading-order Taylor expansion behind DRAG breaks down. Cubic-Hermite-spline controls (this project) are smooth at knots, which helps.
3. **Single-qubit only.** Two-qubit leakage (the coupling channel above) is invisible to per-qubit DRAG. To suppress it you need either smaller $g_\text{eff}/|\eta|$, longer gates, or a full 3-level optimization.
4. **Assumes correct $\eta$.** A miscalibrated $\eta$ degrades the cancellation quadratically: $\delta\eta/\eta = 10\%$ leaves $\sim 10\%$ of the leading-order $|2\rangle$ amplitude uncancelled, i.e. residual leakage scaled up by ~100×.

## Higher-order extensions

**Second-order DRAG (Stark detuning).** Adds a time-dependent detuning channel proportional to $|\Omega|^2/\eta$:

$$\delta_i(t) = \beta\, \frac{u_{Xi}^2(t) + u_{Yi}^2(t)}{\eta}, \qquad H \mathrel{+}= \delta_i(t) \cdot (\hat n_i - I/2)$$

with $\beta \approx 1$ a tunable scalar coefficient. This cancels the residual AC Stark shift left by leading-order DRAG, dropping comp-subspace infidelity from $\mathcal{O}((\Omega/\eta)^2)$ to $\mathcal{O}((\Omega/\eta)^4)$.

**5-parameter DRAG.** Replace the fixed coefficients $(1/\eta, 1/\eta, \beta/\eta)$ in the X-correction, Y-correction, and Stark term with three free scalars $(\alpha_1, \alpha_2, \beta)$ and minimize residual leakage by a 3-parameter sweep. No full optimization needed — these are scalars, not control trajectories.

For this problem the second-order Stark term is the natural next step **if** the leading-order DRAG simulation lands at $1 - F \approx 4 \times 10^{-3}$ as predicted. If the residual is already at the coupling-channel floor ($\sim 10^{-4}$), higher-order DRAG won't help and you'd need the 3-level optimizer.

## Implementation in this project

See `leakage_through_time.ipynb`, cells `4a13482f` (DRAG construction from saved 2-level trajectory) and `b700c665`–`a8b65171` (default pulse, F(t), ε-sweep, six-panel and combined-style plots). The transformation uses a central-difference derivative on the fine simulation grid:

```julia
uX_new = uX .+ central_diff(uY, dt) ./ η
uY_new = uY .- central_diff(uX, dt) ./ η
```

applied per qubit. The `propagate_with_leakage(P_drag)` call then evolves the corrected controls under the 9-dim Duffing Hamiltonian and records leakage at every $dt = 0.05$ ns step.

## References

1. F. Motzoi, J. M. Gambetta, P. Rebentrost, F. K. Wilhelm. *Simple Pulses for Elimination of Leakage in Weakly Nonlinear Qubits*. PRL **103**, 110501 (2009).
2. J. M. Gambetta, F. Motzoi, S. T. Merkel, F. K. Wilhelm. *Analytic control methods for high-fidelity unitary operations in a weakly nonlinear oscillator*. PRA **83**, 012308 (2011). (5-parameter DRAG and higher-order extensions.)
3. L. S. Theis, F. Motzoi, S. Machnes, F. K. Wilhelm. *Counteracting systems of diabaticities using DRAG controls: The status after 10 years*. EPL **123**, 60001 (2018). (Retrospective; covers two-qubit caveats.)
