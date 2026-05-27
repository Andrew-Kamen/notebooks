# Interaction Hamiltonian for leakage and the first-order susceptibility

## 1. Setup

3-level transmon, rotating frame at $\omega_{10}$. Anharmonicity $\eta = \omega_{21} - \omega_{10}$ (transmon: $\eta < 0$). Complex drive envelope $\Omega(t) = u_X(t) + i \, u_Y(t)$.

In the $\{|0\rangle, |1\rangle, |2\rangle\}$ basis, the Hamiltonian splits as $H(t) = H_\mathrm{nom}(t) + V_\mathrm{leak}(t)$ with

$$
H_\mathrm{nom}(t) \;=\; \begin{pmatrix} 0 & \Omega^* & 0 \\ \Omega & 0 & 0 \\ 0 & 0 & \eta \end{pmatrix},
\qquad
V_\mathrm{leak}(t) \;=\; \sqrt{2}\,\Omega(t)\,|2\rangle\langle 1| + \sqrt{2}\,\Omega^*(t)\,|1\rangle\langle 2|.
$$

$H_\mathrm{nom}$ is block-diagonal: qubit-subspace drive in $\{|0\rangle, |1\rangle\}$ + bare $|2\rangle$ energy. $V_\mathrm{leak}$ is the off-diagonal block.

The nominal propagator $U_0(t)$ solves $i\,\partial_t U_0 = H_\mathrm{nom}\,U_0$ and is block-diagonal:

$$
U_0(t) \;=\; U_q(t) \;\oplus\; e^{-i\eta t}
$$

where $U_q(t)$ is the 2×2 qubit-subspace propagator.

## 2. Interaction Hamiltonian for leakage

Go to the interaction picture with respect to $H_\mathrm{nom}$: treat $H_\mathrm{nom}$ as the "free" Hamiltonian and $V_\mathrm{leak}$ as the perturbation. Schrödinger-picture states transform as $|\psi_I(t)\rangle = U_0^\dagger(t)\,|\psi_S(t)\rangle$, and the interaction-picture Hamiltonian is

$$
H_\mathrm{int}(t) \;\equiv\; U_0^\dagger(t)\,V_\mathrm{leak}(t)\,U_0(t).
$$

Note this is **not** the standard interaction picture with respect to a static $H_0$. Here the "free" part $H_\mathrm{nom}(t)$ is itself time-dependent (the qubit-subspace drive lives in $H_\mathrm{nom}$). The transformation is the **toggling-frame** generalization: rotate out *everything* that doesn't connect to $|2\rangle$, leaving only the leakage coupling to first order. This is the right move because we want to treat the gate dynamics exactly and the leakage perturbatively.

Using $U_0^\dagger|2\rangle = e^{+i\eta t}|2\rangle$ and $U_0^\dagger$ acting as $U_q^\dagger$ on the qubit subspace:

$$
\boxed{\; H_\mathrm{int}(t) \;=\; \sqrt{2}\,\Omega(t)\,e^{+i\eta t}\,|2\rangle\langle 1|\,U_q(t) \;+\; \mathrm{h.c.} \;}
$$

Three factors at the $|1\rangle \to |2\rangle$ matrix element: drive envelope $\Omega(t)$, qubit-subspace propagator $\langle 1|U_q(t)$, detuning phase $e^{+i\eta t}$.

## 3. First-order susceptibility

The interaction-picture state $|\psi_I(t)\rangle$ evolves under $H_\mathrm{int}$:

$$
i\,\partial_t|\psi_I(t)\rangle \;=\; H_\mathrm{int}(t)\,|\psi_I(t)\rangle, \qquad |\psi_I(0)\rangle = |\psi_\mathrm{init}\rangle.
$$

First-order Dyson:

$$
|\psi_I(T)\rangle \;\approx\; |\psi_\mathrm{init}\rangle \;-\; i\int_0^T H_\mathrm{int}(t')\,|\psi_\mathrm{init}\rangle\,dt'.
$$

Project onto $|2\rangle$. The leakage amplitude (interaction-picture) is

$$
\hat{c}_2(T) \;=\; -i\int_0^T \langle 2|H_\mathrm{int}(t')|\psi_\mathrm{init}\rangle\,dt' \;=\; -i\sqrt{2}\int_0^T \Omega(t')\,c_1(t')\,e^{+i\eta t'}\,dt'
$$

where $c_1(t) \equiv \langle 1|U_q(t)|\psi_\mathrm{init}\rangle$ is the qubit-subspace $|1\rangle$ amplitude.

Identify the Fourier transform $\hat{f}(\omega) = \int f(t)\,e^{-i\omega t}\,dt$:

$$
\boxed{\; \hat{c}_2(T) \;=\; -i\sqrt{2}\,\widehat{\Omega \cdot c_1}\,(-\eta) \;}
$$

This is the first-order leakage susceptibility — the Fourier coefficient of the *product* of the drive envelope and the qubit-subspace amplitude, evaluated at $\omega = -\eta$.

The leakage probability is

$$
P_\mathrm{leak} \;=\; |\hat{c}_2(T)|^2 \;=\; 2\,|\widehat{\Omega \cdot c_1}(-\eta)|^2.
$$
