# Physics of the modulated-coupling iSWAP problem

## System

Two coupled qubits with a time-dependent exchange interaction. The coupling amplitude $g(t)$ and four single-qubit drives are modulated to implement a two-qubit gate.

---

## Hamiltonian

$$H(u, t) = \frac{\delta_{12}}{2} IZ + g(t)\Big[(XX + YY)\cos(\delta_{12} t) + (YX - XY)\sin(\delta_{12} t)\Big] + u_{X1}(t)\, XI + u_{Y1}(t)\, YI + u_{X2}(t)\, IX + u_{Y2}(t)\, IY$$

### Control vector

$$u(t) = \bigl(g(t),\; u_{X1}(t),\; u_{Y1}(t),\; u_{X2}(t),\; u_{Y2}(t)\bigr)$$

### Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| $\delta_{12}$ | $2\pi \times 0.05$ rad/ns | Qubit–qubit detuning |
| $a_\text{bound}$ | $2\pi \times 0.01$ rad/ns | Drive amplitude bound (10 MHz) |
| $T_f$ | 100 ns | Gate duration |

### Amplitude bounds

$$|u_i(t)| \leq a_\text{bound} = 2\pi \times 10 \text{ MHz}, \quad i = 1,\ldots,5$$

---

## Physical interpretation

**Drift term** $\frac{\delta_{12}}{2} IZ$: static $Z$-rotation on qubit 2 from the detuning. This causes the qubits to accumulate relative phase at rate $\delta_{12}/2$.

**Coupling terms**: $g(t)$ is the envelope of an exchange-type interaction. In the lab frame the coupling is

$$g(t)\,\bigl[(XX+YY)\cos(\delta_{12} t) + (YX-XY)\sin(\delta_{12} t)\bigr]$$

This is the rotating-wave form of an $XY$-exchange interaction modulated at frequency $\delta_{12}$, which drives $|01\rangle \leftrightarrow |10\rangle$ transitions. The cosine quadrature carries $XX+YY$ (symmetric exchange) and the sine quadrature carries $YX-XY$ (antisymmetric).

**Single-qubit drives**: $u_{X1}, u_{Y1}$ drive qubit 1 along $X$ and $Y$; $u_{X2}, u_{Y2}$ drive qubit 2. These provide local $SU(2)$ control on each qubit.

---

## Target gate

$$U_\text{goal} = e^{-i\frac{\pi}{4}(XX + YY)} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 0 & i \\ 0 & 1 & i & 0 \\ 0 & i & 1 & 0 \\ i & 0 & 0 & 1 \end{pmatrix}$$

This is the **iSWAP gate**, which swaps the $|01\rangle$ and $|10\rangle$ states with a phase of $i$.

---

## Fidelity

The gate fidelity between the propagated unitary $U(T_f)$ and the target is

$$\mathcal{F}(U) = \frac{1}{d^2}\left|\mathrm{tr}\!\left(U_\text{goal}^\dagger\, U(T_f)\right)\right|^2, \qquad d = 4$$

The NLP minimizes $1 - \mathcal{F}$ subject to the dynamics, control bounds, and (for the robust problem) the variational penalty.

---

## Error model and robust optimization

The variational (robust) problem adds three static coherent error channels:

$$E_1 = ZI, \qquad E_2 = IZ, \qquad E_3 = ZZ$$

These represent single-qubit dephasing on qubit 1, qubit 2, and correlated $ZZ$ dephasing respectively. Under a perturbation $H \to H + \varepsilon_i E_i$, the propagator shifts to first order as

$$U(T_f;\,\varepsilon_i) \approx U(T_f) + \varepsilon_i \,\partial_{\varepsilon_i} U \Big|_{\varepsilon=0}$$

The variational state $\partial\tilde{U}_i$ satisfies

$$\frac{d}{dt}\partial\tilde{U}_i = G(u,t)\,\partial\tilde{U}_i + G_{\text{var},i}(u,t)\,\tilde{U}$$

where $G_{\text{var},i}$ is the isomorphic generator of $E_i$.

### Robust objective

$$\mathcal{L} = (1 - \mathcal{F}) + \lambda_r \sum_{i=1}^{3} \left\|\partial\tilde{U}_i(T_f)\right\|^2$$

Minimizing $\|\partial\tilde{U}_i(T_f)\|^2$ suppresses the first-order sensitivity of the gate to each error channel, making the solution robust to small fluctuations in $ZI$, $IZ$, and $ZZ$.

In the notebook: `Q = 0.0` (pure robustness, no nominal fidelity weight in objective), `Q_r = 100.0` (robustness weight), with a hard fidelity constraint $\mathcal{F} \geq 0.9999$ enforced via `FinalUnitaryFidelityConstraint`.

---

## Spline parameterization

The controls are parameterized as a cubic Hermite spline with $N$ knot points:

$$u_i(t) = \sum_k \bigl[h_{00}(\tau_k)\,u_{i,k} + h_{10}(\tau_k)\,\Delta t_k\,\dot{u}_{i,k} + h_{01}(\tau_k)\,u_{i,k+1} + h_{11}(\tau_k)\,\Delta t_k\,\dot{u}_{i,k+1}\bigr]$$

Both knot values $u_{i,k}$ and tangents $\dot{u}_{i,k}$ are free NLP variables, giving the optimizer fine-grained control over the pulse shape. See `cubic_hermite_splines.md` for the basis polynomial definitions.