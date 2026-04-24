# TimeDependentBilinearIntegrator

## What it does

The integrator enforces the discrete dynamics constraint

$$x_{k+1} = \Phi(x_k,\, u_k,\, \dot{u}_k,\, u_{k+1},\, \dot{u}_{k+1},\, t_k,\, \Delta t_k)$$

for every consecutive pair of knot points $(k, k+1)$. The constraint is that $x_{k+1}$ must equal the solution of the ODE

$$\dot{x} = G(u(t),\, t)\, x, \qquad x(t_k) = x_k$$

integrated forward by $\Delta t_k$. In the quantum control context $x = \tilde{U}$ (the isomorphic unitary) and $G(u, t)$ is the generator built from the Hamiltonian.

IPOPT treats this as a **residual equality constraint** at every time step:

$$r_k = x_{k+1} - \Phi(\ldots) = 0$$

and drives the residual to zero while simultaneously optimizing the controls.

---

## The ODE and normalized time

The integration is performed on the **normalized interval** $\tau \in [0, 1]$, obtained by the substitution $\tau = (t - t_k) / \Delta t_k$. The ODE becomes

$$\frac{dx}{d\tau} = G\!\bigl(u(\tau),\; t_k + \tau \Delta t_k\bigr)\, x \cdot \Delta t_k$$

The extra $\Delta t_k$ factor comes from the chain rule. Integrating from $\tau=0$ to $\tau=1$ advances the state by exactly one timestep regardless of $\Delta t_k$'s value.

The ODE is solved numerically with **Tsit5** (a standard 4th/5th-order Runge-Kutta pair).

---

## Control interpolation within a step

$u(\tau)$ must be evaluated at intermediate Tsit5 sub-steps inside $[0,1]$. The integrator supports four orders:

### Order 0 — zero-order hold
$$u(\tau) = u_k$$

The control is constant throughout the step. Only $u_k$ is needed.

### Order 1 — linear
$$u(\tau) = (1 - \tau)\, u_k + \tau\, u_{k+1}$$

Linear interpolation between adjacent knot values.

### Order 2 — quadratic Hermite
$$u(\tau) = a\tau^2 + \Delta t_k\, \dot{u}_k\, \tau + u_k, \qquad a = u_{k+1} - u_k - \Delta t_k\, \dot{u}_k$$

Matches value and slope at the left endpoint, and value at the right endpoint. Requires `traj[:du]`.

### Order 3 — cubic Hermite
$$u(\tau) = h_{00}(\tau)\,u_k + h_{10}(\tau)\,\Delta t_k\,\dot{u}_k + h_{01}(\tau)\,u_{k+1} + h_{11}(\tau)\,\Delta t_k\,\dot{u}_{k+1}$$

where $h_{00}, h_{10}, h_{01}, h_{11}$ are the standard Hermite basis polynomials (see `cubic_hermite_splines.md`). Matches values **and** slopes at **both** endpoints. This is the same polynomial used by `CubicSplinePulse`, so with `spline_order=3` the NLP dynamics are exactly consistent with the spline parameterization.

**For this problem:** `CubicSplinePulse` → `dynamics_spline_order=3` is set automatically. The NLP generator at sub-step $\tau$ is:

$$G(u(\tau), t_k + \tau\Delta t_k) = \text{iso}\!\left[\frac{\delta_{12}}{2} IZ + g(\tau)\bigl((XX{+}YY)\cos(\delta_{12}t) + (YX{-}XY)\sin(\delta_{12}t)\bigr) + \ldots\right]$$

where $g(\tau)$ is the cubic Hermite interpolation of the coupling amplitude.

---

## Parameter vector layout

At each step $k$, the relevant knot data is packed into a parameter vector $p_k$:

| Order | $p_k$ layout | Size |
|-------|-------------|------|
| 0 | $[u_k]$ | $n_u$ |
| 1 | $[u_k;\; u_{k+1}]$ | $2n_u$ |
| 2 | $[u_k;\; \dot{u}_k;\; u_{k+1}]$ | $3n_u$ |
| 3 | $[u_k;\; \dot{u}_k;\; u_{k+1};\; \dot{u}_{k+1}]$ | $4n_u$ |

The full ODE parameter is $[p_k;\; \Delta t_k;\; t_k]$.

---

## Jacobian and Hessian

IPOPT requires the Jacobian $\partial r_k / \partial z$ and the Lagrangian Hessian $\mu^T \partial^2 r_k / \partial z^2$ at every iteration, where $z = [z_k;\; z_{k+1}]$ is the stacked knot data vector.

Both are computed via **ForwardDiff** (forward-mode automatic differentiation) through the ODE solve. This is possible because Tsit5 is fully differentiable in Julia — ForwardDiff propagates dual-number arithmetic through every internal Runge-Kutta sub-step.

The Jacobian is **banded**: constraint $r_k$ only depends on $(z_k, z_{k+1})$, so each row of $\partial r / \partial z$ has at most $2 \cdot \text{traj.dim}$ nonzero columns. This sparsity is what makes the NLP tractable for large $N$.

---

## Residual definition

The residual returned to IPOPT at each step is:

$$r_k = x_{k+1} - \text{ODESolve}\!\left(G,\; x_k,\; p_k,\; \Delta t_k,\; t_k\right)$$

IPOPT enforces $r_k = 0$ as an equality constraint. The ODE solve is re-run at every IPOPT function evaluation (typically hundreds of times during optimization), so Tsit5 must be fast — which it is for the small system dimensions typical here ($4\times4$ matrices → $x$-dim = 32 for the isomorphic unitary).

---

## Connection to `VariationalSplinePulseProblem`

For the robust problem, two `TimeDependentBilinearIntegrator` instances are added:

1. **Nominal integrator** — propagates `:Ũ⃗` with $G_\text{nom}(u, t) = I_d \otimes G(u, t)$. This is what the infidelity objective reads from.

2. **Variational integrator** (`VariationalUnitaryIntegrator`) — propagates the augmented block `:var_Ũ⃗ = [Ũ⃗; ∂Ũ⃗_1; ...; ∂Ũ⃗_n]` with the block-triangular generator:

$$G_\text{var} = \begin{pmatrix} G(u,t) & 0 \\ G_{\text{var},i}(u,t) & G(u,t) \end{pmatrix}$$

so that $\partial\tilde{U}_i$ satisfies the variational equation automatically. The robustness objective penalizes $\|\partial\tilde{U}_i(T)\|^2$.

Both integrators use the same `spline_order=3` cubic Hermite interpolation, so the nominal and variational states experience exactly the same $u(t)$ trajectory during integration.

---

## Summary

| Quantity | Value in this problem |
|----------|----------------------|
| ODE solver | Tsit5 (adaptive RK4/5) |
| Integration interval | $\tau \in [0,1]$ per step |
| Control interpolation | cubic Hermite (`spline_order=3`) |
| Jacobian method | ForwardDiff through ODE solve |
| Sparsity pattern | banded (2 adjacent knots per constraint) |
| State variable | `:Ũ⃗` — isomorphic unitary, dim = $2d^2$ |
| Generator | `G(u,t)` from `H_nonlocal(u,t)` |