# `VariationalSplinePulseProblem` — complexity scaling

## TL;DR

For 3-level Duffing optimization, the per-iteration cost is ~50–100× the 2-level cost.
The dominant factor is the **augmented `:var_Ũ⃗` state** (`= [Ũ⃗; ∂Ũ⃗₁; …; ∂Ũ⃗ₙ]`, dim = `iso_dim · (1 + n_vars)`), promoted to a per-knot NLP decision variable with a matching dense bilinear-integrator constraint. Most wall time is spent in `eval_constraint_jacobian` (Ipopt iterates) and the underlying matrix-exponential of `var_G`.

**Reformulating the variational portion as an indirect / shooting method (roll out
`∂Ũ⃗` forward + adjoint sensitivities) would collapse the per-iteration cost back near
the 2-level baseline.** This is a non-trivial Piccolo change.

## Decision variable count

Per knot, the trajectory `datavec` carries (omitting Δt, t for brevity):

| Symbol | Dim | Notes |
|---|---|---|
| `:Ũ⃗` | `iso_dim` | nominal unitary, iso-vec form (`= 2·d²` where d = Hilbert dim) |
| `:var_Ũ⃗` | `iso_dim · (1 + n_vars)` | augmented `[Ũ⃗; ∂Ũ⃗₁; …; ∂Ũ⃗ₙ]` (block 0 duplicates `:Ũ⃗`, kept in sync by integrator) |
| `:u`, `:du` | `n_drives` each | spline controls (and Hermite tangents) |
| `:t`, `:Δt` | 1 each | time stamp + step |

Decision variables per knot: `iso_dim · (2 + n_vars) + 2 · n_drives + 2`

For `N_knots` knots, total NLP variables ≈ `N_knots · iso_dim · (2 + n_vars)`.

**Concrete numbers:**

| Setup | `d` | `iso_dim` | `n_vars` | per-knot vars | total (`N=24`) |
|---|---|---|---|---|---|
| 2-level (4-dim) | 4 | 32 | 3 | ~170 | ~4 100 |
| 3-level (9-dim) | 9 | 162 | 3 | ~820 | ~19 700 |

**Ratio: ~5×** more decision variables in 3-level.

## Constraint Jacobian — where the time actually goes

Each pair of adjacent knots has a dense bilinear-integrator equality constraint for both `:Ũ⃗` and `:var_Ũ⃗`:

```
:Ũ⃗_{k+1}     = exp(−i·Δt · G(u_k, t_k))         · :Ũ⃗_k
:var_Ũ⃗_{k+1} = exp(−i·Δt · var_G(u_k, t_k))     · :var_Ũ⃗_k
```

Jacobian per transition (sparse but dense within its block):

- `∂ constraint / ∂ state_k` and `∂ constraint / ∂ state_{k+1}` — block size `S × S` where `S = iso_dim · (1 + n_vars)` for the variational block. Each evaluation requires the matrix exponential `exp(−i·Δt · var_G)` of an `S × S` matrix.
- `∂ constraint / ∂ u_k` — block size `S × n_drives`. Each entry needs `∂ exp / ∂ u`, typically via the Fréchet derivative of `exp` (one expensive primitive per drive).

Per-transition Jacobian cost is dominated by the matrix exponentials:

```
T_Jacobian_per_transition ≈ O(S³ · n_drives)
                          = matrix-exp + Fréchet derivatives of exp
```

### Sixth-power scaling in Hilbert dimension `D`

Unpacking `S` into the underlying physical quantities:

```
D       = total Hilbert-space dimension      ( d_per_qubit ^ n_qubits )
iso_dim = 2 · D²                              (real iso-vec form of a D×D unitary)
S       = iso_dim · (1 + n_vars)              (size of augmented variational state)
        = 2 · D² · (1 + n_vars)

Per-transition cost ≈ O(S³) = O( D⁶ · (1 + n_vars)³ )

Per Jacobian eval   ≈ O( N_knots · D⁶ · (1 + n_vars)³ · n_drives )
```

**The `D⁶` is the "sixth power scaling" your collaborators mentioned.** It comes from
matrix-exp (or its Fréchet derivative) on an `iso_dim × iso_dim` matrix being `O(iso_dim³)`,
and `iso_dim = 2D²`, so `iso_dim³ ∝ D⁶`. The `(1 + n_vars)³` is *on top of that* — it's
how much the variational augmentation costs you within a given `D`.

Equivalently, if you scale by *per-qubit* levels `d` for fixed number of qubits `Q`:

```
D = d^Q   →   cost ∝ d^(6Q)
```

So with 2 qubits, doubling the per-qubit levels makes the optimization `2^12 ≈ 4100×` slower in principle (`d^12`). Going from `d = 2` to `d = 3` (2-level to 3-level Duffing) at 2 qubits gives `(3/2)^12 ≈ 130×` — which matches the numbers below and your observed slowdown.

| Setup | `D` | `S = 2D²·(1+n_vars)` | `S³ ∝ D⁶·(1+n_vars)³` per transition | × 23 transitions |
|---|---|---|---|---|
| 2-level (2 qubits, d=2) | 4 | 128 | 2.1 M flops | 48 M flops / Jacobian |
| 3-level (2 qubits, d=3) | 9 | 648 | 272 M flops | 6.3 G flops / Jacobian |
| 3-qubit, d=3 (hypothetical) | 27 | ~5800 | ~2 × 10¹¹ flops | ~5 × 10¹² flops / Jacobian |

**Ratio for 2 → 3 levels (your case):** `(9/4)⁶ ≈ 130×`. Matches measured wall-time.

This matches what your flame chart shows: **`MOI.eval_constraint_jacobian` calls into `eval_jacobian(integrator, …)` calls into matrix-exp routines (and their Fréchet derivatives w.r.t. controls) that swallow the bulk of the time.**

## Per-iteration breakdown (Ipopt with `eval_hessian = false`)

| Phase | Cost | Where it shows up |
|---|---|---|
| Constraint evaluation | `N_knots × O(S³)` (matrix exp of var_G) | `MOI.eval_constraint` |
| **Constraint Jacobian** | `N_knots × O(S³ + S²·n_drives)` (Fréchet derivatives of exp) | `MOI.eval_constraint_jacobian` ← **dominant** |
| Objective gradient | `O(N_knots · iso_dim²)` (much smaller than Jacobian) | `MOI.eval_objective_gradient` |
| L-BFGS update | `O(20 × n_vars_NLP)` | inside Ipopt, cheap |
| Linear solve (MUMPS) | Sparse but with dense `S × S` blocks per transition. Memory and time both scale with `S²` per block; total ≈ `N_knots · S² · O(small)` | MUMPS factorization step |

For 3-level, `num_iter = 1000`, you should expect **roughly 1–5 hours of wall time** on a typical machine, vs **5–15 min for the 2-level version**.

## Memory

Per Jacobian sparse-matrix entry: ~12 bytes (value + row + col index). With ~15 M nonzeros for 3-level:

- Jacobian storage: ~180 MB
- L-BFGS Hessian approximation: ~3 MB (small, only ~20 vectors of length `n_NLP_vars`)
- MUMPS factorization workspace: highly problem-dependent; for dense `S × S = 648 × 648` blocks × 23 transitions, the LU factors can balloon to **several GB**

If you're memory-limited rather than CPU-limited, you'll see swap activity / OOM kills before wall-time becomes the issue.

## Why the variational state is so expensive

The whole point of `:var_Ũ⃗` being an NLP decision variable is that **Ipopt's quadratic-programming inner step can move it directly** (rather than re-rolling-out from the controls each iteration). This is the standard direct-collocation advantage: dynamics constraints become sparse equality constraints, and the QP step "knows" how the state will respond to control updates without integrating anything.

The cost: the augmented state has dim `(1 + n_vars)·iso_dim`, and dense Jacobian blocks of that size enter every Ipopt iteration.

The **indirect / shooting alternative**:
- Keep only `:Ũ⃗` as a decision variable (or even just `:u`, `:du`)
- At each Ipopt iteration, **roll out** `∂Ũ⃗(t)` forward from the variational ODE given the current controls
- Add the final-time variational penalty `Q_r · ‖∂Ũ⃗(T)‖²` directly to the objective
- Provide gradients via **adjoint sensitivities** (one backward ODE integration of dim `iso_dim`)

Trade-offs:

| | Direct (current) | Indirect (proposed) |
|---|---|---|
| NLP variable count | `iso_dim · (1 + n_vars)` per knot | `iso_dim` per knot (or zero, if you marginalize `Ũ⃗` too) |
| Jacobian block size | `iso_dim · (1 + n_vars)` × `iso_dim · (1 + n_vars)` | `iso_dim` × `iso_dim` |
| Per-iter cost ratio | 1× | ~`(1 + n_vars)⁻³ ≈ 1/64` for n_vars=3 |
| Convergence | QP step can move state directly | Each iter requires forward+backward ODE rollouts |
| Implementation in Piccolo | works today | requires new abstraction |

For the variational robustness objective specifically — which only depends on `∂Ũ⃗(T)` (final time) — the adjoint method is exact and fast. Each `eval_objective_gradient` becomes: (1) forward integrate `∂Ũ⃗(t)` from 0 to T using current u(t); (2) backward integrate the adjoint equation; (3) collect `dJ/du_k` from the adjoint. Total: 2× forward integrations of dim `iso_dim` per iteration, instead of one big NLP step on dim `iso_dim·(1+n_vars)·N_knots`.

This is essentially what **GRAPE** and **Krotov** methods do, and how `QuantumOptimalControl.jl` / `qiskit-dynamics` handle these problems.

## Practical paths from here, in order of effort

1. **Bandwidth-penalize the 2-level optimization** (no template change). Add or crank up `R_du`, `R_ddu` to suppress high-frequency content of `u(t)`. Run `leakage_estimate` (from the verify notebook) to confirm spectral weight at `|η|` drops. **Cost: re-run 2-level optimization, ~min.**

2. **Apply DRAG post-correction analytically** to the 2-level pulse. **Cost: ~10 min Julia script.** Often suffices.

3. **Reduce 3-level optimization wall time without rewriting:**
   - Lower `N_knots` (24 → 12 → 8). Roughly linear speedup, may hurt fidelity ceiling.
   - Enable Ipopt's exact Hessian (`eval_hessian = true`) — slower per iter, but typically halves iteration count for hard problems.
   - Compile with `--threads=8` and ensure MKL is used by Julia for the dense matrix-exp blocks.
   - **Cost: knobs only.**

4. **Custom variational integrator** that exploits the block-triangular structure of `var_G`. Currently `var_G` is built as one big `(1 + n_vars)·iso_dim` square; its block-triangular structure (lower-triangular) means `exp(var_G)` could be computed by `n_vars` separate `iso_dim`-dim matrix exps plus integral terms. **Cost: ~day of Piccolo work.**

5. **Adjoint / indirect formulation** of the variational portion. Major. **Cost: ~week of Piccolo work; could 50× the 3-level optimization.**

## Recommendation

Try (1) and (2) first. If they get you to acceptable 3-level performance, the indirect formulation isn't worth the engineering. If they don't, (4) is the high-ROI Piccolo change — exploit the block-triangular structure that's already there in `var_G`. (5) is the "right" long-term answer but a significant rewrite.

## Source references

- `Piccolo.jl/src/control/templates/variational_spline_problem.jl:264–298` — augmented `:var_Ũ⃗` state is added to the trajectory as a single block
- `Piccolo.jl/src/control/templates/variational_spline_problem.jl:332–340` — `VariationalUnitaryIntegrator` constructed for the augmented block
- `Piccolo.jl/src/control/templates/variational_spline_problem.jl:1–95` — `VariationalUnitaryIntegrator` definition; builds `var_G` via `Isomorphisms.var_G` and wraps in `BilinearIntegrator`
- `DirectTrajOpt.jl/src/solvers/ipopt_solver/evaluator.jl:354–470` — `MOI.eval_constraint_jacobian` central evaluation loop
- `DirectTrajOpt.jl/src/integrators/_integrators.jl` — per-integrator `eval_jacobian` (matrix-exp Fréchet derivatives)
