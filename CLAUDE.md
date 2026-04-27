# Robust Control (Piccolo / QuantumCollocation.jl)

## Project Overview
Single- and two-qubit robust quantum gate optimization using Piccolo (PiccoloQuantumObjects + QuantumCollocation.jl). Gates are optimized with robustness to Hamiltonian errors (e.g., Z-dephasing), then verified against hardware bandwidth constraints.

## Key Packages
- `PiccoloQuantumObjects` — gates (`GATES.H`, `GATES.X`, ...), Pauli matrices (`PAULIS.X/Y/Z`), quantum system constructors
- `QuantumCollocation` — optimization problem templates, rollout, variational systems
- `QuantumToolbox` — independent Schrödinger equation solver for cross-verification
- `CairoMakie` — plotting (Bloch spheres, fidelity curves, bar charts)
- `FFTW` — Gaussian hardware filter

## Problem Templates

### UnitarySmoothPulseProblem (Default / no robustness)
```julia
def = UnitarySmoothPulseProblem(sys, U_goal, T, Δt;
    Δt_max=Δt, Δt_min=Δt,
    a_bound=..., da_bound=..., dda_bound=..., Q_t=1.0)
```

### UnitaryToggleProblem (Toggling-frame robustness)
```julia
varsys = VariationalQuantumSystem(H_drive, [error_op])
add_prob = UnitaryToggleProblem(varsys, U_goal, T, Δt;
    a_bound=..., da_bound=..., dda_bound=...,
    Δt_min=Δt, Δt_max=Δt, Q=0.0, Q_t=Q)
```

### UnitaryVariationalProblem (Adjoint/variational robustness)
```julia
varsys = VariationalQuantumSystem(H_drive, [error_op])
varadd_prob = UnitaryVariationalProblem(varsys, U_goal, T, Δt;
    robust_times=[[T] for _ in 1:length(varsys.G_vars)],
    a_bound=..., da_bound=..., dda_bound=...,
    Q=0.0, Q_r=Q, Q_s=0.0)
```

## Important Default Parameter Values
- `a_bound = 1.0` (all templates)
- `da_bound = Inf` (all templates — no first-derivative constraint by default)
- `dda_bound = 1.0` (all templates)
- Always define bounds at the top of the notebook and pass them consistently to all three problem types

## Solving
```julia
push!(prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, prob.trajectory))
solve!(prob, max_iter=N, print_level=5, options=IpoptOptions(eval_hessian=false))
```
- `print_level=5` shows Ipopt iteration output
- `eval_hessian=false` for L-BFGS (faster), `true` for exact Hessian (slower, sometimes better convergence)

## Variational Problem: Rollout Initialization
The variational problem benefits from initializing with a unitary rollout:
```julia
init_traj = deepcopy(prob.trajectory)
init_traj.Ũ⃗ .= unitary_rollout(init_traj, varsys)
# Then recreate the problem with init_trajectory = remove_components(init_traj, [:Ũ⃗ᵥ1])
```

## Robustness Metrics

### Toggling objective (E_T)
Discrete toggle-frame integral: `Σ Δt_k U_k† H_err U_k`. Converges to variational objective as timestep → 0.

### Variational objective (E_V)
Adjoint-based: `|Tr((U†∂U)†(U†∂U))| / (TΔt)² / d`. This IS the continuous-time limit.

### Upsampled toggling
`tog_obj_upsample(...; factor=2^16)` — upsample controls and recompute toggle integral. Must use high factor (2^16+) to converge to variational. Factor 2^5 is NOT enough.

## Fidelity Verification

### Piccolo rollout (piecewise-constant, exact for ZOH)
```julia
unitary_rollout_fidelity(prob.trajectory, QuantumSystem(ε * PAULIS.Z, gen))
```
- Product of matrix exponentials — exact for piecewise-constant controls
- `gen` must match `H_drive` (e.g., `[PAULIS.X, PAULIS.Y]` for no-Z)

### Independent QuantumToolbox simulation (ODE-based)
- Uses `sesolve` with adaptive ODE solver — slightly different from Piccolo rollout
- Small discrepancy is expected (ODE solver doesn't step exactly at control boundaries)
- Good for cross-verification: agreement validates the optimization result

## Gaussian Hardware Filter (250 MHz)
Models AWG bandwidth limitation. Pipeline:
1. ZOH upsample coarse controls by factor M (e.g., 20)
2. FFT → multiply by Gaussian `G(f) = exp(-f²/(2σ_f²))` where `σ_f = B_3dB / √(ln2)` → IFFT
3. Simulate filtered controls with QuantumToolbox to verify gate survives

- `B_3dB = 250e6` Hz is the standard -3 dB bandwidth
- Filter X and Y together as complex baseband: `u_xy = u_x + i*u_y`
- Fine timestep: `dt_fine = dt_coarse / M`

## No-Z Control
When `H_drive = [PAULIS.X, PAULIS.Y]` (no Z control):
- Only 2 control channels in trajectory (`a` has 2 rows)
- `gen` for fidelity curves must be `[PAULIS.X, PAULIS.Y]`
- Filter only applies to X and Y channels

## Plotting Conventions
- Wong colors: `Makie.wong_colors()` — [1]=blue (Default), [3]=green (Toggling), [2]=orange (Variational)
- Bloch sphere: use `CairoMakie.mesh!(ax, Sphere(...); color=(0.2,0.6,1.0,0.05), transparency=true)` for sphere surface
- Bar charts for metrics: log scale, with upsampled stars overlaid
- LaTeX labels: `L"\mathcal{E}_T^{(0)}"`, `L"\mathcal{E}_V"`

## Notebook Structure Convention
Notebooks should follow this order:
1. Imports
2. Problem parameters (bounds defined once, used by all problem types)
3. Optimization (Default, Toggling, Variational)
4. Plot controls
5. Piccolo rollout fidelity vs ε
6. Gaussian filter + filtered control plots
7. Independent QT simulation (cross-verification)
8. Robustness metrics bar charts
9. Bloch sphere trajectories
