# Sin/Cos free-phase parameterization for `VariationalSplinePulseProblem`

**Status:** plan, pending approval. Implementation will happen on a feature branch off `modulate`.

## Motivation

The current `free_phase=true` path in `VariationalSplinePulseProblem` parameterizes the per-qubit virtual Z phases as raw scalar globals `:φ_1, :φ_2, …`, with box bounds (default `(-2π, 2π)`). Two observed failure modes:

- **Bounded `(-2π, 2π)`:** φ pins at the wall in the iSwap 2 MHz / 100 ns notebook. The box's bound multiplier dominates the φ-direction KKT condition, so the optimizer can't escape to a periodic copy of the true optimum.
- **Unbounded `(-Inf, Inf)`:** Ipopt log shows runaway oscillation past iter 600 — `inf_du` grows from ~50 to ~230, objective rises from 5.6 to 8.5, line search hits `alpha_pr = 0.0625` ("h 5") on every step. With `eval_hessian=false` the L-BFGS approximation can't capture curvature in the φ direction (which is effectively flat to first order in some pulse-decoupled neighborhoods), so Newton steps wander aimlessly along the periodic objective landscape.

**Root cause:** phase lives on a circle (S¹), but a scalar real variable is a chart on the real line with periodicity hidden. Box bounds create artificial walls; no bounds create artificial flat directions.

## Architecture (sin/cos on the unit circle)

Replace each free phase `:φ_i` with the pair `(cos θ_i, sin θ_i)` and enforce `c_i² + s_i² = 1` as a hard equality constraint. The optimizer then moves on S¹ — no wall, no spiral.

**Layout (matches Andy's `UnitaryFreePhaseProblem` in `QuantumCollocation.jl`):**

| Global symbol | Type | Length | Holds |
|---|---|---|---|
| `:cosθ` | `Vector{Float64}` | `n_qubits` | `[cos θ_1, cos θ_2, …]` |
| `:sinθ` | `Vector{Float64}` | `n_qubits` | `[sin θ_1, sin θ_2, …]` |

`(c_i, s_i)` is free per qubit — `2·n_qubits` independent decision variables total. The unit-norm constraint is vector-valued (one component per qubit):

```julia
phase_norm(z) = z[1:n].^2 .+ z[n+1:end].^2 .- 1
# returns:  [c_1² + s_1² − 1,  c_2² + s_2² − 1,  …]
```

Each qubit's pair lives on its own unit circle, independently.

The phase symbol prefix is configurable via `phase_name::Symbol = :θ`, so users can have `:cosθ_iswap`, `:sinθ_iswap` if they prefer.

## Concrete file changes (Piccolo.jl, `modulate` branch)

### 1. `src/control/templates/_problem_templates.jl`

Add a new helper alongside `setup_free_phase_globals!`:

```julia
function setup_sincos_phase_globals!(
    n_qubits::Int,
    phase_name::Symbol,
    global_data::Union{Nothing, Dict{Symbol, Vector{Float64}}};
    init_phases::Union{Nothing, Vector{Float64}} = nothing,
    verbose::Bool = false,
)
    x = Symbol("cos$(phase_name)")
    y = Symbol("sin$(phase_name)")
    init_phases = something(init_phases, zeros(n_qubits))
    @assert length(init_phases) == n_qubits

    if isnothing(global_data); global_data = Dict{Symbol, Vector{Float64}}(); end
    global_data[x] = cos.(init_phases)
    global_data[y] = sin.(init_phases)

    verbose && println("\tsincos free-phase globals: $(x), $(y)")
    return ([x, y], global_data)
end
```

No bounds are added — the unit-norm constraint handles boundedness.

### 2. `src/control/templates/spline_pulse_problem.jl`

Add `_make_sincos_phase_goal` alongside `_make_free_phase_goal`:

```julia
function _make_sincos_phase_goal(op::EmbeddedOperator)
    U_base   = unembed(op)
    subspace = op.subspace
    levels   = op.subsystem_levels
    n        = length(levels)

    function goal_fn(z)
        x, y = z[1:n], z[n+1:end]
        # Z(θ_i) = diag(1, c_i + i·s_i) per qubit; tensor over qubits.
        R = reduce(kron,
            [Diagonal([one(eltype(z)), xᵢ + im*yᵢ]) for (xᵢ, yᵢ) in zip(x, y)])
        return EmbeddedOperator(Matrix(R * U_base), subspace, levels)
    end
    return goal_fn
end
```

Type-generic for ForwardDiff compatibility (`one(eltype(z))` instead of `1.0`).

### 3. `src/control/templates/variational_spline_problem.jl`

Replace the `if free_phase` block. New version:

```julia
if free_phase
    @assert U_goal isa EmbeddedOperator "free_phase=true requires an EmbeddedOperator goal"
    n_qubits  = length(U_goal.subsystem_levels)
    U_goal_fn = _make_sincos_phase_goal(U_goal)
    phase_names, global_data = setup_sincos_phase_globals!(
        n_qubits, phase_name, global_data;
        init_phases = initial_phases,
        verbose     = piccolo_options.verbose,
    )
end
```

And in the constraint-assembly section, add the unit-norm constraint:

```julia
if free_phase
    function phase_norm(z)
        n = length(z) ÷ 2
        return z[1:n].^2 .+ z[n+1:end].^2 .- 1
    end
    push!(all_constraints, NonlinearGlobalConstraint(phase_norm, phase_names, traj))
end
```

The existing `UnitaryFreePhaseInfidelityObjective(U_goal_fn, state_sym, phase_names, traj; Q=Q)` call already accepts the new symbol list as-is — no change there.

**New kwarg:**
- `phase_name::Symbol = :θ` — prefix for the global symbol names (`:cosθ`, `:sinθ`).

**Removed kwargs (no longer used):**
- `global_bounds` for `:φ_*` — gone, replaced by the equality constraint. (If user-supplied bounds for *other* globals are still useful, keep the kwarg and just don't auto-populate it for the phase symbols.)

**Preserved kwargs:**
- `free_phase::Bool` — same meaning, on/off switch
- `initial_phases::Vector{Float64}` — same meaning, angles in radians; constructor expands to (cos, sin)

### 4. Notebook side (`robust_iswap_free_phase.ipynb`, future runs)

Setup cell becomes:

```julia
qcp = VariationalSplinePulseProblem(
    varsys, pulse, U_goal, N_knots;
    Q                     = 2.0,
    Q_r                   = Q_r,
    R                     = 1e-3,
    du_bound              = Inf,
    Δt_bounds             = (Δt, Δt),
    dynamics_spline_order = 3,
    n_path_samples        = 3,
    free_phase            = true,
    initial_phases        = [0.0, 0.0],
    phase_name            = :θ,
    piccolo_options       = PiccoloOptions(timesteps_all_equal = true, verbose = true),
)
```

Phase recovery after solve:

```julia
cs = get_trajectory(qcp).global_data[:cosθ]
sn = get_trajectory(qcp).global_data[:sinθ]
φ_1, φ_2 = atan.(sn, cs)
@printf("φ_1 = %.6f rad, φ_2 = %.6f rad\n", φ_1, φ_2)
```

`FinalUnitaryFidelityConstraint` would also take the new phase names:

```julia
push!(qcp.prob.constraints,
    FinalUnitaryFidelityConstraint(
        U_goal_fn, :Ũ⃗, [:cosθ, :sinθ], F_threshold, get_trajectory(qcp)
    )
)
```

where `U_goal_fn` is `_make_sincos_phase_goal(U_goal)` (exported, or accessed via the constructor's return).

### 5. Test (`@testitem` in `variational_spline_problem.jl`)

Mirror the existing `free_phase=true` test, asserting:
- `traj.global_components` contains `:cosθ` and `:sinθ`
- Each has length `n_qubits`
- After `solve!`, `cos² + sin² ≈ 1` for each qubit (within constraint tolerance)
- Optimization yields a sensible final fidelity

```julia
@testitem "VariationalSplinePulseProblem with sincos free_phase" begin
    # ... setup as in existing free_phase test ...
    qcp = VariationalSplinePulseProblem(
        varsys, pulse, U_goal, N;
        Q = 100.0, Q_r = 1.0, du_bound = 5.0,
        free_phase = true, initial_phases = [0.0, 0.0],
        phase_name = :θ,
    )
    traj = get_trajectory(qcp)
    @test haskey(traj.global_components, :cosθ)
    @test haskey(traj.global_components, :sinθ)
    @test length(traj.global_data[:cosθ]) == 2
    @test length(traj.global_data[:sinθ]) == 2

    solve!(qcp, max_iter = 50, print_level = 0)

    cs = traj.global_data[:cosθ]; sn = traj.global_data[:sinθ]
    for i in 1:2
        @test abs(cs[i]^2 + sn[i]^2 - 1) < 1e-6
    end
end
```

## Out-of-scope (not in this PR)

- Porting the same pattern to `SmoothPulseProblem` / `SplinePulseProblem`'s own `free_phase` path. (Same architecture would apply; do separately if needed.)
- Switching the goal-function convention to allow custom phase rotations (not just per-qubit Z). Andy's `UnitaryFreePhaseProblem` is more general — user supplies any function. For now we keep the convenience helper that builds the standard `diag(1, e^{iθ})` per qubit, since that's the use case in `robust_iswap_free_phase`.

## Open decisions

1. **Replace or co-exist?** Replace the angle-based `free_phase=true` path entirely (cleaner, but breaks any old notebook), or keep both behind a new kwarg like `phase_param ∈ {:angle, :sincos}` (backward compat, more code paths).
   - **Recommendation:** replace. The angle version is documented above as failing in both bounded and unbounded modes; no scenario where it's preferable.

2. **Symbol naming.** Andy uses `:cosθ`, `:sinθ`. We could use `:c_θ`, `:s_θ` or `:phase_cos`, `:phase_sin`. Cosmetic only — pick whatever reads best.
   - **Recommendation:** match Andy (`:cosθ`, `:sinθ`).

3. **Should `initial_phases` accept (cos, sin) pairs as well as angles?** Andy only accepts angles. Pairs would be a minor extension if you ever want to warm-start from a previous run's (c, s).
   - **Recommendation:** angles only for v1. Add pair support later if needed.

4. **Hessian.** Equality constraints add Lagrange-multiplier terms to the KKT system. With `eval_hessian=false`, Ipopt approximates the Lagrangian Hessian via L-BFGS — should still work (the equality constraint is quadratic, so its Hessian is constant). If conditioning issues recur, can revisit with `eval_hessian=true`.

## Implementation order (~1 hour total)

1. `setup_sincos_phase_globals!` in `_problem_templates.jl`  (~15 min)
2. `_make_sincos_phase_goal` in `spline_pulse_problem.jl`  (~10 min)
3. Edit `variational_spline_problem.jl`: swap helpers, add `NonlinearGlobalConstraint`, new kwarg  (~20 min)
4. Update `@testitem` and run the test suite  (~15 min)
5. Sanity-check by running `robust_iswap_free_phase.ipynb` with the new path on the laptop  (~10 min runtime + small notebook edits)

## After approval

I'll create branch `ak/sincos-free-phase` off `modulate`, do the edits, run tests, push to `private`. You review the diff (`git diff modulate..ak/sincos-free-phase`), merge into `modulate`, push to private modulate. SSH machine pulls and tries it on the iSwap notebook.
