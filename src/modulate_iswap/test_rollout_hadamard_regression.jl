# =============================================================================
# test_rollout_hadamard_regression.jl
#
# Fast regression test for VariationalRolloutObjective + VariationalRolloutProblem
# correctness. Runs in ~20 seconds. Use this after any speed optimization to
# verify (a) the *value* the rollout produces is still correct, (b) the
# gradient still matches finite differences, and (c) the optimizer still
# converges to the Hadamard.
#
# Checks:
#   1.  cached_g_vars probe — constant variational directions detected
#   2a. FD gradient (fast path, state_init_fixed=true) — u/du partials only
#   2b. FD gradient (full path, state_init_fixed=false) — entire trajectory
#   2c. **Value-level**: ∂Ũ⃗ᵢ(T) from the rollout objective matches an
#       independent FD reference computed via Piccolo's unitary_rollout on the
#       ε-perturbed quantum system. This validates that the optimizations
#       have not changed *what the rollout computes*, only *how fast*.
#   3.  Optimization — 100-iter solve from random init converges to F > 0.99
# =============================================================================

import Pkg
Pkg.activate(@__DIR__)
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
Pkg.develop([
    Pkg.PackageSpec(path = piccolo_path),
    Pkg.PackageSpec(path = directtrajopt_path),
])
Pkg.add(["FiniteDiff"])

using Piccolo
using LinearAlgebra
using FiniteDiff
using Random
using Printf

Random.seed!(42)

println("=" ^ 60)
println("VariationalRollout regression — Hadamard")
println("=" ^ 60)

H_drift = 0.0 * PAULIS.Z
H_drives = [PAULIS.X, PAULIS.Y]
H_vars   = [PAULIS.X, PAULIS.Y, PAULIS.Z]
a_bound = 2π * 0.5
varsys = VariationalQuantumSystem(H_drift, H_drives, H_vars, fill(a_bound, 2))

# Small problem for FD speed
T_gate = 3.0
N_knots = 6
times = collect(range(0.0, T_gate, length = N_knots))
pulse_fd = CubicSplinePulse(0.1 * randn(2, N_knots), 0.1 * randn(2, N_knots), times)
U_goal = GATES.H

# ---- Test 1: cached_g_vars detection
print("[1/3] cached_g_vars detection...  ")
qcp_fd = VariationalRolloutProblem(varsys, pulse_fd, U_goal, N_knots;
    Q = 0.0, Q_r = 1.0, R = 0.0, du_bound = 5 * a_bound,
    piccolo_options = PiccoloOptions(verbose = false, timesteps_all_equal = true))
traj_fd = get_trajectory(qcp_fd)

rollout_obj = nothing
for sub in qcp_fd.prob.objective.objectives
    if sub isa VariationalRolloutObjective
        global rollout_obj = sub; break
    end
end
@assert !isnothing(rollout_obj)
@assert !isnothing(rollout_obj.cached_g_vars) "cached_g_vars should be populated (X,Y,Z are constants)"
@assert length(rollout_obj.cached_g_vars) == 3
println("OK  (cached $(length(rollout_obj.cached_g_vars)) constant directions)")

# ---- Test 2a: FD gradient — fast path (Ũ⃗_init fixed, partials skipped)
#
# In this path the gradient at the initial-state slice is intentionally zero
# (the optimizer holds Ũ⃗_1 fixed via traj.initial → an equality constraint, so
# any gradient there is unused). We compare AD vs FD only at the components
# the optimizer can actually move: :u, :du.
print("[2a/3] FD gradient check (fast)...  ")
@assert rollout_obj.state_init_fixed "Template should auto-set state_init_fixed=true"

∇ = zeros(traj_fd.dim * traj_fd.N + traj_fd.global_dim)
gradient!(∇, rollout_obj, traj_fd)

# Build a mask over the trajectory datavec that selects only :u and :du knots.
# Layout per knot k: indices `(k-1)*traj.dim+1 .. k*traj.dim`, with each
# component occupying `traj.components[sym]` *within* that knot's slice.
mask_movable = falses(traj_fd.dim * traj_fd.N + traj_fd.global_dim)
for sym in (:u, :du)
    if haskey(traj_fd.components, sym)
        comp_range = traj_fd.components[sym]
        for k = 1:traj_fd.N
            base = (k - 1) * traj_fd.dim
            for j in comp_range
                mask_movable[base + j] = true
            end
        end
    end
end

Z⃗_vec = collect(vec(traj_fd))
∇_fd = FiniteDiff.finite_difference_gradient(Z⃗_vec) do Z⃗
    traj_data = Z⃗[1:(traj_fd.dim * traj_fd.N)]
    global_data = Z⃗[(traj_fd.dim * traj_fd.N + 1):end]
    return objective_value(rollout_obj,
        NamedTrajectory(traj_fd; datavec = traj_data, global_data = global_data))
end

# Pad ∇_fd to match ∇'s length (∇_fd is over datavec + globals; ∇ has the same shape).
@assert length(∇) == length(∇_fd) "gradient length mismatch ($(length(∇)) vs $(length(∇_fd)))"

rel_err_movable = norm(∇[mask_movable] - ∇_fd[mask_movable]) / max(norm(∇_fd[mask_movable]), eps())
@assert rel_err_movable < 1e-6 "FD/AD gradient mismatch on movable components: rel_err = $rel_err_movable"
@printf("OK  (rel err on u/du = %.2e)\n", rel_err_movable)

# ---- Test 2b: FD gradient — full path (Ũ⃗_init also differentiated)
#
# Force the slow/correct path explicitly. AD should now match FD on ALL
# trajectory components, including the :Ũ⃗ slice.
print("[2b/3] FD gradient check (full)...  ")
J_full = 0.0
let
    qcp_full = VariationalRolloutProblem(varsys, pulse_fd, U_goal, N_knots;
        Q = 0.0, Q_r = 1.0, R = 0.0, du_bound = 5 * a_bound,
        piccolo_options = PiccoloOptions(verbose = false, timesteps_all_equal = true),
        rollout_state_init_fixed = false,
    )
    traj_full = get_trajectory(qcp_full)
    rollout_obj_full = nothing
    for sub in qcp_full.prob.objective.objectives
        if sub isa VariationalRolloutObjective
            rollout_obj_full = sub; break
        end
    end
    @assert !isnothing(rollout_obj_full)
    @assert !rollout_obj_full.state_init_fixed "Explicit rollout_state_init_fixed=false should propagate"

    ∇_full = zeros(traj_full.dim * traj_full.N + traj_full.global_dim)
    gradient!(∇_full, rollout_obj_full, traj_full)
    Z⃗_full = collect(vec(traj_full))
    ∇_full_fd = FiniteDiff.finite_difference_gradient(Z⃗_full) do Z⃗
        return objective_value(rollout_obj_full,
            NamedTrajectory(traj_full;
                datavec = Z⃗[1:(traj_full.dim * traj_full.N)],
                global_data = Z⃗[(traj_full.dim * traj_full.N + 1):end]))
    end
    rel_err = norm(∇_full - ∇_full_fd) / norm(∇_full_fd)
    @assert rel_err < 1e-6 "FD/AD gradient mismatch on full trajectory: rel_err = $rel_err"
    @printf("OK  (rel err = %.2e)\n", rel_err)
end

# ---- Test 2c: VALUE-LEVEL — rollout's ∂Ũ⃗ᵢ(T) matches an FD reference
#
# The FD gradient tests above only verify AD is *internally consistent* with
# the rollout. If the rollout itself computes the wrong ∂Ũ⃗ᵢ(T) value, those
# tests would still pass. This test computes ∂Ũ⃗ᵢ(T) via an INDEPENDENT
# method — FD on Piccolo's `unitary_rollout` over the ε-perturbed quantum
# system — and compares to the rollout objective's internal value.
#
# Reference: U(T; H₀ + ε·Hᵥ) − U(T; H₀ − ε·Hᵥ) ≈ 2·ε · ∂U(T)/∂ε
print("[2c/4] rollout value vs FD ref... ")
let
    # Use the same pulse / trajectory as the gradient tests; cubic-Hermite
    # interpolation matches the spline_order=3 NLP dynamics.
    sys_nominal = QuantumSystem(H_drift, H_drives, fill(a_bound, 2))

    # Internal rollout state at the current trajectory's controls.
    # `objective_value` returns J = (Q_r/norm) Σᵢ scaleᵢ⁴ · ‖∂Ũ⃗ᵢ(T)/scaleᵢ‖²;
    # for scales=1 and Q_r=1 this is just (Σᵢ ‖∂Ũ⃗ᵢ(T)‖²) / norm_factor.
    J_internal = objective_value(rollout_obj, traj_fd)

    # Independent ∂Ũ⃗ᵢ(T) via central FD on unitary_rollout.
    δ = 1e-5
    ∂Ũ⃗_ref = Vector{Vector{Float64}}(undef, length(H_vars))
    for i in eachindex(H_vars)
        H_pos = H_drift + δ * H_vars[i]
        H_neg = H_drift - δ * H_vars[i]
        sys_pos = QuantumSystem(H_pos, H_drives, fill(a_bound, 2))
        sys_neg = QuantumSystem(H_neg, H_drives, fill(a_bound, 2))

        # Use cubic_hermite interpolation to match spline_order=3 NLP dynamics.
        Ũ⃗_pos = unitary_rollout(traj_fd, sys_pos;
            interpolation = :cubic_hermite, abstol = 1e-10, reltol = 1e-10)
        Ũ⃗_neg = unitary_rollout(traj_fd, sys_neg;
            interpolation = :cubic_hermite, abstol = 1e-10, reltol = 1e-10)
        ∂Ũ⃗_ref[i] = (Ũ⃗_pos[:, end] - Ũ⃗_neg[:, end]) ./ (2δ)
    end

    # Reconstruct J from the independent reference.
    norm_factor = rollout_obj.norm_factor
    J_ref = rollout_obj.Q_r * sum(
        rollout_obj.variational_scales[i]^2 * sum(abs2, ∂Ũ⃗_ref[i])
        for i in eachindex(∂Ũ⃗_ref)
    ) / norm_factor

    rel_err = abs(J_internal - J_ref) / max(abs(J_ref), eps())
    # Tolerance allows for: Tsit5 reltol=1e-5 on the rollout side, FD δ²·‖∂²U‖
    # truncation on the reference side. 1e-3 is comfortable headroom.
    @assert rel_err < 1e-3 "Rollout value mismatch vs FD reference: J_internal=$(J_internal), J_ref=$(J_ref), rel_err=$rel_err"
    @printf("OK  (J_internal=%.6e, J_ref=%.6e, rel_err=%.2e)\n",
            J_internal, J_ref, rel_err)
end

# ---- Test 3: end-to-end solve + integrator-honesty check
#
# Solve the Hadamard problem, then verify TWO things:
#   (a) the NLP says it achieved high fidelity, AND
#   (b) when the optimized controls are re-evolved with an INDEPENDENT
#       high-accuracy ODE solver (`unitary_rollout` with cubic_hermite
#       interpolation matching the NLP's spline_order=3 dynamics), the
#       resulting unitary AGREES with the NLP's `:Ũ⃗[:, end]`.
#
# If (a) passes and (b) fails, the bilinear integrator is biased and the NLP
# is reporting fictitious fidelity — the optimizer "solved" a different
# problem than the real-world one. This is the check the user explicitly
# asked for after seeing speedups.
print("[3/4] 100-iter Hadamard solve...  ")
Random.seed!(42)
T_solve = 2.0; N_solve = 15
times_s = collect(range(0.0, T_solve, length = N_solve))
pulse_s = CubicSplinePulse(0.1 * randn(2, N_solve), 0.1 * randn(2, N_solve), times_s)
qcp = VariationalRolloutProblem(varsys, pulse_s, U_goal, N_solve;
    Q = 200.0, Q_r = 1.0, R = 1e-3, du_bound = 5 * a_bound,
    piccolo_options = PiccoloOptions(verbose = false, timesteps_all_equal = true))
t0 = time()
# constr_viol_tol = 1e-8 (vs Ipopt default 1e-4) closes the NLP-vs-reality
# fidelity gap from ~1e-3 to ~1e-6 at no extra wall cost — see Test 4 below
# and `optimizations.md` for the analysis.
solve!(qcp; max_iter = 100,
    options = IpoptOptions(
        eval_hessian = false, print_level = 0,
        constr_viol_tol = 1e-8, tol = 1e-8, acceptable_tol = 1e-8,
    ))
elapsed = time() - t0
traj = get_trajectory(qcp)
U_f = iso_vec_to_operator(traj[:Ũ⃗][:, end])
F = abs2(tr(U_goal' * U_f)) / 4
@assert F > 0.99 "F should be > 0.99 after 100 iters, got $F"
@printf("OK  (F_nlp = %.6f, %.1f s)\n", F, elapsed)

print("[4/4] integrator honesty check... ")
let
    # Re-evolve the optimized controls with high-accuracy independent ODE.
    sys_nominal = QuantumSystem(H_drift, H_drives, fill(a_bound, 2))
    Ũ⃗_ref = unitary_rollout(traj, sys_nominal;
        interpolation = :cubic_hermite, abstol = 1e-12, reltol = 1e-12)
    U_ref = iso_vec_to_operator(Ũ⃗_ref[:, end])
    U_nlp = iso_vec_to_operator(traj[:Ũ⃗][:, end])

    # Both should describe the same physical evolution. Compare directly.
    op_err = opnorm(U_nlp - U_ref)
    F_ref = abs2(tr(U_goal' * U_ref)) / 4
    F_gap = abs(F - F_ref)

    # With Ipopt's `constr_viol_tol = 1e-8` (set on the solve! call above),
    # F_nlp should agree with the independent rollout's F to ~1e-6.
    # Operator-norm gap floors at ~1e-3 due to a tiny global phase mismatch
    # (gauge-invariant: F = |tr(U_goal'·U)|²/d is invariant under U → U·e^{iφ}).
    @assert F_gap < 1e-4 "NLP fidelity fictitious: F_nlp=$F, F_ref=$F_ref, ΔF=$F_gap"
    @printf("OK  (F_ref=%.6f, |ΔF|=%.2e, ‖ΔU‖=%.2e — last is harmless global phase)\n",
            F_ref, F_gap, op_err)
end

println()
println("All checks PASSED.")
