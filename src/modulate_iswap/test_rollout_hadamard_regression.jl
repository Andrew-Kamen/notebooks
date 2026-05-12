# =============================================================================
# test_rollout_hadamard_regression.jl
#
# Fast regression test for VariationalRolloutObjective + VariationalRolloutProblem
# correctness. Runs in ~20 seconds. Use this after any speed optimization to
# verify the gradient still matches finite differences and that the optimizer
# still converges to the Hadamard.
#
# Three checks:
#   1. cached_g_vars probe — constant variational directions should be detected
#   2. FD gradient — ForwardDiff gradient matches FiniteDiff to rtol < 1e-6
#   3. Optimization — 100-iter solve from random init converges to F > 0.99
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

# ---- Test 2: FD gradient
print("[2/3] FD gradient check...        ")
∇ = zeros(traj_fd.dim * traj_fd.N + traj_fd.global_dim)
gradient!(∇, rollout_obj, traj_fd)
Z⃗_vec = collect(vec(traj_fd))
∇_fd = FiniteDiff.finite_difference_gradient(Z⃗_vec) do Z⃗
    traj_data = Z⃗[1:(traj_fd.dim * traj_fd.N)]
    global_data = Z⃗[(traj_fd.dim * traj_fd.N + 1):end]
    return objective_value(rollout_obj,
        NamedTrajectory(traj_fd; datavec = traj_data, global_data = global_data))
end
rel_err = norm(∇ - ∇_fd) / norm(∇_fd)
@assert rel_err < 1e-6 "FD gradient mismatch: rel_err = $rel_err"
@printf("OK  (rel err = %.2e)\n", rel_err)

# ---- Test 3: end-to-end solve
print("[3/3] 100-iter Hadamard solve...  ")
Random.seed!(42)
T_solve = 2.0; N_solve = 15
times_s = collect(range(0.0, T_solve, length = N_solve))
pulse_s = CubicSplinePulse(0.1 * randn(2, N_solve), 0.1 * randn(2, N_solve), times_s)
qcp = VariationalRolloutProblem(varsys, pulse_s, U_goal, N_solve;
    Q = 200.0, Q_r = 1.0, R = 1e-3, du_bound = 5 * a_bound,
    piccolo_options = PiccoloOptions(verbose = false, timesteps_all_equal = true))
t0 = time()
solve!(qcp; max_iter = 100,
    options = IpoptOptions(eval_hessian = false, print_level = 0))
elapsed = time() - t0
traj = get_trajectory(qcp)
U_f = iso_vec_to_operator(traj[:Ũ⃗][:, end])
F = abs2(tr(U_goal' * U_f)) / 4
@assert F > 0.99 "F should be > 0.99 after 100 iters, got $F"
@printf("OK  (F = %.6f, %.1f s)\n", F, elapsed)

println()
println("All checks PASSED.")
