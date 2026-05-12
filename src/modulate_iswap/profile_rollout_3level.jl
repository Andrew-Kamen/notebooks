# =============================================================================
# profile_rollout_3level.jl
#
# Profile VariationalRolloutObjective on the 3-level Duffing iSWAP setup —
# scaled-down (N_knots=6) so it finishes in seconds. Goal: find where time goes
# inside gradient!, then run a single Ipopt iter to see how much of total
# per-iter cost is the rollout objective vs the rest of the NLP machinery.
#
# Output: stderr breakdown of timings + PProf flame graph (profile.pb.gz).
# View flame graph with:   pprof -http=:1234 profile.pb.gz
# =============================================================================

import Pkg
Pkg.activate(@__DIR__)
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
Pkg.develop([
    Pkg.PackageSpec(path = piccolo_path),
    Pkg.PackageSpec(path = directtrajopt_path),
])
Pkg.instantiate()

using Piccolo
using LinearAlgebra
using Random
using Printf
using Profile
using PProf

# =============================================================================
# 3-level Duffing operators (single mode) — same as the production script
# =============================================================================
const n_lvl = 3
b = zeros(ComplexF64, n_lvl, n_lvl)
for j in 1:n_lvl-1
    b[j, j+1] = sqrt(j)
end
const b_3  = copy(b)
const bd_3 = b_3'
const I3   = Matrix{ComplexF64}(I, n_lvl, n_lvl)

const X3 = b_3 + bd_3
const Y3 = im * (bd_3 - b_3)
const n3 = bd_3 * b_3

const I9 = Matrix{ComplexF64}(I, n_lvl^2, n_lvl^2)
const XI = kron(X3, I3); const YI = kron(Y3, I3)
const IX = kron(I3, X3); const IY = kron(I3, Y3)
const XX = kron(X3, X3); const YY = kron(Y3, Y3)
const nI = kron(n3, I3); const In = kron(I3, n3); const nn = nI * In

const g_eff = 2π * 0.002
const Δ_mw1 = 0.0; const Δ_mw2 = 0.0
const a_bound = 2π * 0.01
const drive_bounds = fill(a_bound, 4)
const η_anh = -2π * 0.170
const H_anh = (η_anh / 2) * (nI * (nI - I9) + In * (In - I9))

# =============================================================================
# Tiny problem: N_knots = 6 (vs 8 in production), T_total_ns = 130 ns
# =============================================================================
const F_threshold     = 0.9999
const Q_r             = 1e2
const T_total_ns      = 130.0
const N_knots         = 6
const SEED            = 42

const subspace_indices = [1, 2, 4, 5]
const U_goal_4x4 = let
    σx = ComplexF64[0.0 1.0; 1.0 0.0]
    σy = ComplexF64[0.0 -im; im 0.0]
    exp(-im * π/4 * (kron(σx, σx) + kron(σy, σy)))
end
const U_goal = EmbeddedOperator(U_goal_4x4, subspace_indices, [n_lvl, n_lvl])

function H_gate_frame(u, t)
    uX1, uY1, uX2, uY2 = u
    H = g_eff * (XX + YY) + H_anh
    c1 = cos(Δ_mw1 * t); s1 = sin(Δ_mw1 * t)
    H += uX1 * (XI * c1 + YI * s1)
    H += uY1 * (YI * c1 - XI * s1)
    c2 = cos(Δ_mw2 * t); s2 = sin(Δ_mw2 * t)
    H += uX2 * (IX * c2 + IY * s2)
    H += uY2 * (IY * c2 - IX * s2)
    return H
end
const H_vars = [(u, t) -> nI, (u, t) -> In, (u, t) -> nn]

const varsys = VariationalQuantumSystem(
    H_gate_frame, H_vars, 4, drive_bounds;
    time_dependent = true,
)

T_f = T_total_ns
Δt  = T_f / N_knots
Random.seed!(SEED)
controls = 2 .* a_bound .* rand(4, 300) .- a_bound
times    = collect(LinRange(0.0, T_f, 300))
du_init  = zeros(4, 300)
pulse    = CubicSplinePulse(controls, du_init, times)

println("Building VariationalRolloutProblem (N_knots=$N_knots, levels=$(varsys.levels))...")
qcp = VariationalRolloutProblem(
    varsys, pulse, U_goal, N_knots;
    Q = 0.0, Q_r = Q_r, R = 1e-3, du_bound = Inf,
    Δt_bounds = (Δt, Δt),
    dynamics_spline_order = 3, n_path_samples = 3,
    piccolo_options = PiccoloOptions(
        timesteps_all_equal = true, verbose = false,
        leakage_constraint = true,
        leakage_constraint_value  = 1e-3,
        leakage_cost = 1.0,
    ),
)
push!(qcp.prob.constraints,
    FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F_threshold, get_trajectory(qcp))
)

# Locate the rollout objective and report cache status
rollout_obj = nothing
for sub in qcp.prob.objective.objectives
    if sub isa VariationalRolloutObjective
        global rollout_obj = sub; break
    end
end
@assert !isnothing(rollout_obj) "VariationalRolloutObjective not found in objective list"
println("Cache status:")
println("  cached_g_vars   : ", isnothing(rollout_obj.cached_g_vars) ? "FALLBACK" : "ACTIVE")
println("  cached_g_linear : ", isnothing(rollout_obj.cached_g_linear) ? "FALLBACK" : "ACTIVE")

traj = get_trajectory(qcp)
println("\nTrajectory geometry:")
println("  N (knots)           : $(traj.N)")
println("  iso_dim (:Ũ⃗ dim)    : $(traj.dims[:Ũ⃗])")
println("  aug_dim (rollout)   : $(rollout_obj.aug_dim)")
println("  total NLP variables : $(traj.dim * traj.N + traj.global_dim)")

# =============================================================================
# Warm-up (force compilation) then time
# =============================================================================
println("\n=== Warm-up ===")
@time J0 = objective_value(rollout_obj, traj)
@printf("J0 = %.6e\n", J0)
∇ = zeros(traj.dim * traj.N + traj.global_dim)
@time gradient!(∇, rollout_obj, traj)
@printf("‖∇‖ = %.6e\n", norm(∇))

println("\n=== Timing (3 reps each, after warm-up) ===")

println("objective_value:")
for r in 1:3
    @time objective_value(rollout_obj, traj)
end

println("\ngradient!:")
for r in 1:3
    fill!(∇, 0.0)
    @time gradient!(∇, rollout_obj, traj)
end

# =============================================================================
# PProf the gradient! call
# =============================================================================
println("\n=== Profile (gradient!, 10 reps) ===")
Profile.clear()
@profile begin
    for _ in 1:10
        fill!(∇, 0.0)
        gradient!(∇, rollout_obj, traj)
    end
end
pprof_path = joinpath(@__DIR__, "profile_rollout_gradient.pb.gz")
PProf.pprof(out = pprof_path, web = false)
println("PProf written to: $pprof_path")
println("View with: pprof -http=:1234 $pprof_path")

# =============================================================================
# One full Ipopt iter (using max_iter=1) for context on rollout-vs-NLP cost split
# =============================================================================
println("\n=== solve!(max_iter=1) — full Ipopt iter wall time ===")
@time solve!(qcp; max_iter = 1, print_level = 0,
    options = IpoptOptions(eval_hessian = false))
println("\n=== solve!(max_iter=3) — for per-iter trend ===")
@time solve!(qcp; max_iter = 3, print_level = 0,
    options = IpoptOptions(eval_hessian = false))

println("\nDone.")
