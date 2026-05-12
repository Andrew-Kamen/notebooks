# =============================================================================
# robust_iswap_detuned_2MHz_150ns_5nsbuf_rollout_1kiter.jl
#
# 2-qubit (2-level Pauli) variational-rollout version of
# robust_iswap_detuned_2MHz_150ns_5nsbuf_10kiter.jl
#
# Same physics setup (g_eff=2MHz, 5 ns Gauss + 5 ns buf + 130 ns MW + 5 ns buf
# + 5 ns Gauss, ZI/IZ/ZZ robustness, 4 detuned drives), but uses
# `VariationalRolloutProblem` (indirect / forward-rollout formulation) instead
# of `VariationalSplinePulseProblem` (direct collocation). At 2-level this is
# only a modest speedup vs the direct version (the augmented :var_Ũ⃗ block is
# small), but it shares all the rollout-side speedups in `optimizations.md`
# with the 3-level script — see that doc for the per-call breakdown.
#
# Differences from the direct 10kiter script:
#   * `VariationalRolloutProblem` instead of `VariationalSplinePulseProblem`
#   * num_iter = 1000 (vs 10000)
#   * `constr_viol_tol = 1e-8` in IpoptOptions — needed for the NLP-reported F
#     to actually match the cubic-Hermite-driven physical dynamics. Default
#     1e-4 leaves a ~10^-3 fictitious-fidelity gap; see optimizations.md.
#   * Post-solve integrator-honesty check (re-evolves the optimised controls
#     with high-accuracy `unitary_rollout` and reports the gap).
# =============================================================================

# # Imports
import Pkg
Pkg.activate(@__DIR__)
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
Pkg.develop([
    Pkg.PackageSpec(path = piccolo_path),
    Pkg.PackageSpec(path = directtrajopt_path),
])
Pkg.add(["CairoMakie", "Ipopt", "FFTW", "JLD2"])
Pkg.instantiate()

using Piccolo
using LinearAlgebra
using Random
using Printf
using CairoMakie
using FFTW
using JLD2

# # Pauli operators (2-level, 4-dim Hilbert)
XX = operator_from_string("XX")
YY = operator_from_string("YY")
XI = operator_from_string("XI")
YI = operator_from_string("YI")
IX = operator_from_string("IX")
IY = operator_from_string("IY")
IZ = operator_from_string("IZ")
ZI = operator_from_string("ZI")
ZZ = operator_from_string("ZZ")

# # Physical parameters
const g_eff   = 2π * 0.002          # 2 MHz coupling
const δ₁₂     = 2π * 0.06           # 60 MHz idle detuning
const Δ_mw1   = -2π * 0.00
const Δ_mw2   = +2π * 0.00
const a_bound = 2π * 0.01           # 10 MHz amplitude bound
const drive_bounds = fill(a_bound, 4)

const σ_rise              = 1.25
const buffer_flat_duration = 5.0
const buffer_duration     = 4 * σ_rise

# # Target
const U_iswap = exp(-im * π/4 * (XX + YY))

# # Optimization knobs
const F_threshold     = 0.9999
const Q_r             = 1e2
const T_total_gate_ns = 150.0
const T_total_ns      = T_total_gate_ns - 2 * 4 * σ_rise - 2 * buffer_flat_duration
@assert T_total_ns > 0

const n_samples = 300
const N_knots   = 24
const num_iter  = 1000
const SEED      = 42

println("2-qubit (Pauli) variational-rollout iSWAP optimization.")
println("Hilbert dim = 4")
println("σ_rise = $σ_rise ns; buffer = $buffer_flat_duration ns; mw region = $T_total_ns ns; total = $T_total_gate_ns ns")
println("N_knots = $N_knots, num_iter = $num_iter, seed = $SEED")

# # Compute V_rise, V_fall, U_goal
const dt_fine = 0.01
let
    t_rise = collect(0:dt_fine:buffer_duration)
    g_rise = g_eff .* exp.(-(t_rise .- buffer_duration).^2 ./ (2 * σ_rise^2))
    global A_gauss  = sum(g_rise) * dt_fine
    global A_buffer = g_eff * buffer_flat_duration
    global A_pre    = A_gauss + A_buffer
    global A_post   = A_pre
end
const θ_goal = π/4 - A_pre - A_post
@assert θ_goal > 0

const U_goal = exp(-im * θ_goal * (XX + YY))
const V_rise = exp(-im * A_pre  * (XX + YY))
const V_fall = exp(-im * A_post * (XX + YY))

# Verification: V_fall · U_goal · V_rise = iSWAP (commutes since all are
# (XX+YY) rotations)
let U_check = V_fall * U_goal * V_rise
    @printf("Sanity: F(V_fall · U_goal · V_rise, iSWAP) = %.10f\n",
            abs2(tr(U_iswap' * U_check)) / 16)
end

const g_eff_MHz = round(Int, g_eff/(2π)*1e3)
const T_mw_ns   = round(Int, T_total_ns)
const run_tag   = "$(g_eff_MHz)MHz_$(T_mw_ns)nsmw_5nsgauss_5nsbuf_rollout_1kiter_seed$(SEED)"
println("θ_goal = ", round(θ_goal, digits=6), " rad")
println("run_tag = $run_tag")

# # Gate-frame Hamiltonian (2-level)
function H_gate_frame(u, t)
    uX1, uY1, uX2, uY2 = u
    H = g_eff * (XX + YY)
    c1 = cos(Δ_mw1 * t); s1 = sin(Δ_mw1 * t)
    H += uX1 * (XI * c1 + YI * s1)
    H += uY1 * (YI * c1 - XI * s1)
    c2 = cos(Δ_mw2 * t); s2 = sin(Δ_mw2 * t)
    H += uX2 * (IX * c2 + IY * s2)
    H += uY2 * (IY * c2 - IX * s2)
    return H
end

const H_vars = [
    (u, t) -> ZI,
    (u, t) -> IZ,
    (u, t) -> ZZ,
]

const varsys = VariationalQuantumSystem(
    H_gate_frame, H_vars, 4, drive_bounds;
    time_dependent = true,
)
println("varsys.levels = $(varsys.levels), n_drives = $(varsys.n_drives), n_vars = $(length(varsys.G_vars))")

# # Build problem (VariationalRolloutProblem — indirect/rollout formulation)
T_f = T_total_ns
Δt  = T_f / N_knots

Random.seed!(SEED)
controls = 2 .* a_bound .* rand(4, n_samples) .- a_bound
times    = collect(LinRange(0.0, T_f, n_samples))
du_init  = zeros(4, n_samples)
pulse    = CubicSplinePulse(controls, du_init, times)

qcp = VariationalRolloutProblem(
    varsys, pulse, U_goal, N_knots;
    Q                     = 0.0,
    Q_r                   = Q_r,
    R                     = 1e-3,
    du_bound              = Inf,
    Δt_bounds             = (Δt, Δt),
    dynamics_spline_order = 3,
    n_path_samples        = 3,
    piccolo_options       = PiccoloOptions(
        timesteps_all_equal = true,
        verbose             = true,
    ),
)

push!(qcp.prob.constraints,
    FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F_threshold, get_trajectory(qcp))
)

# Cache-status diagnostic — both should be ACTIVE for this Hamiltonian
let rollout_obj = nothing
    for sub in qcp.prob.objective.objectives
        if sub isa VariationalRolloutObjective
            rollout_obj = sub
            break
        end
    end
    if !isnothing(rollout_obj)
        println("VariationalRolloutObjective caches:")
        println("  cached_g_vars   : ", isnothing(rollout_obj.cached_g_vars)
            ? "FALLBACK (per-call)" : "ACTIVE ($(length(rollout_obj.cached_g_vars)) directions cached)")
        println("  cached_g_linear : ", isnothing(rollout_obj.cached_g_linear)
            ? "FALLBACK (per-call)" : "ACTIVE (drift + $(length(rollout_obj.cached_g_linear[2])) drives cached)")
    end
end

# Rollout-initialise nominal unitary trajectory (no :var_Ũ⃗ in rollout template)
let
    traj    = get_trajectory(qcp)
    us      = traj[:u]; dus = traj[:du]; ts = vec(traj[:t])

    nominal_sys = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
    nlp_pulse   = CubicSplinePulse(us, dus, ts)
    nlp_qtraj   = UnitaryTrajectory(nominal_sys, nlp_pulse, U_goal)

    Ũ⃗_init = hcat([operator_to_iso_vec(nlp_qtraj(t)) for t in ts]...)
    traj.data[traj.components[:Ũ⃗], :] .= Ũ⃗_init

    U_init = iso_vec_to_operator(Ũ⃗_init[:, end])
    F0 = abs2(tr(U_goal' * U_init)) / 16
    @printf("Initial fidelity (rollout init): %.6f\n", F0)
end

# # Solve
#
# `constr_viol_tol = 1e-8` (vs Ipopt default 1e-4) is the critical setting for
# the NLP-reported :Ũ⃗[:, end] to actually match a high-accuracy independent
# re-rollout. See `optimizations.md` for the analysis.
println("\nStarting solve (num_iter = $num_iter)...")
solve_t0 = time()
solve!(qcp; max_iter = num_iter, print_level = 5,
    options = IpoptOptions(
        eval_hessian = false,
        constr_viol_tol = 1e-8,
        tol             = 1e-8,
        acceptable_tol  = 1e-8,
        output_file = "ipopt_$(run_tag).log",
    ))
solve_wall = time() - solve_t0
@printf("Solve wall time: %.1f s  (%.3f s / iter)\n", solve_wall, solve_wall / num_iter)

# # Results
traj   = get_trajectory(qcp)
U_flat = iso_vec_to_operator(traj[:Ũ⃗][:, end])
F_flat = abs2(tr(U_goal' * U_flat)) / 16
@printf("Flat-top F to U_goal: %.8f\n", F_flat)
@printf("Flat-top infidelity:  %.2e\n", 1 - F_flat)

U_full = V_fall * U_flat * V_rise
F_full = abs2(tr(U_iswap' * U_full)) / 16
@printf("Full gate F to iSWAP: %.8f\n", F_full)
@printf("Full gate infidelity: %.2e\n", 1 - F_full)

T_total = 2 * buffer_duration + 2 * buffer_flat_duration + traj[:t][end]
φ_vz    = δ₁₂ * T_total
println("Total gate time = ", round(T_total, digits=2), " ns")
println("Virtual Z on q2 = ", round(φ_vz, digits=4), " rad")

# # Integrator-honesty check
# Re-evolve the optimised controls with high-accuracy independent ODE and
# confirm the NLP isn't lying about the fidelity. With constr_viol_tol = 1e-8
# the gap should be ≲ 1e-5.
let
    nominal_sys_v = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
    Ũ⃗_ref = unitary_rollout(traj, nominal_sys_v;
        interpolation = :cubic_hermite, abstol = 1e-12, reltol = 1e-12)
    U_flat_ref      = iso_vec_to_operator(Ũ⃗_ref[:, end])
    F_flat_ref      = abs2(tr(U_goal' * U_flat_ref)) / 16

    U_full_ref      = V_fall * U_flat_ref * V_rise
    F_full_ref      = abs2(tr(U_iswap' * U_full_ref)) / 16

    @printf("\n--- Integrator-honesty check (high-acc independent re-rollout) ---\n")
    @printf("Flat-top F  (NLP report):           %.8f\n", F_flat)
    @printf("Flat-top F  (independent reference):%.8f\n", F_flat_ref)
    @printf("              |ΔF|:                  %.2e\n", abs(F_flat - F_flat_ref))
    @printf("Full-gate F (NLP report):           %.8f\n", F_full)
    @printf("Full-gate F (independent reference):%.8f\n", F_full_ref)
    @printf("              |ΔF|:                  %.2e\n", abs(F_full - F_full_ref))
    if abs(F_flat - F_flat_ref) > 1e-4
        @warn "NLP fidelity overstates reality by >1e-4 — the optimizer's reported F is fictitious. Tighten constr_viol_tol further."
    end
end

# # Save controls + trajectory + parameters
outdir = "robust_iswap_detuned_$(run_tag)"
mkpath(joinpath(outdir, "figs"))

function _spline_fine(traj)
    ts = vec(traj[:t]); us = traj[:u]; dus = traj[:du]
    pulse_loc = CubicSplinePulse(us, dus, ts)
    ts_f = collect(LinRange(ts[1], ts[end], 2000))
    us_f = sample(pulse_loc, ts_f)
    return ts, us, ts_f, us_f
end

ts_knots, us_knots, ts_fine, us_fine = _spline_fine(traj)
dus_knots = traj[:du]

open(joinpath(outdir, "controls_knots.csv"), "w") do io
    println(io, "time_ns,u_X1,u_Y1,u_X2,u_Y2,du_X1,du_Y1,du_X2,du_Y2")
    for j in eachindex(ts_knots)
        @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
            ts_knots[j],
            us_knots[1,j],  us_knots[2,j],  us_knots[3,j],  us_knots[4,j],
            dus_knots[1,j], dus_knots[2,j], dus_knots[3,j], dus_knots[4,j])
    end
end

open(joinpath(outdir, "controls_fine.csv"), "w") do io
    println(io, "time_ns,u_X1,u_Y1,u_X2,u_Y2")
    for j in eachindex(ts_fine)
        @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g\n",
            ts_fine[j], us_fine[1,j], us_fine[2,j], us_fine[3,j], us_fine[4,j])
    end
end

@save joinpath(outdir, "trajectory.jld2") traj=traj

open(joinpath(outdir, "parameters.txt"), "w") do io
    println(io, "# 2-qubit (Pauli) variational-rollout iSWAP optimization, 5nsbuf layout")
    println(io, "template = VariationalRolloutProblem")
    println(io, "n_levels_per_qubit = 2")
    println(io, "g_eff = $g_eff rad/ns ($(round(g_eff/(2π)*1e3, digits=1)) MHz)")
    println(io, "δ₁₂   = $δ₁₂ rad/ns")
    println(io, "Δ_mw1 = $Δ_mw1 rad/ns; Δ_mw2 = $Δ_mw2 rad/ns")
    println(io, "a_bound = $a_bound rad/ns")
    println(io, "σ_rise = $σ_rise ns")
    println(io, "buffer_duration = $buffer_duration ns")
    println(io, "buffer_flat_duration = $buffer_flat_duration ns")
    println(io, "T_total_gate_ns = $T_total_gate_ns ns")
    println(io, "T_total_ns (mw region) = $T_total_ns ns")
    println(io, "θ_goal = $θ_goal rad")
    println(io, "Q = 0.0, Q_r = $Q_r, R = 1e-3")
    println(io, "F_threshold = $F_threshold")
    println(io, "N_knots = $N_knots, num_iter = $num_iter, seed = $SEED")
    println(io, "Variational errors: ZI, IZ, ZZ")
    println(io, "Ipopt: constr_viol_tol = 1e-8, tol = 1e-8, acceptable_tol = 1e-8")
    println(io, "F_flat = $F_flat")
    println(io, "F_full (vs iSWAP) = $F_full")
end

println("\nSaved to: ", abspath(outdir))
