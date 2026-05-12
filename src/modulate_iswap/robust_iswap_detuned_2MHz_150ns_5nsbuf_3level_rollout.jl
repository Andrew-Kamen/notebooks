# =============================================================================
# robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_rollout.jl
#
# Same problem as robust_iswap_detuned_2MHz_150ns_5nsbuf_3level.jl
# (3-level Duffing, anharmonicity η = -2π·0.170, n̂_1/n̂_2/n̂_1·n̂_2 robustness,
# 5 ns Gauss + 5 ns buf + 130 ns MW + 5 ns buf + 5 ns Gauss pulse layout)
# but uses VariationalRolloutProblem (indirect / forward-rollout formulation)
# instead of VariationalSplinePulseProblem (direct collocation).
#
# WHY: the direct template carries the augmented [Ũ⃗; ∂Ũ⃗₁; ∂Ũ⃗₂; ∂Ũ⃗₃] state as
# an NLP decision variable, giving per-knot Jacobian cost O((iso_dim·(1+n_vars))³)
# — for n_vars=3 and iso_dim=162 (3-level, 2 qubits) that's a 648³ matrix-exp
# per knot transition. The rollout version drops :var_Ũ⃗ from the NLP and
# computes ‖∂Ũ⃗(T)‖² by forward-integrating the variational ODE inside the
# objective. Gradient via ForwardDiff through that rollout (consistent with
# how every other Piccolo integrator computes its Jacobian).
#
# Expected speedup at 3-level: 5-20× per Ipopt iteration.
# =============================================================================

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

# =============================================================================
# 3-level Duffing operators (single mode)
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

# =============================================================================
# 9-dim 2-qubit operators
# =============================================================================
const I9 = Matrix{ComplexF64}(I, n_lvl^2, n_lvl^2)

const XI = kron(X3, I3); const YI = kron(Y3, I3)
const IX = kron(I3, X3); const IY = kron(I3, Y3)
const XX = kron(X3, X3); const YY = kron(Y3, Y3)
const XY = kron(X3, Y3); const YX = kron(Y3, X3)

# Number operators — variational error directions
const nI = kron(n3, I3)
const In = kron(I3, n3)
const nn = nI * In

# =============================================================================
# Physical parameters (identical to the direct 3-level script)
# =============================================================================
const g_eff = 2π * 0.002       # 2 MHz coupling
const δ₁₂   = 2π * 0.06         # 60 MHz idle detuning
const Δ_mw1 = -2π * 0.00
const Δ_mw2 = +2π * 0.00
const a_bound      = 2π * 0.01
const drive_bounds = fill(a_bound, 4)

const σ_rise              = 1.25
const buffer_flat_duration = 5.0
const buffer_duration     = 4 * σ_rise

const η_anh = -2π * 0.170                        # -170 MHz (rad/ns)
const H_anh = (η_anh / 2) * (nI * (nI - I9) + In * (In - I9))

# =============================================================================
# Optimization knobs (match direct 3-level)
# =============================================================================
const F_threshold     = 0.9999
const Q_r             = 1e2
const T_total_gate_ns = 150.0
const T_total_ns      = T_total_gate_ns - 2 * 4 * σ_rise - 2 * buffer_flat_duration
@assert T_total_ns > 0

const n_samples = 300
const N_knots   = 20
const num_iter  = 1000
const SEED      = 42

println("3-level Duffing rollout optimization. Hilbert dim = $(n_lvl^2). η = $(round(η_anh/(2π)*1e3, digits=1)) MHz.")
println("σ_rise = $σ_rise ns; buffer = $buffer_flat_duration ns; mw region = $T_total_ns ns; total = $T_total_gate_ns ns")
println("seed = $SEED")

# =============================================================================
# Gaussian edges + V_rise / V_fall + U_goal
# =============================================================================
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

const U_iswap_4x4 = let
    σx = ComplexF64[0.0 1.0; 1.0 0.0]
    σy = ComplexF64[0.0 -im; im 0.0]
    exp(-im * π/4 * (kron(σx, σx) + kron(σy, σy)))
end
const U_goal_4x4 = let
    σx = ComplexF64[0.0 1.0; 1.0 0.0]
    σy = ComplexF64[0.0 -im; im 0.0]
    exp(-im * θ_goal * (kron(σx, σx) + kron(σy, σy)))
end
const subspace_indices = [1, 2, 4, 5]
const U_goal  = EmbeddedOperator(U_goal_4x4,  subspace_indices, [n_lvl, n_lvl])
const U_iswap = EmbeddedOperator(U_iswap_4x4, subspace_indices, [n_lvl, n_lvl])

const V_rise = exp(-im * A_pre  * (XX + YY))
const V_fall = exp(-im * A_post * (XX + YY))

const g_eff_MHz = round(Int, g_eff/(2π)*1e3)
const η_MHz     = round(Int, abs(η_anh)/(2π)*1e3)
const run_tag   = "$(g_eff_MHz)MHz_$(round(Int, T_total_ns))nsmw_5nsgauss_5nsbuf_3lvl_$(η_MHz)MHzanh_rollout_seed$(SEED)_Q100"
println("θ_goal = ", round(θ_goal, digits=4), " rad")
println("run_tag = $run_tag")

# =============================================================================
# Gate-frame Hamiltonian (3-level, with anharmonicity)
# =============================================================================
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

const H_vars = [
    (u, t) -> nI,
    (u, t) -> In,
    (u, t) -> nn,
]

const varsys = VariationalQuantumSystem(
    H_gate_frame, H_vars, 4, drive_bounds;
    time_dependent = true,
)
println("varsys.levels = $(varsys.levels) (expected $(n_lvl^2))")

# =============================================================================
# Build problem (VariationalRolloutProblem — indirect/rollout formulation)
# =============================================================================
T_f = T_total_ns
Δt  = T_f / N_knots

Random.seed!(SEED)
controls = 2 .* a_bound .* rand(4, n_samples) .- a_bound
times    = collect(LinRange(0.0, T_f, n_samples))
du_init  = zeros(4, n_samples)
pulse    = CubicSplinePulse(controls, du_init, times)

qcp = VariationalRolloutProblem(
    varsys, pulse, U_goal, N_knots;
    Q                     = 100.0,
    Q_r                   = Q_r,
    R                     = 1e-3,
    du_bound              = Inf,
    Δt_bounds             = (Δt, Δt),
    dynamics_spline_order = 3,
    n_path_samples        = 3,
    piccolo_options       = PiccoloOptions(
        timesteps_all_equal = true,
        verbose             = true,
        leakage_constraint  = true,
        leakage_constraint_value  = 1e-3,
        leakage_cost              = 1.0,
    ),
)

# push!(qcp.prob.constraints,
#     FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F_threshold, get_trajectory(qcp))
# )

# Diagnostics: confirm which speed caches in VariationalRolloutObjective fired.
# Both should be populated for this Hamiltonian: H_vars are constant matrices
# (nI, In, nn) and H_gate_frame is linear in u with Δ_mw=0 making it effectively
# time-independent. If either is `nothing`, the optimization fell back to the
# slower per-RHS code path for that piece.
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

# Rollout-initialize the nominal unitary trajectory (anharmonicity included).
# In the rollout template there is NO :var_Ũ⃗ component in the trajectory, so
# we only need to warm-start :Ũ⃗.
let
    traj    = get_trajectory(qcp)
    us      = traj[:u]; dus = traj[:du]; ts = vec(traj[:t])

    nominal_sys = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
    nlp_pulse   = CubicSplinePulse(us, dus, ts)
    nlp_qtraj   = UnitaryTrajectory(nominal_sys, nlp_pulse, U_goal)

    Ũ⃗_init = hcat([operator_to_iso_vec(nlp_qtraj(t)) for t in ts]...)
    traj.data[traj.components[:Ũ⃗], :] .= Ũ⃗_init

    U_init     = iso_vec_to_operator(Ũ⃗_init[:, end])
    U_init_sub = U_init[subspace_indices, subspace_indices]
    F0    = abs2(tr(U_goal_4x4' * U_init_sub)) / 4^2
    leak0 = 1 - real(tr(U_init_sub' * U_init_sub)) / 4
    @printf("Initial fidelity (rollout init, comp subspace): %.6f, leak %.3e\n", F0, leak0)
end

println("\nStarting solve (num_iter = $num_iter)...")
solve_t0 = time()
# constr_viol_tol = 1e-8 (vs Ipopt default 1e-4) is the critical setting that
# makes the NLP's reported :Ũ⃗[:, end] actually match a high-accuracy
# independent re-rollout of the optimized controls. With the default, F_nlp
# overstates the real fidelity by ~10⁻³ to ~10⁻⁴ because Ipopt declares
# success while bilinear-integrator residuals are still ~10⁻⁴ per row, and
# those compound across N−1 knot transitions. See `optimizations.md` (the
# "NLP-vs-reality fidelity gap" section) for the analysis.
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

# =============================================================================
# Results
# =============================================================================
traj   = get_trajectory(qcp)
U_flat = iso_vec_to_operator(traj[:Ũ⃗][:, end])

U_flat_sub = U_flat[subspace_indices, subspace_indices]
F_flat     = abs2(tr(U_goal_4x4' * U_flat_sub)) / 4^2
leak       = 1 - real(tr(U_flat_sub' * U_flat_sub)) / 4
@printf("Flat-top F to U_goal (computational): %.8f\n", F_flat)
@printf("Flat-top infidelity:                  %.2e\n", 1 - F_flat)
@printf("Leakage out of computational subspace: %.2e\n", leak)

U_full     = V_fall * U_flat * V_rise
U_full_sub = U_full[subspace_indices, subspace_indices]
F_full     = abs2(tr(U_iswap_4x4' * U_full_sub)) / 4^2
@printf("Full gate F to iSWAP (computational):  %.8f\n", F_full)

# -----------------------------------------------------------------------------
# Integrator-honesty check: re-evolve the optimized controls with an
# independent high-accuracy ODE solver and confirm the resulting unitary
# matches what the NLP reported. If `constr_viol_tol = 1e-8` is doing its job
# the gap should be ≲ 1e-5; a large gap means the optimiser was fooled by a
# loose constraint tolerance and the reported fidelity is fictitious.
# -----------------------------------------------------------------------------
let
    nominal_sys_v = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
    Ũ⃗_ref = unitary_rollout(traj, nominal_sys_v;
        interpolation = :cubic_hermite, abstol = 1e-12, reltol = 1e-12)
    U_flat_ref      = iso_vec_to_operator(Ũ⃗_ref[:, end])
    U_flat_ref_sub  = U_flat_ref[subspace_indices, subspace_indices]
    F_flat_ref      = abs2(tr(U_goal_4x4' * U_flat_ref_sub)) / 4^2
    leak_ref        = 1 - real(tr(U_flat_ref_sub' * U_flat_ref_sub)) / 4

    U_full_ref      = V_fall * U_flat_ref * V_rise
    U_full_ref_sub  = U_full_ref[subspace_indices, subspace_indices]
    F_full_ref      = abs2(tr(U_iswap_4x4' * U_full_ref_sub)) / 4^2

    @printf("\n--- Integrator-honesty check (high-acc independent re-rollout) ---\n")
    @printf("Flat-top F  (NLP report): %.8f\n", F_flat)
    @printf("Flat-top F  (independent reference): %.8f\n", F_flat_ref)
    @printf("              |ΔF|:        %.2e\n", abs(F_flat - F_flat_ref))
    @printf("Full-gate F (NLP report): %.8f\n", F_full)
    @printf("Full-gate F (independent reference): %.8f\n", F_full_ref)
    @printf("              |ΔF|:        %.2e\n", abs(F_full - F_full_ref))
    @printf("Leakage (independent):     %.2e  (NLP: %.2e)\n", leak_ref, leak)
    if abs(F_flat - F_flat_ref) > 1e-4
        @warn "NLP fidelity overstates reality by >1e-4 — the optimizer's reported F is fictitious. Tighten constr_viol_tol further."
    end
end

T_total = 2 * buffer_duration + 2 * buffer_flat_duration + traj[:t][end]
φ_vz    = δ₁₂ * T_total
println("Total gate time = ", round(T_total, digits=2), " ns")
println("Virtual Z on q2 = ", round(φ_vz, digits=4), " rad")

# =============================================================================
# Save controls + trajectory + parameters
# =============================================================================
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
            us_knots[1,j], us_knots[2,j], us_knots[3,j], us_knots[4,j],
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
    println(io, "# 3-level Duffing rollout-form optimization, 5nsbuf layout")
    println(io, "template = VariationalRolloutProblem")
    println(io, "n_levels_per_qubit = $n_lvl")
    println(io, "η_anh = $η_anh rad/ns ($(round(η_anh/(2π)*1e3, digits=1)) MHz)")
    println(io, "g_eff = $g_eff rad/ns ($(round(g_eff/(2π)*1e3, digits=1)) MHz)")
    println(io, "δ₁₂   = $δ₁₂ rad/ns")
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
    println(io, "Variational errors: n̂_1, n̂_2, n̂_1·n̂_2")
    println(io, "Computational subspace indices: $subspace_indices")
    println(io, "F_flat (subspace) = $F_flat")
    println(io, "Leakage           = $leak")
    println(io, "F_full (vs iSWAP, subspace) = $F_full")
end

println("\nSaved to: ", abspath(outdir))
println("Next: run duffing_3level_verify_5nsbuf.ipynb pointed at this outdir, replacing")
println("the source RUN_DIR with $(basename(outdir)), to verify the rollout-3-level pulse")
println("performs as advertised.")
