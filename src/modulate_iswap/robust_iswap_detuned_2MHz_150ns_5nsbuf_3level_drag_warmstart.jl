# =============================================================================
# robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_drag_warmstart.jl
#
# Same 3-level Duffing rollout setup as
#   robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_rollout.jl
# but WARM-STARTED from the analytically DRAG-corrected 2-level converged
# solution at
#   robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_rollout_1kiter_seed42/
#
# Initialization recipe:
#   1. Load 2-lvl trajectory.jld2  → (u_2lvl[k], du_2lvl[k]) at 24 knots
#   2. Apply symmetric DRAG at each knot:
#         uX1_new = uX1 + duY1/η
#         uY1_new = uY1 − duX1/η
#         uX2_new = uX2 + duY2/η
#         uY2_new = uY2 − duX2/η
#      Tangents du_new are kept = du_orig (optimizer will refine).
#   3. Use as the CubicSplinePulse initial guess for the 3-lvl problem.
#
# Why this is interesting: the leakage-through-time notebook showed the analytic
# DRAG correction lands at F ≈ 0.997 in 3-level Duffing with ~10⁻³ leakage,
# inheriting the 2-lvl robustness. The 3-lvl optimizer's job here is just to
# polish the leading-order DRAG residual (~3×10⁻³ infidelity) and handle the
# |11⟩↔|02⟩/|20⟩ coupling channel that single-qubit DRAG can't suppress.
#
# Expected convergence: tens to hundreds of iters (vs >1000 from random init),
# landing at F > 0.9999 with leakage ≲ 10⁻³.
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
using DirectTrajOpt: NonlinearKnotPointConstraint
using LinearAlgebra
using Random
using Printf
using CairoMakie
using FFTW
using JLD2

# ---------------------------------------------------------------------------
# Stricter fidelity constraint: bare comp-subspace process fidelity ≥ F_thr.
#
# Why this exists:
#   Piccolo's stock FinalUnitaryFidelityConstraint with an EmbeddedOperator
#   uses the *average* gate fidelity F_avg = (Tr(M†M) + |Tr(M)|²)/(n(n+1)),
#   where M = U_goal_4x4' * U_sub. The Tr(M†M) term is the comp-subspace
#   purity = n·(1 − leakage). So F_avg stays inflated when leakage is high
#   even if |Tr(M)|²/n² (the actual "did the gate work") collapses. That's
#   the failure mode the Q100 run exposed: F_avg satisfied at 0.9999 while
#   bare comp-subspace process F = 0.955 with 3.4% leakage.
#
# This constraint uses just |Tr(M)|² / n² — no purity inflation. Leakage
# reduces it directly (since |Tr(M)| ≤ √(n·Tr(M†M)) ≤ n·√(1−L)).
# ---------------------------------------------------------------------------
function FinalSubspaceProcessFidelityConstraint(
    U_goal_4x4::AbstractMatrix,
    subspace::AbstractVector{Int},
    state_name::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory,
)
    n = size(U_goal_4x4, 1)
    function terminal_constraint(Ũ⃗)
        U = iso_vec_to_operator(Ũ⃗)
        U_sub = U[subspace, subspace]
        F = abs2(tr(U_goal_4x4' * U_sub)) / n^2
        return [final_fidelity - F]
    end
    return NonlinearKnotPointConstraint(
        terminal_constraint,
        state_name,
        traj;
        equality = false,
        times = [traj.N],
    )
end

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

const nI = kron(n3, I3)
const In = kron(I3, n3)
const nn = nI * In

# =============================================================================
# Physical parameters
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
# Optimization knobs — match the 2-lvl warm-start grid (N_knots = 24)
# =============================================================================
const F_threshold     = 0.9999
const Q_r             = 1e2
const T_total_gate_ns = 150.0
const T_total_ns      = T_total_gate_ns - 2 * 4 * σ_rise - 2 * buffer_flat_duration
@assert T_total_ns > 0

# IMPORTANT: N_knots and the 2-lvl trajectory's knot count MUST match, since we
# load (u, du) at the 2-lvl knots and use them as the initial pulse. The 2-lvl
# rollout_1kiter run used N_knots = 24, T_mw = 130 ns.
const N_knots   = 24
const num_iter  = 200       # local 200-iter probe; bump to 1000+ for overnight SSH run
const SEED      = 42

# Warm-start source — must point at a 2-lvl run with N_knots = 24, T_mw = 130 ns
const WARMSTART_DIR = joinpath(
    @__DIR__,
    "robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_rollout_1kiter_seed42",
)
@assert isdir(WARMSTART_DIR)  "Warm-start directory not found: $WARMSTART_DIR"
@assert isfile(joinpath(WARMSTART_DIR, "trajectory.jld2"))  "trajectory.jld2 missing in $WARMSTART_DIR"

println("3-level Duffing rollout — DRAG-warmstart from 2-lvl converged solution.")
println("Warm-start dir: $(basename(WARMSTART_DIR))")
println("Hilbert dim    = $(n_lvl^2);  η = $(round(η_anh/(2π)*1e3, digits=1)) MHz")
println("σ_rise = $σ_rise ns; buffer = $buffer_flat_duration ns; mw region = $T_total_ns ns; total = $T_total_gate_ns ns")
println("N_knots = $N_knots, num_iter = $num_iter, seed = $SEED")

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
    σx::Matrix{ComplexF64} = ComplexF64[0.0 1.0; 1.0 0.0]
    σy::Matrix{ComplexF64} = ComplexF64[0.0 -im; im 0.0]
    exp(-im * π/4 * (kron(σx, σx) + kron(σy, σy)))
end
const U_goal_4x4 = let
    σx::Matrix{ComplexF64} = ComplexF64[0.0 1.0; 1.0 0.0]
    σy::Matrix{ComplexF64} = ComplexF64[0.0 -im; im 0.0]
    exp(-im * θ_goal * (kron(σx, σx) + kron(σy, σy)))
end
const subspace_indices = [1, 2, 4, 5]
const U_goal  = EmbeddedOperator(U_goal_4x4,  subspace_indices, [n_lvl, n_lvl])
const U_iswap = EmbeddedOperator(U_iswap_4x4, subspace_indices, [n_lvl, n_lvl])

const V_rise = exp(-im * A_pre  * (XX + YY))
const V_fall = exp(-im * A_post * (XX + YY))

const g_eff_MHz = round(Int, g_eff/(2π)*1e3)
const η_MHz     = round(Int, abs(η_anh)/(2π)*1e3)
const T_mw_int  = round(Int, T_total_ns)
const run_tag   = "$(g_eff_MHz)MHz_$(T_mw_int)nsmw_5nsgauss_5nsbuf_3lvl_$(η_MHz)MHzanh_rollout_dragwarmstart_seed$(SEED)"
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
# Load 2-lvl warm-start and apply analytic DRAG at the knots.
# Symmetric DRAG:  uX_new = uX + duY/η;  uY_new = uY − duX/η
# (matches the construction in leakage_through_time.ipynb cell `4a13482f`)
# du tangents are kept at their original 2-lvl values; the 3-lvl optimizer
# refines them as needed.
# =============================================================================
warm_traj = nothing
@load joinpath(WARMSTART_DIR, "trajectory.jld2") traj
warm_traj = traj
@assert warm_traj.N == N_knots  "Warm-start trajectory has N_knots=$(warm_traj.N), expected $N_knots"

u_2lvl  = warm_traj[:u]    # 4 × N_knots
du_2lvl = warm_traj[:du]   # 4 × N_knots
t_2lvl  = vec(warm_traj[:t])

# Apply symmetric DRAG at each knot
u_drag  = similar(u_2lvl)
u_drag[1, :] .= u_2lvl[1, :] .+ du_2lvl[2, :] ./ η_anh   # uX1 + duY1/η
u_drag[2, :] .= u_2lvl[2, :] .- du_2lvl[1, :] ./ η_anh   # uY1 − duX1/η
u_drag[3, :] .= u_2lvl[3, :] .+ du_2lvl[4, :] ./ η_anh   # uX2 + duY2/η
u_drag[4, :] .= u_2lvl[4, :] .- du_2lvl[3, :] ./ η_anh   # uY2 − duX2/η
du_drag = du_2lvl  # keep original tangents

# Diagnostic — how big is the DRAG correction, and does it stay within a_bound?
@printf("DRAG correction sizes (max |Δu| / a_bound):\n")
for q in 1:2
    Δ_uX = maximum(abs, u_drag[2q-1, :] .- u_2lvl[2q-1, :])
    Δ_uY = maximum(abs, u_drag[2q,   :] .- u_2lvl[2q,   :])
    @printf("  qubit %d:  ΔuX = %.2f%%,  ΔuY = %.2f%%\n",
        q, 100*Δ_uX/a_bound, 100*Δ_uY/a_bound)
end
@printf("  After DRAG: max |u| / a_bound = %.2f%%\n", 100*maximum(abs, u_drag)/a_bound)

if maximum(abs, u_drag) > a_bound
    @warn "DRAG-corrected pulse exceeds a_bound by $(round(100*(maximum(abs, u_drag)/a_bound - 1), digits=2))% — Ipopt will project to feasible at iter 1."
    # Clip to a_bound. Mild distortion at the DRAG correction's peak, but
    # the optimizer will refine from there.
    u_drag = clamp.(u_drag, -a_bound, a_bound)
end

# =============================================================================
# Build the 3-level problem with the DRAG-warmstarted CubicSplinePulse
# =============================================================================
T_f = T_total_ns
Δt  = T_f / N_knots

pulse = CubicSplinePulse(u_drag, du_drag, t_2lvl)

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
        timesteps_all_equal       = true,
        verbose                   = true,
        leakage_constraint        = true,     # silently ignored by VariationalRolloutProblem
        leakage_constraint_value  = 1e-3,     # (see optimizations.md "leakage bug")
        leakage_cost              = 1.0,      # but harmless to pass; flag for the future fix
    ),
)

push!(qcp.prob.constraints,
    FinalSubspaceProcessFidelityConstraint(
        U_goal_4x4, subspace_indices, :Ũ⃗, F_threshold, get_trajectory(qcp)
    )
)

# Cache-status diagnostic
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

# Rollout-initialize the nominal unitary trajectory under the DRAG-warmstarted
# controls, in the FULL 3-level Duffing (so :Ũ⃗ is consistent with the actual
# 9-dim dynamics from iter 0).
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
    @printf("Initial fidelity (DRAG-warmstart, comp subspace): %.6f, leak %.3e\n", F0, leak0)
    @printf("  → optimizer's job: polish from this point.\n")
end

println("\nStarting solve (num_iter = $num_iter)...")
solve_t0 = time()
solve!(qcp; max_iter = num_iter, print_level = 5,
    options = IpoptOptions(
        eval_hessian    = false,
        constr_viol_tol = 1e-8,
        tol             = 1e-8,
        acceptable_tol  = 1e-8,
        output_file     = "ipopt_$(run_tag).log",
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

# Integrator-honesty check
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
    @printf("Flat-top F  (NLP report):            %.8f\n", F_flat)
    @printf("Flat-top F  (independent reference): %.8f\n", F_flat_ref)
    @printf("              |ΔF|:                   %.2e\n", abs(F_flat - F_flat_ref))
    @printf("Full-gate F (NLP report):            %.8f\n", F_full)
    @printf("Full-gate F (independent reference): %.8f\n", F_full_ref)
    @printf("              |ΔF|:                   %.2e\n", abs(F_full - F_full_ref))
    @printf("Leakage (independent):                %.2e  (NLP: %.2e)\n", leak_ref, leak)
    if abs(F_flat - F_flat_ref) > 1e-4
        @warn "NLP fidelity overstates reality by >1e-4 — tighten constr_viol_tol further."
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
    println(io, "# 3-level Duffing rollout, DRAG-warmstart from 2-lvl converged solution")
    println(io, "template = VariationalRolloutProblem")
    println(io, "warmstart_source = $(basename(WARMSTART_DIR))")
    println(io, "warmstart_strategy = symmetric DRAG at knots, du unchanged")
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
println("Next: open in `duffing_3level_verify_5nsbuf.ipynb` to verify the")
println("DRAG-warmstart 3-lvl pulse against analytic-DRAG and original 3-lvl runs.")
