# =============================================================================
# robust_iswap_detuned_1MHz_300ns_3level_stretched_warmstart.jl
#
# STRETCHED-TIME variant of the 3-level DRAG-warmstart optimization.
#
# Construction:
#   - All amplitudes (g_eff, a_bound, u_X1/Y1/X2/Y2) HALVED
#   - All time scales (σ_rise, buffer_flat, T_total) DOUBLED
#   - N_knots = 24 unchanged → Δt doubled (so spline bandwidth halved)
#   - du tangents divided by 4 (since du = d(u)/d(t) and u→u/2, t→2t)
#   - All rotation areas ∫g·dt and ∫u·dt preserved → same iSWAP target
#   - g_eff_new = 1 MHz coupling (so file naming "1MHz" reflects effective coupling
#     in the stretched-physics frame), even though the HARDWARE coupling is
#     unchanged — this script just runs the optimization in a regime where the
#     time-averaged g is halved by gate construction.
#
# Why we expect this to give better robustness:
#   1. Ω/|η| drops 0.06 → 0.03 → leading-order DRAG residual drops 4× from 4×10⁻³
#      to 10⁻³. Means optimizer's |2⟩-population manifold is structurally cleaner.
#   2. Q_r = 1000 (vs 100 in the 150 ns run): robustness pressure increased 10×.
#      The strict F_threshold = 0.9999 constraint prevents leakage gaming, so
#      the optimizer is forced to find robustness in the high-purity manifold.
#   3. With strict F + low leakage from start, the variational sensitivity
#      ‖∂U/∂n̂‖² has more freedom to drop without the optimizer being lured into
#      the bad |2⟩-mediated basin.
#
# Warm-start source:
#   robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_dragwarmstart_seed42
#   (the 200-iter run that got F_flat = 0.9999, leakage = 4×10⁻⁵, but barely
#    moved robustness over the 2-level baseline — visible in the user's plot).
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
# Strict subspace process fidelity constraint (same as the 150 ns warm-start)
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
# 3-level Duffing operators
# =============================================================================
const n_lvl = 3
b = zeros(ComplexF64, n_lvl, n_lvl)
for j in 1:n_lvl-1
    b[j, j+1] = sqrt(j)
end
const b_3  = copy(b)
const bd_3 = b_3'
const I3   = Matrix{ComplexF64}(I, n_lvl, n_lvl)
const I9   = Matrix{ComplexF64}(I, n_lvl^2, n_lvl^2)

const X3 = b_3 + bd_3
const Y3 = im * (bd_3 - b_3)
const n3 = bd_3 * b_3

const XI = kron(X3, I3); const YI = kron(Y3, I3)
const IX = kron(I3, X3); const IY = kron(I3, Y3)
const XX = kron(X3, X3); const YY = kron(Y3, Y3)
const nI = kron(n3, I3)
const In = kron(I3, n3)
const nn = nI * In

# =============================================================================
# Physical parameters — STRETCHED (g and u halved, times doubled)
# =============================================================================
const g_eff = 2π * 0.001            # 1 MHz coupling (was 2 MHz) — halved
const δ₁₂   = 2π * 0.06             # 60 MHz idle detuning (unchanged)
const Δ_mw1 = -2π * 0.00
const Δ_mw2 = +2π * 0.00
const a_bound      = 2π * 0.01     # 5 MHz amplitude bound (was 10 MHz) — halved
const drive_bounds = fill(a_bound, 4)

const σ_rise              = 2.5     # was 1.25 — doubled
const buffer_flat_duration = 10.0   # was 5.0 — doubled
const buffer_duration     = 4 * σ_rise   # = 10 ns — doubled

const η_anh = -2π * 0.170           # -170 MHz unchanged (it's a hardware property)
const H_anh = (η_anh / 2) * (nI * (nI - I9) + In * (In - I9))

# =============================================================================
# Optimization knobs
# =============================================================================
const F_threshold     = 0.9999
const Q_r             = 1e3         # 10× the 150 ns run's Q_r (was 1e2)
const T_total_gate_ns = 300.0       # was 150 — doubled
const T_total_ns      = T_total_gate_ns - 2 * buffer_duration - 2 * buffer_flat_duration
@assert T_total_ns > 0              # = 260 ns

const N_knots   = 24                # unchanged (so Δt doubled)
const num_iter  = 200                # local probe; bump for overnight SSH
const SEED      = 42

# Warm-start source — the just-finished 150 ns 3-lvl DRAG-warmstart run
const WARMSTART_DIR = joinpath(
    @__DIR__,
    "robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_3lvl_170MHzanh_rollout_dragwarmstart_seed42",
)
@assert isdir(WARMSTART_DIR)  "Warm-start directory not found: $WARMSTART_DIR"
@assert isfile(joinpath(WARMSTART_DIR, "trajectory.jld2"))  "trajectory.jld2 missing in $WARMSTART_DIR"

println("3-level Duffing rollout — STRETCHED warm-start (g, u halved; time doubled)")
println("Warm-start dir: $(basename(WARMSTART_DIR))")
println("New params: T_total = $T_total_gate_ns ns, g_eff = $(g_eff/(2π)*1e3) MHz, " *
        "a_bound = $(a_bound/(2π)*1e3) MHz")
println("σ_rise = $σ_rise ns; buffer = $buffer_flat_duration ns; mw region = $T_total_ns ns")
println("Q_r = $Q_r (10× the 150 ns run)")
println("N_knots = $N_knots, num_iter = $num_iter, seed = $SEED")

# =============================================================================
# Gaussian edges + V_rise / V_fall + U_goal (same rotation angles as before)
# =============================================================================
const dt_fine = 0.02
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
@printf("A_gauss = %.4f rad, A_buffer = %.4f rad, A_pre = %.4f rad, θ_goal = %.4f rad\n",
    A_gauss, A_buffer, A_pre, θ_goal)

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
const run_tag   = "$(g_eff_MHz)MHz_$(T_mw_int)nsmw_10nsgauss_10nsbuf_3lvl_$(η_MHz)MHzanh_stretched_Qr$(round(Int,Q_r))_seed$(SEED)"
println("run_tag = $run_tag")

# =============================================================================
# Gate-frame Hamiltonian (3-level, with anharmonicity) — same structure
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
# Load 3-level DRAG-warmstart trajectory and STRETCH:
#   u_new[k]  = u_old[k] / 2          (amplitudes halved)
#   du_new[k] = du_old[k] / 4          (since du = d(u)/d(t); u→u/2, t→2t  ⇒  du→du/4)
#   t_new[k]  = 2 · t_old[k]           (time stretched 2×)
# =============================================================================
warm_traj = nothing
@load joinpath(WARMSTART_DIR, "trajectory.jld2") traj
warm_traj = traj
@assert warm_traj.N == N_knots  "Warm-start trajectory has N_knots=$(warm_traj.N), expected $N_knots"

u_old   = warm_traj[:u]    # 4 × N_knots
du_old  = warm_traj[:du]   # 4 × N_knots
t_old   = vec(warm_traj[:t])

u_stretched  = u_old ./ 2
du_stretched = du_old ./ 4
t_stretched  = t_old .* 2

@printf("Stretched warm-start construction:\n")
@printf("  old time range: 0 → %.2f ns;  new: 0 → %.2f ns\n", t_old[end], t_stretched[end])
@printf("  old |u|∞ = %.5f rad/ns;       new: %.5f rad/ns  (a_bound = %.5f rad/ns)\n",
    maximum(abs, u_old), maximum(abs, u_stretched), a_bound)
@printf("  old |du|∞ = %.5f rad/ns²;     new: %.5f rad/ns²\n",
    maximum(abs, du_old), maximum(abs, du_stretched))

if maximum(abs, u_stretched) > a_bound
    @warn "Stretched amplitudes exceed new a_bound by $(round(100*(maximum(abs, u_stretched)/a_bound - 1), digits=2))% — clipping."
    u_stretched = clamp.(u_stretched, -a_bound, a_bound)
end

# =============================================================================
# Build the 3-level problem with the stretched-warmstart CubicSplinePulse
# =============================================================================
T_f = T_total_ns
Δt  = T_f / N_knots

pulse = CubicSplinePulse(u_stretched, du_stretched, t_stretched)

qcp = VariationalRolloutProblem(
    varsys, pulse, U_goal, N_knots;
    Q                     = 0.0,
    Q_r                   = Q_r,      # 10× higher robustness weight in objective
    R                     = 1e-3,
    du_bound              = Inf,
    Δt_bounds             = (Δt, Δt),
    dynamics_spline_order = 3,
    n_path_samples        = 3,
    piccolo_options       = PiccoloOptions(
        timesteps_all_equal       = true,
        verbose                   = true,
        leakage_constraint        = true,
        leakage_constraint_value  = 1e-3,
        leakage_cost              = 1.0,
    ),
)

# Strict process fidelity constraint (the same fix that worked for the 150 ns run)
push!(qcp.prob.constraints,
    FinalSubspaceProcessFidelityConstraint(
        U_goal_4x4, subspace_indices, :Ũ⃗, F_threshold, get_trajectory(qcp)
    )
)

# Rollout-initialize :Ũ⃗ under the stretched controls in 3-level Duffing
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
    @printf("Initial fidelity (stretched-warmstart, comp subspace): %.6f, leak %.3e\n", F0, leak0)
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
@printf("Full gate F to iSWAP (script's static-V_rise calc): %.8f\n", F_full)
@printf("  NOTE: F_full here uses static exp(-iA(XX+YY)) for V_rise/V_fall; not\n")
@printf("        physically faithful in 3-level. Use the verify notebook for true F_full.\n")

# Integrator-honesty check
let
    nominal_sys_v = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
    Ũ⃗_ref = unitary_rollout(traj, nominal_sys_v;
        interpolation = :cubic_hermite, abstol = 1e-12, reltol = 1e-12)
    U_flat_ref      = iso_vec_to_operator(Ũ⃗_ref[:, end])
    U_flat_ref_sub  = U_flat_ref[subspace_indices, subspace_indices]
    F_flat_ref      = abs2(tr(U_goal_4x4' * U_flat_ref_sub)) / 4^2
    leak_ref        = 1 - real(tr(U_flat_ref_sub' * U_flat_ref_sub)) / 4

    @printf("\n--- Integrator-honesty check ---\n")
    @printf("Flat-top F  (NLP report):            %.8f\n", F_flat)
    @printf("Flat-top F  (independent reference): %.8f\n", F_flat_ref)
    @printf("              |ΔF|:                   %.2e\n", abs(F_flat - F_flat_ref))
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
# Save outputs
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
    println(io, "# Stretched 3-level Duffing rollout (g, u halved; time doubled)")
    println(io, "template = VariationalRolloutProblem + strict subspace F constraint")
    println(io, "warmstart_source = $(basename(WARMSTART_DIR))")
    println(io, "warmstart_strategy = u → u/2, du → du/4, t → 2t")
    println(io, "η_anh = $η_anh rad/ns ($(round(η_anh/(2π)*1e3, digits=1)) MHz)")
    println(io, "g_eff = $g_eff rad/ns ($(round(g_eff/(2π)*1e3, digits=1)) MHz) [halved]")
    println(io, "δ₁₂   = $δ₁₂ rad/ns")
    println(io, "a_bound = $a_bound rad/ns [halved]")
    println(io, "σ_rise = $σ_rise ns [doubled]")
    println(io, "buffer_duration = $buffer_duration ns [doubled]")
    println(io, "buffer_flat_duration = $buffer_flat_duration ns [doubled]")
    println(io, "T_total_gate_ns = $T_total_gate_ns ns [doubled]")
    println(io, "T_total_ns (mw region) = $T_total_ns ns")
    println(io, "θ_goal = $θ_goal rad")
    println(io, "Q = 0.0, Q_r = $Q_r [10× the 150 ns run], R = 1e-3")
    println(io, "F_threshold = $F_threshold (strict subspace process F)")
    println(io, "N_knots = $N_knots, num_iter = $num_iter, seed = $SEED")
    println(io, "Variational errors: n̂_1, n̂_2, n̂_1·n̂_2")
    println(io, "Computational subspace indices: $subspace_indices")
    println(io, "F_flat (subspace) = $F_flat")
    println(io, "Leakage           = $leak")
    println(io, "F_full (vs iSWAP, static-V_rise calc) = $F_full")
end

println("\nSaved to: ", abspath(outdir))
println("Next: verify in duffing_3level_verify_5nsbuf.ipynb (set RUN_DIR to the new outdir).")
