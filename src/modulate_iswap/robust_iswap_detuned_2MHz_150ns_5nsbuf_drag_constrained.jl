# =============================================================================
# robust_iswap_detuned_2MHz_150ns_5nsbuf_drag_constrained.jl
#
# 2-qubit (2-level Pauli) variational-rollout iSWAP, with DRAG enforced as a
# path-sample equality constraint at intermediate points between every pair of
# consecutive knots (mirroring how amplitude bounds are enforced between knots
# via CubicHermitePathConstraint).
#
# DRAG relation per qubit (asymmetric — X primary, Y derived):
#     u_Y1(t) + (du_X1(t) / η_drag) = 0
#     u_Y2(t) + (du_X2(t) / η_drag) = 0
#
# η_drag is the 3-level Duffing anharmonicity the pulse will be lifted into
# (set to -2π * 0.170 rad/ns = -170 MHz to match the transmon model).
#
# The optimization runs in 2-level (4-dim Pauli, no |2⟩, no anharmonicity in
# the dynamics), so the constraint is enforced purely on the controls. When
# the resulting pulse is later simulated in 3-level Duffing, leading-order DRAG
# theory predicts:
#   - leakage suppression: (Ω/η)^4 instead of (Ω/η)^2
#   - comp-subspace fidelity:  ≈ 1 - (Ω/η)^2 ≈ 0.996 for Ω/|η|=0.06
#
# This is the OPTIMIZED version of what the leakage_through_time notebook does
# analytically — the optimizer here is free to choose u_X1, u_X2 (and their
# tangents du_X1, du_X2) subject to fidelity + robustness, with u_Y1, u_Y2
# automatically forced into DRAG-compatible shape at every path sample.
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
Pkg.add(["CairoMakie", "Ipopt", "FFTW", "JLD2",
         "NamedTrajectories", "TrajectoryIndexingUtils", "ForwardDiff"])
Pkg.instantiate()

using Piccolo
using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Random
using Printf
using CairoMakie
using FFTW
using JLD2

# DirectTrajOpt's CommonInterface is the dispatch target for nonlinear
# constraint methods (evaluate!, eval_jacobian, eval_hessian_of_lagrangian).
import DirectTrajOpt.CommonInterface

# =============================================================================
# DRAGPathConstraint — equality constraint that enforces
#     u_Y_q(τ) + du_X_q(τ) / η_drag = 0
# at `n_samples` interior points per knot interval, for q = 1..n_qubits.
#
# Mirrors DirectTrajOpt's CubicHermitePathConstraint pattern: same Hermite
# basis evaluation, same per-interval bundling of (uₖ, duₖ, uₖ₊₁, duₖ₊₁, Δtₖ),
# same ForwardDiff-driven Jacobian/Hessian. The only differences are:
#   - equality = true (vs false for amplitude bounds)
#   - residual function returns DRAG equality residuals instead of ±bound
#   - includes the Hermite *derivative* basis to express u̇_X(τ)
# =============================================================================

@inline _h00(τ) =  2τ^3 - 3τ^2 + 1
@inline _h10(τ) =   τ^3 - 2τ^2 + τ
@inline _h01(τ) = -2τ^3 + 3τ^2
@inline _h11(τ) =   τ^3 - τ^2

# Derivative basis with respect to physical time t. With τ = (t - t_k)/Δt_k:
#   u(t)   = h00·u_k + h10·Δt·du_k + h01·u_{k+1} + h11·Δt·du_{k+1}
#   u̇(t)  = (1/Δt) d/dτ [h00·u_k + h10·Δt·du_k + h01·u_{k+1} + h11·Δt·du_{k+1}]
#         = (1/Δt) h00'·u_k + h10'·du_k + (1/Δt) h01'·u_{k+1} + h11'·du_{k+1}
# The Δt cancels from the h10·Δt·du and h11·Δt·du terms when we differentiate.
@inline _g00(τ, Δt) = ( 6τ^2 -  6τ) / Δt
@inline _g10(τ)     =   3τ^2 -  4τ + 1
@inline _g01(τ, Δt) = (-6τ^2 +  6τ) / Δt
@inline _g11(τ)     =   3τ^2 -  2τ

struct DRAGPathConstraint <: AbstractNonlinearConstraint
    u_name::Symbol
    du_name::Symbol
    Δt_name::Symbol
    η_drag::Float64
    sample_τs::Vector{Float64}
    equality::Bool
    dim::Int
    u_dim::Int
    n_qubits::Int
    n_intervals::Int
end

"""
    DRAGPathConstraint(u_name, du_name, η_drag, traj; n_samples=5, Δt_name=traj.timestep)

Build a DRAG equality constraint enforcing `u_Y_q(τ) + du_X_q(τ)/η_drag = 0`
for each qubit q ∈ 1..n_qubits at `n_samples` interior points per knot interval.

The control vector layout is assumed to be `[u_X1, u_Y1, u_X2, u_Y2, ...]`
(X/Y interleaved per qubit), so `u_dim = 2·n_qubits` and X is at index `2q-1`,
Y at index `2q`.
"""
function DRAGPathConstraint(
    u_name::Symbol,
    du_name::Symbol,
    η_drag::Float64,
    traj::NamedTrajectory;
    n_samples::Int = 5,
    Δt_name::Symbol = traj.timestep,
)
    u_dim       = traj.dims[u_name]
    @assert iseven(u_dim) "DRAGPathConstraint expects an even u_dim ([u_X1, u_Y1, u_X2, u_Y2, ...] layout)"
    n_qubits    = u_dim ÷ 2
    n_intervals = traj.N - 1
    sample_τs   = [j / (n_samples + 1) for j in 1:n_samples]
    # n_qubits equality residuals per sample × n_samples × n_intervals
    dim = n_qubits * length(sample_τs) * n_intervals
    return DRAGPathConstraint(
        u_name, du_name, Δt_name, η_drag, sample_τs, true,
        dim, u_dim, n_qubits, n_intervals,
    )
end

function Base.show(io::IO, c::DRAGPathConstraint)
    print(io, "DRAGPathConstraint: η_drag=$(c.η_drag), " *
              "$(length(c.sample_τs)) samples/interval, " *
              "$(c.n_intervals) intervals, $(c.n_qubits) qubits, dim=$(c.dim)")
end

# Residuals for one knot interval. z = [uₖ; duₖ; uₖ₊₁; duₖ₊₁; Δtₖ].
function _drag_interval_residuals(z, sample_τs, u_dim, n_qubits, η_drag)
    uₖ    = z[1:u_dim]
    duₖ   = z[(u_dim+1):(2u_dim)]
    uₖ₊₁  = z[(2u_dim+1):(3u_dim)]
    duₖ₊₁ = z[(3u_dim+1):(4u_dim)]
    Δtₖ   = z[4u_dim+1]

    n_s = length(sample_τs)
    out = Vector{eltype(z)}(undef, n_qubits * n_s)
    idx = 0
    for τ in sample_τs
        h00, h10, h01, h11 = _h00(τ), _h10(τ), _h01(τ), _h11(τ)
        g00, g10, g01, g11 = _g00(τ, Δtₖ), _g10(τ), _g01(τ, Δtₖ), _g11(τ)
        for q in 1:n_qubits
            ux_idx = 2q - 1
            uy_idx = 2q
            uY_at_τ  = h00 * uₖ[uy_idx]  + h10 * Δtₖ * duₖ[uy_idx]  +
                       h01 * uₖ₊₁[uy_idx] + h11 * Δtₖ * duₖ₊₁[uy_idx]
            duX_at_τ = g00 * uₖ[ux_idx]  + g10 * duₖ[ux_idx]  +
                       g01 * uₖ₊₁[ux_idx] + g11 * duₖ₊₁[ux_idx]
            out[idx + 1] = uY_at_τ + duX_at_τ / η_drag
            idx += 1
        end
    end
    return out
end

function CommonInterface.evaluate!(
    values::AbstractVector,
    C::DRAGPathConstraint,
    traj::NamedTrajectory,
)
    n_s     = length(C.sample_τs)
    g_k_dim = C.n_qubits * n_s

    for k in 1:C.n_intervals
        uₖ    = traj[k][C.u_name]
        duₖ   = traj[k][C.du_name]
        uₖ₊₁  = traj[k+1][C.u_name]
        duₖ₊₁ = traj[k+1][C.du_name]
        Δtₖ   = traj[k][C.Δt_name][1]

        z = vcat(uₖ, duₖ, uₖ₊₁, duₖ₊₁, [Δtₖ])
        values[slice(k, g_k_dim)] =
            _drag_interval_residuals(z, C.sample_τs, C.u_dim, C.n_qubits, C.η_drag)
    end
    return nothing
end

@views function CommonInterface.eval_jacobian(
    C::DRAGPathConstraint,
    traj::NamedTrajectory,
)
    n_s     = length(C.sample_τs)
    g_k_dim = C.n_qubits * n_s

    ∂C = spzeros(C.dim, traj.dim * traj.N + traj.global_dim)

    u_comps  = traj.components[C.u_name]
    du_comps = traj.components[C.du_name]
    Δt_comps = traj.components[C.Δt_name]

    for k in 1:C.n_intervals
        uₖ    = traj[k][C.u_name]
        duₖ   = traj[k][C.du_name]
        uₖ₊₁  = traj[k+1][C.u_name]
        duₖ₊₁ = traj[k+1][C.du_name]
        Δtₖ   = traj[k][C.Δt_name][1]

        z = vcat(uₖ, duₖ, uₖ₊₁, duₖ₊₁, [Δtₖ])

        col_ids = vcat(
            slice(k,   u_comps,  traj.dim),
            slice(k,   du_comps, traj.dim),
            slice(k+1, u_comps,  traj.dim),
            slice(k+1, du_comps, traj.dim),
            slice(k,   Δt_comps, traj.dim),
        )
        row_ids = slice(k, g_k_dim)

        ForwardDiff.jacobian!(
            ∂C[row_ids, col_ids],
            z -> _drag_interval_residuals(z, C.sample_τs, C.u_dim, C.n_qubits, C.η_drag),
            z,
        )
    end
    return ∂C
end

@views function CommonInterface.eval_hessian_of_lagrangian(
    C::DRAGPathConstraint,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    # When Δt is fixed (Δt_bounds = (Δt, Δt)), the constraint is linear in
    # (u, du) and the Hessian is identically zero. With variable Δt, the
    # Hermite-derivative basis g00/g01 ∝ 1/Δt make the constraint bilinear,
    # giving a non-zero Hessian. We compute it generically via ForwardDiff
    # so the constraint is correct under both regimes.
    n_s     = length(C.sample_τs)
    g_k_dim = C.n_qubits * n_s

    μ∂²C = spzeros(traj.dim * traj.N + traj.global_dim, traj.dim * traj.N + traj.global_dim)

    u_comps  = traj.components[C.u_name]
    du_comps = traj.components[C.du_name]
    Δt_comps = traj.components[C.Δt_name]

    for k in 1:C.n_intervals
        uₖ    = traj[k][C.u_name]
        duₖ   = traj[k][C.du_name]
        uₖ₊₁  = traj[k+1][C.u_name]
        duₖ₊₁ = traj[k+1][C.du_name]
        Δtₖ   = traj[k][C.Δt_name][1]

        z = vcat(uₖ, duₖ, uₖ₊₁, duₖ₊₁, [Δtₖ])

        col_ids = vcat(
            slice(k,   u_comps,  traj.dim),
            slice(k,   du_comps, traj.dim),
            slice(k+1, u_comps,  traj.dim),
            slice(k+1, du_comps, traj.dim),
            slice(k,   Δt_comps, traj.dim),
        )

        μₖ = μ[slice(k, g_k_dim)]

        ForwardDiff.hessian!(
            μ∂²C[col_ids, col_ids],
            z -> μₖ' * _drag_interval_residuals(
                z, C.sample_τs, C.u_dim, C.n_qubits, C.η_drag),
            z,
        )
    end
    return μ∂²C
end

# =============================================================================
# Pauli operators (2-level, 4-dim Hilbert)
# =============================================================================
XX = operator_from_string("XX")
YY = operator_from_string("YY")
XI = operator_from_string("XI")
YI = operator_from_string("YI")
IX = operator_from_string("IX")
IY = operator_from_string("IY")
IZ = operator_from_string("IZ")
ZI = operator_from_string("ZI")
ZZ = operator_from_string("ZZ")

# =============================================================================
# Physical parameters
# =============================================================================
const g_eff   = 2π * 0.002          # 2 MHz coupling
const δ₁₂     = 2π * 0.06           # 60 MHz idle detuning
const Δ_mw1   = -2π * 0.00
const Δ_mw2   = +2π * 0.00
const a_bound = 2π * 0.01           # 10 MHz amplitude bound
const drive_bounds = fill(a_bound, 4)

const σ_rise              = 1.25
const buffer_flat_duration = 5.0
const buffer_duration     = 4 * σ_rise

# DRAG correction targets the anharmonicity of the 3-level Duffing system this
# pulse will be lifted into for verification. The 2-level optimization itself
# doesn't see η — it just enforces the DRAG relation on the controls.
const η_drag = -2π * 0.170          # -170 MHz transmon anharmonicity

# Number of DRAG path samples per knot interval. The residual between samples
# decays roughly polynomially in Δt and exponentially in N_DRAG_PATH; start
# with 5 and bump up if the 3-level verification leakage is above the DRAG
# leading-order prediction.
const N_DRAG_PATH = 5

# =============================================================================
# Target
# =============================================================================
const U_iswap = exp(-im * π/4 * (XX + YY))

# =============================================================================
# Optimization knobs
# =============================================================================
const F_threshold     = 0.9999
const Q_r             = 1e2
const T_total_gate_ns = 150.0
const T_total_ns      = T_total_gate_ns - 2 * 4 * σ_rise - 2 * buffer_flat_duration
@assert T_total_ns > 0

const n_samples = 300
const N_knots   = 24
const num_iter  = 1000
const SEED      = 42

println("2-qubit (Pauli) DRAG-CONSTRAINED rollout iSWAP optimization.")
println("Hilbert dim = 4 (Pauli, 2-level)")
println("DRAG η      = $η_drag rad/ns ($(round(η_drag/(2π)*1e3, digits=1)) MHz)")
println("DRAG samples/interval = $N_DRAG_PATH")
println("σ_rise = $σ_rise ns; buffer = $buffer_flat_duration ns; mw region = $T_total_ns ns; total = $T_total_gate_ns ns")
println("N_knots = $N_knots, num_iter = $num_iter, seed = $SEED")

# =============================================================================
# Edge areas and U_goal
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

const U_goal = exp(-im * θ_goal * (XX + YY))
const V_rise = exp(-im * A_pre  * (XX + YY))
const V_fall = exp(-im * A_post * (XX + YY))

let U_check = V_fall * U_goal * V_rise
    @printf("Sanity: F(V_fall · U_goal · V_rise, iSWAP) = %.10f\n",
            abs2(tr(U_iswap' * U_check)) / 16)
end

const g_eff_MHz = round(Int, g_eff/(2π)*1e3)
const T_mw_ns   = round(Int, T_total_ns)
const η_MHz     = round(Int, abs(η_drag)/(2π)*1e3)
const run_tag   = "$(g_eff_MHz)MHz_$(T_mw_ns)nsmw_5nsgauss_5nsbuf_dragconstr_$(η_MHz)MHzeta_npath$(N_DRAG_PATH)_seed$(SEED)"
println("θ_goal  = ", round(θ_goal, digits=6), " rad")
println("run_tag = $run_tag")

# =============================================================================
# Gate-frame Hamiltonian (2-level)
# =============================================================================
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

# =============================================================================
# Add DRAG path-equality constraints + final fidelity constraint
# =============================================================================
let traj = get_trajectory(qcp)
    drag_con = DRAGPathConstraint(:u, :du, η_drag, traj; n_samples = N_DRAG_PATH)
    push!(qcp.prob.constraints, drag_con)
    println("Added DRAGPathConstraint: ", drag_con)
    println("  → $(drag_con.dim) DRAG equality residuals total " *
            "($(N_DRAG_PATH) samples/interval × $(traj.N - 1) intervals × $(drag_con.n_qubits) qubits)")
end

push!(qcp.prob.constraints,
    FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F_threshold, get_trajectory(qcp))
)

# =============================================================================
# Rollout-initialise nominal unitary trajectory.
# Project the initial random controls onto the DRAG manifold first so Ipopt
# doesn't waste early iters fighting a large initial DRAG residual.
# =============================================================================
let
    traj = get_trajectory(qcp)
    us   = traj[:u]
    dus  = traj[:du]
    # Project: set u_Y_q = -du_X_q / η at every knot. After this, the knot-level
    # DRAG relation holds exactly; path samples may still have small residual
    # which Ipopt will close to its tolerance.
    @views begin
        us[2, :]  .= .-dus[1, :] ./ η_drag    # u_Y1
        us[4, :]  .= .-dus[3, :] ./ η_drag    # u_Y2
    end

    ts = vec(traj[:t])
    nominal_sys = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
    nlp_pulse   = CubicSplinePulse(us, dus, ts)
    nlp_qtraj   = UnitaryTrajectory(nominal_sys, nlp_pulse, U_goal)

    Ũ⃗_init = hcat([operator_to_iso_vec(nlp_qtraj(t)) for t in ts]...)
    traj.data[traj.components[:Ũ⃗], :] .= Ũ⃗_init

    U_init = iso_vec_to_operator(Ũ⃗_init[:, end])
    F0 = abs2(tr(U_goal' * U_init)) / 16
    @printf("Initial fidelity (rollout init, DRAG-projected u_Y): %.6f\n", F0)
end

# =============================================================================
# Solve
# =============================================================================
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
# Results (2-level, in-plane)
# =============================================================================
traj   = get_trajectory(qcp)
U_flat = iso_vec_to_operator(traj[:Ũ⃗][:, end])
F_flat = abs2(tr(U_goal' * U_flat)) / 16
@printf("Flat-top F to U_goal: %.8f\n", F_flat)
@printf("Flat-top infidelity:  %.2e\n", 1 - F_flat)

U_full = V_fall * U_flat * V_rise
F_full = abs2(tr(U_iswap' * U_full)) / 16
@printf("Full gate F to iSWAP (2-level): %.8f\n", F_full)
@printf("Full gate infidelity:           %.2e\n", 1 - F_full)

T_total = 2 * buffer_duration + 2 * buffer_flat_duration + traj[:t][end]
φ_vz    = δ₁₂ * T_total
println("Total gate time = ", round(T_total, digits=2), " ns")
println("Virtual Z on q2 = ", round(φ_vz, digits=4), " rad")

# =============================================================================
# DRAG-residual diagnostic — how well did Ipopt satisfy the constraints?
# =============================================================================
let
    drag_con = nothing
    for c in qcp.prob.constraints
        if c isa DRAGPathConstraint
            drag_con = c; break
        end
    end
    @assert !isnothing(drag_con)
    res = zeros(drag_con.dim)
    CommonInterface.evaluate!(res, drag_con, traj)
    @printf("\n--- DRAG path-constraint residual ---\n")
    @printf("  max |residual| = %.2e   (target: ≤ constr_viol_tol = 1e-8)\n", maximum(abs, res))
    @printf("  mean|residual| = %.2e\n", sum(abs, res) / drag_con.dim)
end

# =============================================================================
# Integrator-honesty check (2-level — confirms NLP F matches independent ODE)
# =============================================================================
let
    nominal_sys_v = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
    Ũ⃗_ref = unitary_rollout(traj, nominal_sys_v;
        interpolation = :cubic_hermite, abstol = 1e-12, reltol = 1e-12)
    U_flat_ref = iso_vec_to_operator(Ũ⃗_ref[:, end])
    F_flat_ref = abs2(tr(U_goal' * U_flat_ref)) / 16

    U_full_ref = V_fall * U_flat_ref * V_rise
    F_full_ref = abs2(tr(U_iswap' * U_full_ref)) / 16

    @printf("\n--- Integrator-honesty check (high-acc independent re-rollout) ---\n")
    @printf("Flat-top F  (NLP report):            %.8f\n", F_flat)
    @printf("Flat-top F  (independent reference): %.8f\n", F_flat_ref)
    @printf("              |ΔF|:                   %.2e\n", abs(F_flat - F_flat_ref))
    @printf("Full-gate F (NLP report):            %.8f\n", F_full)
    @printf("Full-gate F (independent reference): %.8f\n", F_full_ref)
    @printf("              |ΔF|:                   %.2e\n", abs(F_full - F_full_ref))
    if abs(F_flat - F_flat_ref) > 1e-4
        @warn "NLP fidelity overstates reality by >1e-4 — tighten constr_viol_tol further."
    end
end

# =============================================================================
# 3-level verification — simulate the optimized 2-level + DRAG pulse in the
# 9-dim Duffing model and report comp-subspace F + leakage. This is the
# experimental payoff: does the path-constrained DRAG actually suppress
# leakage as theory predicts?
# =============================================================================
let
    n_lvl = 3
    b3 = zeros(ComplexF64, n_lvl, n_lvl)
    for j in 1:n_lvl-1; b3[j, j+1] = sqrt(j); end
    bd3::Matrix{ComplexF64} = Matrix(b3')
    I3::Matrix{ComplexF64}  = Matrix{ComplexF64}(I, n_lvl, n_lvl)
    I9::Matrix{ComplexF64}  = Matrix{ComplexF64}(I, n_lvl^2, n_lvl^2)
    X3::Matrix{ComplexF64} = b3 + bd3
    Y3::Matrix{ComplexF64} = im * (bd3 - b3)
    n3::Matrix{ComplexF64} = bd3 * b3
    XI_3 = kron(X3, I3); YI_3 = kron(Y3, I3)
    IX_3 = kron(I3, X3); IY_3 = kron(I3, Y3)
    XX_3 = kron(X3, X3); YY_3 = kron(Y3, Y3)
    nI_3 = kron(n3, I3); In_3 = kron(I3, n3)
    H_anh = (η_drag / 2) * (nI_3 * (nI_3 - I9) + In_3 * (In_3 - I9))
    subspace = [1, 2, 4, 5]
    U_iswap_4x4 = let
        σx::Matrix{ComplexF64} = ComplexF64[0 1; 1 0]
        σy::Matrix{ComplexF64} = ComplexF64[0 -im; im 0]
        exp(-im * π/4 * (kron(σx, σx) + kron(σy, σy)))
    end

    # Sample the optimized cubic spline finely
    ts_knots = vec(traj[:t])
    us_knots = traj[:u]; dus_knots = traj[:du]
    nlp_pulse_loc = CubicSplinePulse(us_knots, dus_knots, ts_knots)
    dt_3lvl  = 0.05
    ts_3lvl  = collect(0.0:dt_3lvl:T_total)
    mw_start = buffer_duration + buffer_flat_duration
    mw_end   = mw_start + ts_knots[end]
    function g_env(t)
        if t < buffer_duration
            return g_eff * exp(-(t - buffer_duration)^2 / (2 * σ_rise^2))
        elseif t <= mw_end + buffer_flat_duration
            return g_eff
        elseif t <= T_total
            t_post = t - (mw_end + buffer_flat_duration)
            return g_eff * exp(-t_post^2 / (2 * σ_rise^2))
        else
            return 0.0
        end
    end
    function u_at(t)
        if t < mw_start || t > mw_end
            return (0.0, 0.0, 0.0, 0.0)
        end
        samp = sample(nlp_pulse_loc, [t - mw_start])
        return (samp[1, 1], samp[2, 1], samp[3, 1], samp[4, 1])
    end
    function H_3lvl(t)
        g = g_env(t)
        uX1, uY1, uX2, uY2 = u_at(t)
        H = g * (XX_3 + YY_3) + H_anh
        c1 = cos(Δ_mw1 * t); s1 = sin(Δ_mw1 * t)
        H += uX1 * (XI_3 * c1 + YI_3 * s1)
        H += uY1 * (YI_3 * c1 - XI_3 * s1)
        c2 = cos(Δ_mw2 * t); s2 = sin(Δ_mw2 * t)
        H += uX2 * (IX_3 * c2 + IY_3 * s2)
        H += uY2 * (IY_3 * c2 - IX_3 * s2)
        return H
    end

    U = Matrix{ComplexF64}(I, n_lvl^2, n_lvl^2)
    for k in 1:length(ts_3lvl)-1
        dt_k = ts_3lvl[k+1] - ts_3lvl[k]
        H_mid = 0.5 * (H_3lvl(ts_3lvl[k]) + H_3lvl(ts_3lvl[k+1]))
        U = exp(-im * dt_k * H_mid) * U
    end
    Usub = U[subspace, subspace]
    F_3lvl = abs2(tr(U_iswap_4x4' * Usub)) / 16
    leak_3lvl = 1 - real(tr(Usub' * Usub)) / 4

    @printf("\n--- 3-level Duffing verification (η = %.1f MHz) ---\n", η_drag/(2π)*1e3)
    @printf("F(iSWAP, comp subspace): %.6f   (infidelity %.2e)\n", F_3lvl, 1 - F_3lvl)
    @printf("Leakage out of comp:     %.3f%%\n", leak_3lvl * 100)
    @printf("Leading-order DRAG prediction (Ω/η)^2 ≈ %.2e\n",
            (a_bound / abs(η_drag))^2)
end

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
    println(io, "# 2-qubit (Pauli) DRAG-constrained variational-rollout iSWAP")
    println(io, "template = VariationalRolloutProblem + DRAGPathConstraint")
    println(io, "n_levels_per_qubit = 2  (optimization domain)")
    println(io, "η_drag (constraint target) = $η_drag rad/ns ($(round(η_drag/(2π)*1e3, digits=1)) MHz)")
    println(io, "N_DRAG_PATH = $N_DRAG_PATH samples per knot interval")
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
    println(io, "F_full (vs iSWAP, 2-level) = $F_full")
end

println("\nSaved to: ", abspath(outdir))
println("Next: open in `leakage_through_time.ipynb` to plot DRAG-constrained vs")
println("the analytic-DRAG and 3-level-optimized pulses on the same axes.")
