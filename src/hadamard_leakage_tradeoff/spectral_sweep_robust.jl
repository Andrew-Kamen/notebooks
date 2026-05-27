# ============================================================================
# spectral_sweep_robust.jl
#
# Pareto sweep for the robust Hadamard with spectral leakage constraint:
# warm-start chain, sweeping LEAK_BOUND (renamed from ε_MAX in the notebooks)
# from 1.0 down to 1e-5 over 10 log-spaced points.
#
# - Same parameters as `hadamard_spectral_proxy_constraint.ipynb`
# - Only the ROBUST pulse (Q_r = Q_R_ROBUST = 1000)
# - Spectral leakage **constraint** only, NO barrier objective (R_LEAK_BARRIER = 0)
# - Warm-starts each subsequent solve from the previous trajectory's u + du
# - Saves trajectories + summary into `spectral_sweep_results/`
# - Reports variational susceptibility (var_sus, from bandwidth_no_Z.ipynb)
#   and plots var_sus vs LEAK_BOUND on log-log
#
# Run:  julia --project=. spectral_sweep_robust.jl
# ============================================================================

import Pkg
Pkg.activate(@__DIR__)
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
Pkg.develop([
    Pkg.PackageSpec(path = piccolo_path),
    Pkg.PackageSpec(path = directtrajopt_path),
])
Pkg.add(["CairoMakie", "MathTeXEngine", "LaTeXStrings", "JLD2", "Ipopt", "Statistics", "ForwardDiff"])
Pkg.instantiate()

using Piccolo
using LinearAlgebra, Random, Printf, Statistics, JLD2
using CairoMakie, MathTeXEngine, LaTeXStrings
using SparseArrays

import DirectTrajOpt.Objectives: AbstractObjective, objective_value, gradient!, hessian_structure, get_full_hessian
import DirectTrajOpt.Constraints: AbstractNonlinearConstraint
import DirectTrajOpt.CommonInterface

set_theme!(Theme(
    fonts = (;
        regular = texfont(:text), bold = texfont(:bold),
        italic = texfont(:italic), bold_italic = texfont(:bolditalic),
        ticks = "TeX Gyre Heros Makie",
    ),
    Axis = (; xgridvisible = false),
))

# ============================================================================
# Parameters — must match hadamard_spectral_proxy_constraint.ipynb
# ============================================================================
const T_NS        = 50.0
const a_bound     = 2π * 0.01            # 10 MHz drive bound
const η_anh       = -2π * 0.170          # -170 MHz anharmonicity (transmon)
const F_THRESHOLD = 0.9999
const N_KNOTS     = 15
const Δt_GATE     = T_NS / (N_KNOTS - 1)
const Q_R_ROBUST  = 1000.0
const R_DDU       = 10.0
const DDU_BOUND   = 100.0
const NUM_ITER    = 1000
const SEED        = 42

# 2-level operators
const σx = ComplexF64[0  1; 1  0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1  0; 0 -1]

const U_target = (1/sqrt(2)) * ComplexF64[1.0  1.0; 1.0 -1.0]
const MHz_per_radperns = 1e3 / (2π)

# ============================================================================
# SpectralLeakageConstraint (inline, in case Piccolo source not reloaded)
# Targets the FIRST 2 components of `:u` (= u_X, u_Y).
# Bounds |Ω̂(ω)|² ≤ leak_bound, with Ω̂(ω) = Σ_k Δt_k·(u_X[k] + i·u_Y[k])·exp(-iω·t_k)
# ============================================================================
struct SpectralLeakageConstraint <: AbstractNonlinearConstraint
    name::Symbol
    ω::Float64
    leak_bound::Float64
    times_t::Vector{Float64}
    Δts::Vector{Float64}
    cosωt::Vector{Float64}
    sinωt::Vector{Float64}
    dim::Int
    equality::Bool
end

function SpectralLeakageConstraint(name::Symbol, ω::Float64, leak_bound::Float64, traj::NamedTrajectory)
    @assert traj.dims[name] >= 2 "SpectralLeakageConstraint expects ≥ 2 components."
    times = get_times(traj)
    Δts   = get_timesteps(traj)
    return SpectralLeakageConstraint(name, ω, leak_bound,
        Vector{Float64}(times), Vector{Float64}(Δts),
        cos.(ω .* times), sin.(ω .* times),
        1, false)
end

function _ReIm(c::SpectralLeakageConstraint, u::AbstractMatrix)
    Re_sum = zero(eltype(u)); Im_sum = zero(eltype(u))
    @inbounds for k in 1:length(c.times_t)
        cos_k = c.cosωt[k]; sin_k = c.sinωt[k]; Δt_k = c.Δts[k]
        Re_sum += Δt_k * (u[1, k] * cos_k + u[2, k] * sin_k)
        Im_sum += Δt_k * (u[2, k] * cos_k - u[1, k] * sin_k)
    end
    return Re_sum, Im_sum
end

function CommonInterface.evaluate!(values::AbstractVector,
                                    c::SpectralLeakageConstraint,
                                    traj::NamedTrajectory)
    u = traj[c.name]
    Re_sum, Im_sum = _ReIm(c, u)
    values[1] = Re_sum^2 + Im_sum^2 - c.leak_bound
    return nothing
end

function CommonInterface.eval_jacobian(c::SpectralLeakageConstraint, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂g = spzeros(c.dim, Z_dim)
    u = traj[c.name]
    Re_sum, Im_sum = _ReIm(c, u)
    comps = traj.components[c.name]
    @inbounds for k in 1:traj.N
        cos_k = c.cosωt[k]; sin_k = c.sinωt[k]; Δt_k = c.Δts[k]
        idxX = traj.dim * (k - 1) + comps[1]
        idxY = traj.dim * (k - 1) + comps[2]
        ∂g[1, idxX] = 2 * Δt_k * (Re_sum * cos_k - Im_sum * sin_k)
        ∂g[1, idxY] = 2 * Δt_k * (Re_sum * sin_k + Im_sum * cos_k)
    end
    return ∂g
end

function CommonInterface.eval_hessian_of_lagrangian(c::SpectralLeakageConstraint,
                                                     traj::NamedTrajectory,
                                                     μ::AbstractVector)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²g = spzeros(Z_dim, Z_dim)
    comps = traj.components[c.name]
    a = zeros(Z_dim); b = zeros(Z_dim)
    @inbounds for k in 1:traj.N
        idxX = traj.dim * (k - 1) + comps[1]
        idxY = traj.dim * (k - 1) + comps[2]
        a[idxX] = c.Δts[k] * c.cosωt[k]
        a[idxY] = c.Δts[k] * c.sinωt[k]
        b[idxX] = -c.Δts[k] * c.sinωt[k]
        b[idxY] =  c.Δts[k] * c.cosωt[k]
    end
    nzs = findall(!iszero, a) ∪ findall(!iszero, b)
    for i in nzs, j in nzs
        v = 2 * μ[1] * (a[i]*a[j] + b[i]*b[j])
        if v != 0.0
            ∂²g[i, j] = v
        end
    end
    return ∂²g
end

println("SpectralLeakageConstraint defined.")

# ============================================================================
# Variational susceptibility (from bandwidth_no_Z.ipynb cell 31)
# var_sus = |tr((U†·∂U)†·(U†·∂U))| / (T·Δt)² / d
# ============================================================================
function var_sus(traj::NamedTrajectory)
    Δt = traj.Δt[1]
    T  = traj.N    # number of knot points
    iso_dim = traj.dims[:Ũ⃗]   # 8 for 2-lvl unitary in iso-vec form

    # VariationalSplinePulseProblem stores the augmented state as :var_Ũ⃗
    # with layout [Ũ⃗ (iso_dim) | ∂Ũ⃗_1 (iso_dim) | ∂Ũ⃗_2 | ...].
    # For 1 variational direction (σ_z), :var_Ũ⃗ has 2·iso_dim rows.
    var_state_end = traj[:var_Ũ⃗][:, end]
    Ũ⃗_end  = var_state_end[1:iso_dim]
    ∂Ũ⃗_end = var_state_end[(iso_dim+1):(2*iso_dim)]

    U  = iso_vec_to_operator(Ũ⃗_end)
    ∂U = iso_vec_to_operator(∂Ũ⃗_end)
    d  = size(U, 1)
    return abs(tr((U' * ∂U)' * (U' * ∂U))) / (T * Δt)^2 / d
end

# ============================================================================
# Single-LEAK_BOUND optimizer call, with optional warm-start (u + du)
# ============================================================================
function optimize_robust_spectral(;
    leak_bound::Float64,
    init_controls::Union{Nothing,Matrix{Float64}} = nothing,
    init_tangents::Union{Nothing,Matrix{Float64}} = nothing,
    seed::Int = SEED,
)
    Random.seed!(seed)
    drive_bounds = fill(a_bound, 2)
    H_fn      = (u, t) -> u[1] * σx + u[2] * σy
    H_vars_fn = Function[(u, t) -> σz]
    varsys = VariationalQuantumSystem(H_fn, H_vars_fn, 2, drive_bounds; time_dependent = true)

    times_knots = collect(range(0.0, T_NS, length = N_KNOTS))

    pulse = if isnothing(init_controls)
        controls_init = 0.1 .* a_bound .* randn(2, N_KNOTS)
        CubicSplinePulse(controls_init, times_knots)
    else
        @assert size(init_controls) == (2, N_KNOTS)
        if isnothing(init_tangents)
            CubicSplinePulse(init_controls, times_knots)
        else
            @assert size(init_tangents) == (2, N_KNOTS)
            CubicSplinePulse(init_controls, init_tangents, times_knots)
        end
    end

    qcp = VariationalSplinePulseProblem(
        varsys, pulse, U_target;
        Q              = 0.0,
        Q_r            = Q_R_ROBUST,
        R              = 1e-3,
        R_ddu          = R_DDU,
        du_bound       = Inf,
        ddu_bound      = DDU_BOUND,
        Δt_bounds      = (Δt_GATE, Δt_GATE),
        n_path_samples = 3,
    )
    push!(qcp.prob.constraints,
        FinalUnitaryFidelityConstraint(U_target, :Ũ⃗, F_THRESHOLD, get_trajectory(qcp)))

    # Spectral leakage constraint (no barrier objective)
    traj0 = get_trajectory(qcp)
    spec_constr = SpectralLeakageConstraint(:u, abs(η_anh), leak_bound, traj0)
    push!(qcp.prob.constraints, spec_constr)

    solve!(qcp; max_iter = NUM_ITER, print_level = 0,
        options = IpoptOptions(
            eval_hessian = false, constr_viol_tol = 1e-8,
            tol = 1e-8, acceptable_tol = 1e-8,
        ))

    return get_trajectory(qcp)
end

# ============================================================================
# The sweep
# ============================================================================
function run_sweep()
    LEAK_BOUNDs = 10.0 .^ range(0.0, -5.0, length = 10)   # 1.0 → 1e-5, log-spaced

    out_dir = joinpath(@__DIR__, "spectral_sweep_results")
    mkpath(out_dir)

    prev_u  = nothing
    prev_du = nothing

    sweep_data = NamedTuple[]

    for (i, leak_bound) in enumerate(LEAK_BOUNDs)
        @printf("\n=========================================================\n")
        @printf("[%d/%d]  LEAK_BOUND = %.3e\n", i, length(LEAK_BOUNDs), leak_bound)
        @printf("=========================================================\n")
        t0 = time()
        traj = optimize_robust_spectral(;
            leak_bound = leak_bound,
            init_controls = prev_u,
            init_tangents = prev_du,
        )
        wall = time() - t0

        vs = var_sus(traj)

        # Achieved |Ω̂(η)|² (for reference; we plot against LEAK_BOUND as requested)
        ts = collect(range(0.0, T_NS, length = N_KNOTS))
        Δt = Δt_GATE
        ω  = abs(η_anh)
        uX = traj[:u][1, :]; uY = traj[:u][2, :]
        Re_sum = sum(Δt * (uX[k]*cos(ω*ts[k]) + uY[k]*sin(ω*ts[k])) for k in 1:N_KNOTS)
        Im_sum = sum(Δt * (uY[k]*cos(ω*ts[k]) - uX[k]*sin(ω*ts[k])) for k in 1:N_KNOTS)
        achieved_omega_sq = Re_sum^2 + Im_sum^2

        @printf("  var_sus = %.4e   |Ω̂(η)|² achieved = %.4e   wall = %.1fs\n",
            vs, achieved_omega_sq, wall)

        # Save per-iteration trajectory
        @save joinpath(out_dir, @sprintf("traj_%02d_leak%.0e.jld2", i, leak_bound)) traj leak_bound vs achieved_omega_sq

        push!(sweep_data, (
            iter              = i,
            leak_bound        = leak_bound,
            var_sus           = vs,
            achieved_omega_sq = achieved_omega_sq,
            wall              = wall,
        ))

        # Warm-start for next iteration
        prev_u  = Matrix{Float64}(traj[:u])
        prev_du = haskey(traj.components, :du) ? Matrix{Float64}(traj[:du]) : nothing
    end

    # Save aggregated sweep
    leak_bounds_arr        = [d.leak_bound for d in sweep_data]
    var_sus_arr            = [d.var_sus for d in sweep_data]
    achieved_omega_sq_arr  = [d.achieved_omega_sq for d in sweep_data]
    wall_arr               = [d.wall for d in sweep_data]
    @save joinpath(out_dir, "sweep_summary.jld2") leak_bounds_arr var_sus_arr achieved_omega_sq_arr wall_arr

    return sweep_data, out_dir
end

sweep_data, out_dir = run_sweep()

# ============================================================================
# Plot:  var_sus  vs  LEAK_BOUND  (the constraint value, not the achieved one)
# ============================================================================
leak_bounds_arr = [d.leak_bound for d in sweep_data]
var_sus_arr     = [d.var_sus    for d in sweep_data]

fig = Figure(size = (1000, 650), fontsize = 18)
ax = Axis(fig[1, 1];
    xlabel = "LEAK_BOUND  ( |Ω̂(η)|²  constraint )",
    ylabel = "Variational susceptibility",
    xscale = log10, yscale = log10,
    title = "Robust Hadamard: var_sus vs spectral-leakage constraint")

scatter!(ax, leak_bounds_arr, var_sus_arr;
    markersize = 14, color = :crimson)
lines!(ax, leak_bounds_arr, var_sus_arr;
    color = :crimson, linewidth = 2.0)

# Annotate each point with iteration index
for (i, (x, y)) in enumerate(zip(leak_bounds_arr, var_sus_arr))
    text!(ax, x, y; text = string(i),
        offset = (8, 8), fontsize = 11, color = :black)
end

save(joinpath(out_dir, "pareto_var_sus_vs_LEAK_BOUND.png"), fig; px_per_unit = 4)
display(fig)

@printf("\n=== Sweep complete ===\n")
@printf("Output dir: %s\n", out_dir)
@printf("Files: sweep_summary.jld2, traj_*.jld2, pareto_var_sus_vs_LEAK_BOUND.png\n")
