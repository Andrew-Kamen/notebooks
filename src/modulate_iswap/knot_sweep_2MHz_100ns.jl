# =============================================================================
# Knot-count sweep for 100 ns flat-top iSwap (2 MHz coupling, gate frame)
#
# Identical optimization setup to robust_iswap_detuned_2MHz_100ns.ipynb;
# sweeps N_knots ∈ N_KNOTS_LIST × N_SEEDS random seeds. For each (N, seed):
#   - solves VariationalSplinePulseProblem (Q_r robustness on ZI, IZ, ZZ)
#   - records runtime, final fidelity, and per-error variational objective
#   - saves controls (knots, fine sample, JLD2 trajectory) + susceptibility CSV
#
# After all runs:
#   - per-seed plots of var_obj vs N_knots and runtime vs N_knots
#   - aggregated plots with mean ± std shading across seeds
#   - summary.csv with one row per (N_knots, seed)
# =============================================================================

import Pkg
Pkg.activate(@__DIR__)
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
# Develop both local packages in a SINGLE Pkg call. Calling Pkg.develop
# twice (once per package) triggers two separate resolves; the second one
# can downgrade packages and invalidate artifacts the first compiled,
# producing "Package X required but does not seem to be installed" errors.
# A single combined call also overwrites both stale dev paths in the same
# resolve, which is what makes this work cleanly across machines.
Pkg.develop([
    Pkg.PackageSpec(path = piccolo_path),
    Pkg.PackageSpec(path = directtrajopt_path),
])
Pkg.instantiate()

using Piccolo
using LinearAlgebra
using Random
using Printf
using CairoMakie
using JLD2
using Statistics

# =============================================================================
# Sweep hyperparameters
# =============================================================================
const N_KNOTS_LIST = [4, 8, 12, 16, 20, 24]
const N_SEEDS      = 5
const SEED_BASE    = 42

const OUTDIR = joinpath(@__DIR__, "knot_sweep_2MHz_100ns")
mkpath(OUTDIR)
mkpath(joinpath(OUTDIR, "summary"))

println("N_KNOTS_LIST = ", N_KNOTS_LIST)
println("N_SEEDS      = ", N_SEEDS, "  (seeds: $(SEED_BASE):$(SEED_BASE + N_SEEDS - 1))")
println("OUTDIR       = ", OUTDIR)

# =============================================================================
# Physics (identical to robust_iswap_detuned_2MHz_100ns)
# =============================================================================
XX = operator_from_string("XX"); YY = operator_from_string("YY")
XY = operator_from_string("XY"); YX = operator_from_string("YX")
XI = operator_from_string("XI"); YI = operator_from_string("YI")
IX = operator_from_string("IX"); IY = operator_from_string("IY")
IZ = operator_from_string("IZ"); ZI = operator_from_string("ZI")
ZZ = operator_from_string("ZZ")

const g_eff = 2π * 0.002
const δ₁₂   = 2π * 0.06
const Δ_mw1 = -2π * 0.00
const Δ_mw2 = +2π * 0.00

const a_bound      = 2π * 0.01
const drive_bounds = fill(a_bound, 4)
const σ_rise       = 1.0

const F_threshold = 0.9999
const Q_r         = 1e2
const T_total_ns  = 100.0
const n_samples   = 300
const num_iter    = 1000

# Gaussian rise/fall edge areas → flat-top θ_goal
const buffer_duration = 4 * σ_rise
const dt_fine_edge    = 0.01
let
    t_rise = collect(0:dt_fine_edge:buffer_duration)
    g_rise = g_eff .* exp.(-(t_rise .- buffer_duration).^2 ./ (2 * σ_rise^2))
    t_fall = collect(0:dt_fine_edge:buffer_duration)
    g_fall = g_eff .* exp.(-(t_fall).^2 ./ (2 * σ_rise^2))
    global A_rise = sum(g_rise) * dt_fine_edge
    global A_fall = sum(g_fall) * dt_fine_edge
end
const θ_goal = π/4 - A_rise - A_fall
@assert θ_goal > 0 "Edges over-rotate past iSWAP! Reduce g_eff or σ_rise."

const U_iswap = exp(-im * π/4 * (XX + YY))
const U_goal  = exp(-im * θ_goal * (XX + YY))
const V_rise  = exp(-im * A_rise * (XX + YY))
const V_fall  = exp(-im * A_fall * (XX + YY))

println("θ_goal = ", round(θ_goal, digits=4), " rad")

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
const ERR_NAMES = ["ZI", "IZ", "ZZ"]

const varsys = VariationalQuantumSystem(
    H_gate_frame, H_vars, 4, drive_bounds;
    time_dependent = true,
)

# =============================================================================
# Helpers
# =============================================================================

# Variational objective per error, computed from the optimized trajectory's
# stored ∂Ũ⃗ blocks (assumes variational_scales = 1.0):
#   var_obj_i = |tr((U†·∂Uᵢ)†(U†·∂Uᵢ))| / ((T-1)·Δt)² / d
function var_obj_per_error(traj)
    iso_dim   = traj.dims[:Ũ⃗]
    var_block = traj[:var_Ũ⃗]
    T_knots   = size(var_block, 2)
    Δt        = traj.Δt[1]
    d         = varsys.levels

    U = iso_vec_to_operator(var_block[1:iso_dim, end])
    out = Float64[]
    for i in 1:length(varsys.G_vars)
        rng = (i*iso_dim + 1):((i+1)*iso_dim)
        ∂U  = iso_vec_to_operator(var_block[rng, end])
        push!(out, abs(tr((U'*∂U)' * (U'*∂U))) / ((T_knots - 1) * Δt)^2 / d)
    end
    return out
end

# One optimization run for given (N_knots, seed)
function run_single(N_knots::Int, seed::Int, log_path::String)
    Δt = T_total_ns / N_knots

    Random.seed!(seed)
    controls = 2 .* a_bound .* rand(4, n_samples) .- a_bound
    times    = collect(LinRange(0.0, T_total_ns, n_samples))
    du_init  = zeros(4, n_samples)
    pulse    = CubicSplinePulse(controls, du_init, times)

    qcp = VariationalSplinePulseProblem(
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
            verbose             = false,
        ),
    )

    push!(qcp.prob.constraints,
        FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F_threshold, get_trajectory(qcp))
    )

    # Rollout-init :Ũ⃗ and the nominal block of :var_Ũ⃗
    let
        traj = get_trajectory(qcp)
        us   = traj[:u]; dus = traj[:du]; ts = vec(traj[:t])
        iso_dim     = traj.dims[:Ũ⃗]
        nominal_sys = QuantumSystem(H_gate_frame, drive_bounds; time_dependent = true)
        nlp_pulse   = CubicSplinePulse(us, dus, ts)
        nlp_qtraj   = UnitaryTrajectory(nominal_sys, nlp_pulse, U_goal)
        Ũ⃗_init     = hcat([operator_to_iso_vec(nlp_qtraj(t)) for t in ts]...)
        traj.data[traj.components[:Ũ⃗], :] .= Ũ⃗_init
        rows_var = traj.components[:var_Ũ⃗]
        traj.data[rows_var[1:iso_dim], :] .= Ũ⃗_init
    end

    runtime = @elapsed solve!(
        qcp;
        max_iter    = num_iter,
        print_level = 5,
        options     = IpoptOptions(eval_hessian = false, output_file = log_path),
    )

    traj   = get_trajectory(qcp)
    U_flat = iso_vec_to_operator(traj[:Ũ⃗][:, end])
    F_flat = abs2(tr(U_goal' * U_flat)) / varsys.levels^2
    U_full = V_fall * U_flat * V_rise
    F_full = abs2(tr(U_iswap' * U_full)) / 4^2

    obj_per_err = var_obj_per_error(traj)
    converged   = F_flat ≥ F_threshold - 1e-6

    return (
        traj      = traj,
        runtime   = runtime,
        F_flat    = F_flat,
        F_full    = F_full,
        var_obj   = obj_per_err,            # [ZI, IZ, ZZ]
        var_obj_sum = sum(obj_per_err),
        converged = converged,
    )
end

# Save controls + per-run artifacts
function save_run(run_outdir::String, N_knots::Int, seed::Int, r::NamedTuple)
    mkpath(run_outdir)

    # Full trajectory for later simulations
    @save joinpath(run_outdir, "trajectory.jld2") traj=r.traj

    ts_k  = vec(r.traj[:t])
    us_k  = r.traj[:u]
    dus_k = r.traj[:du]

    # Knot-level controls (with Hermite tangents, sufficient to rebuild the spline)
    open(joinpath(run_outdir, "controls_knots.csv"), "w") do io
        println(io, "time_ns,u_X1,u_Y1,u_X2,u_Y2,du_X1,du_Y1,du_X2,du_Y2")
        for j in 1:length(ts_k)
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                ts_k[j],
                us_k[1,j],  us_k[2,j],  us_k[3,j],  us_k[4,j],
                dus_k[1,j], dus_k[2,j], dus_k[3,j], dus_k[4,j])
        end
    end

    # Fine-sampled controls (2000 points) for plotting / external simulators
    pulse_fine = CubicSplinePulse(us_k, dus_k, ts_k)
    ts_fine    = collect(LinRange(ts_k[1], ts_k[end], 2000))
    us_fine    = sample(pulse_fine, ts_fine)
    open(joinpath(run_outdir, "controls_fine.csv"), "w") do io
        println(io, "time_ns,u_X1,u_Y1,u_X2,u_Y2")
        for j in 1:length(ts_fine)
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g\n",
                ts_fine[j],
                us_fine[1,j], us_fine[2,j], us_fine[3,j], us_fine[4,j])
        end
    end

    # Susceptibility values (per-error variational objective)
    open(joinpath(run_outdir, "susceptibility.csv"), "w") do io
        println(io, "error,var_obj")
        for (name, v) in zip(ERR_NAMES, r.var_obj)
            @printf(io, "%s,%.10g\n", name, v)
        end
        @printf(io, "sum,%.10g\n", r.var_obj_sum)
    end

    # Plain-text run summary
    open(joinpath(run_outdir, "run_summary.txt"), "w") do io
        println(io, "N_knots   = $N_knots")
        println(io, "seed      = $seed")
        @printf(io, "runtime_s = %.4f\n",  r.runtime)
        @printf(io, "F_flat    = %.10f\n", r.F_flat)
        @printf(io, "F_full    = %.10f\n", r.F_full)
        for (name, v) in zip(ERR_NAMES, r.var_obj)
            @printf(io, "var_obj_%s = %.6g\n", name, v)
        end
        @printf(io, "var_obj_sum = %.6g\n", r.var_obj_sum)
        println(io, "converged = $(r.converged)")
    end
end

# =============================================================================
# Main sweep
# =============================================================================
const SEEDS = collect(SEED_BASE:(SEED_BASE + N_SEEDS - 1))

# results[(N_knots, seed)] = NamedTuple from run_single
results = Dict{Tuple{Int,Int}, NamedTuple}()

for N_knots in N_KNOTS_LIST
    for seed in SEEDS
        run_outdir = joinpath(OUTDIR,
            "N_knots_$(lpad(N_knots, 2, '0'))",
            "seed_$(seed)")
        mkpath(run_outdir)
        log_path = joinpath(run_outdir, "ipopt.log")

        @printf("\n=== N_knots = %2d, seed = %d ===\n", N_knots, seed)
        r = run_single(N_knots, seed, log_path)
        results[(N_knots, seed)] = r
        save_run(run_outdir, N_knots, seed, r)

        @printf("    runtime = %.2fs | F_flat = %.6f | var_obj_sum = %.4g | converged = %s\n",
            r.runtime, r.F_flat, r.var_obj_sum, r.converged)
    end
end

println("\n=== Sweep complete ===")

# =============================================================================
# Summary CSV (one row per (N_knots, seed))
# =============================================================================
open(joinpath(OUTDIR, "summary", "summary.csv"), "w") do io
    println(io, "N_knots,seed,runtime_s,F_flat,F_full,var_obj_ZI,var_obj_IZ,var_obj_ZZ,var_obj_sum,converged")
    for N in N_KNOTS_LIST, s in SEEDS
        r = results[(N, s)]
        @printf(io, "%d,%d,%.4f,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n",
            N, s, r.runtime, r.F_flat, r.F_full,
            r.var_obj[1], r.var_obj[2], r.var_obj[3], r.var_obj_sum,
            r.converged)
    end
end

# =============================================================================
# Per-seed plots
# =============================================================================
const SEED_COLORS = Makie.wong_colors()

# var_obj per seed
let fig = Figure(fontsize=22, size=(900, 500))
    ax = Axis(fig[1,1];
        xlabel = "N_knots",
        ylabel = "var_obj (sum over ZI, IZ, ZZ)",
        yscale = log10,
        title  = "Variational objective vs N_knots — per seed")
    for (i, seed) in enumerate(SEEDS)
        ys = [results[(N, seed)].var_obj_sum for N in N_KNOTS_LIST]
        lines!(ax, N_KNOTS_LIST, ys; color=SEED_COLORS[i], linewidth=2, label="seed $seed")
        scatter!(ax, N_KNOTS_LIST, ys; color=SEED_COLORS[i], markersize=10)
    end
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "summary", "var_obj_per_seed.png"), fig)
    display(fig)
end

# runtime per seed
let fig = Figure(fontsize=22, size=(900, 500))
    ax = Axis(fig[1,1];
        xlabel = "N_knots",
        ylabel = "Runtime (s)",
        title  = "Runtime vs N_knots — per seed")
    for (i, seed) in enumerate(SEEDS)
        ys = [results[(N, seed)].runtime for N in N_KNOTS_LIST]
        lines!(ax, N_KNOTS_LIST, ys; color=SEED_COLORS[i], linewidth=2, label="seed $seed")
        scatter!(ax, N_KNOTS_LIST, ys; color=SEED_COLORS[i], markersize=10)
    end
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "summary", "runtime_per_seed.png"), fig)
    display(fig)
end

# =============================================================================
# Aggregate (mean ± std) plots
# =============================================================================
function aggregate_metric(metric_fn)
    means = Float64[]; stds = Float64[]
    for N in N_KNOTS_LIST
        vals = [metric_fn(results[(N, s)]) for s in SEEDS]
        push!(means, mean(vals))
        push!(stds,  std(vals))
    end
    return means, stds
end

# var_obj sum: mean ± std band, with errorbars
let fig = Figure(fontsize=22, size=(900, 500))
    ax = Axis(fig[1,1];
        xlabel = "N_knots",
        ylabel = "var_obj (sum over ZI, IZ, ZZ)",
        yscale = log10,
        title  = "Variational objective vs N_knots (mean ± std, $N_SEEDS seeds)")
    m, s = aggregate_metric(r -> r.var_obj_sum)
    band!(ax, N_KNOTS_LIST, max.(m .- s, 1e-16), m .+ s; color=(:dodgerblue, 0.25))
    lines!(ax, N_KNOTS_LIST, m; color=:dodgerblue, linewidth=2)
    scatter!(ax, N_KNOTS_LIST, m; color=:dodgerblue, markersize=12)
    errorbars!(ax, N_KNOTS_LIST, m, s; whiskerwidth=8, color=:dodgerblue)
    save(joinpath(OUTDIR, "summary", "var_obj_vs_knots.png"), fig)
    display(fig)
end

# Per-error breakdown (ZI, IZ, ZZ on one plot, mean ± std)
let fig = Figure(fontsize=22, size=(900, 500))
    ax = Axis(fig[1,1];
        xlabel = "N_knots",
        ylabel = "var_obj (per error)",
        yscale = log10,
        title  = "Per-error variational objective ($N_SEEDS seeds)")
    err_colors = [:crimson, :forestgreen, :purple]
    for (idx, (name, col)) in enumerate(zip(ERR_NAMES, err_colors))
        m, s = aggregate_metric(r -> r.var_obj[idx])
        band!(ax, N_KNOTS_LIST, max.(m .- s, 1e-16), m .+ s; color=(col, 0.2))
        lines!(ax, N_KNOTS_LIST, m; color=col, linewidth=2, label=name)
        scatter!(ax, N_KNOTS_LIST, m; color=col, markersize=10)
        errorbars!(ax, N_KNOTS_LIST, m, s; whiskerwidth=6, color=col)
    end
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "summary", "var_obj_per_error.png"), fig)
    display(fig)
end

# Runtime: mean ± std
let fig = Figure(fontsize=22, size=(900, 500))
    ax = Axis(fig[1,1];
        xlabel = "N_knots",
        ylabel = "Runtime (s)",
        title  = "Runtime vs N_knots (mean ± std, $N_SEEDS seeds)")
    m, s = aggregate_metric(r -> r.runtime)
    band!(ax, N_KNOTS_LIST, m .- s, m .+ s; color=(:purple, 0.25))
    lines!(ax, N_KNOTS_LIST, m; color=:purple, linewidth=2)
    scatter!(ax, N_KNOTS_LIST, m; color=:purple, markersize=12)
    errorbars!(ax, N_KNOTS_LIST, m, s; whiskerwidth=8, color=:purple)
    save(joinpath(OUTDIR, "summary", "runtime_vs_knots.png"), fig)
    display(fig)
end

println("\nSummary written to $(joinpath(OUTDIR, "summary"))")
