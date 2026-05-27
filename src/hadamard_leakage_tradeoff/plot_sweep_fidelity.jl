import Pkg
Pkg.activate(@__DIR__)

using Piccolo, JLD2, LinearAlgebra, Printf, CairoMakie
using DataInterpolations: CubicHermiteSpline

const T_NS = 50.0
const a_bound = 2π * 0.01
const η_anh = -2π * 0.170
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const U_target = (1/sqrt(2)) * ComplexF64[1.0 1.0; 1.0 -1.0]

const I3 = Matrix{ComplexF64}(I, 3, 3)
const a3 = ComplexF64[0 1 0; 0 0 sqrt(2); 0 0 0]
const X3 = a3 + adjoint(a3)
const Y3 = im * (adjoint(a3) - a3)
const n3 = adjoint(a3) * a3
const H_anh3 = (η_anh / 2) * n3 * (n3 - I3)
const subspace_indices = [1, 2]

const sys_2lvl = QuantumSystem(zeros(ComplexF64, 2, 2), [σx, σy], fill(a_bound, 2))

function F_2lvl_rollout(traj)
    Ũ⃗ = unitary_rollout(traj, sys_2lvl;
        interpolation = :cubic_hermite, abstol = 1e-12, reltol = 1e-12)
    U_T = iso_vec_to_operator(Ũ⃗[:, end])
    return abs2(tr(U_target' * U_T)) / 4
end

function FL_3lvl(traj; N_fine = 2000)
    us = traj[:u]; dus = traj[:du]
    ts_knots = collect(range(0.0, T_NS, length = size(us, 2)))
    sp_X = CubicHermiteSpline(dus[1, :], us[1, :], ts_knots)
    sp_Y = CubicHermiteSpline(dus[2, :], us[2, :], ts_knots)
    ts_f = collect(range(0.0, T_NS, length = N_fine))
    dt_sub = ts_f[2] - ts_f[1]
    U3 = Matrix{ComplexF64}(I, 3, 3)
    for k in 1:N_fine - 1
        t_mid = 0.5 * (ts_f[k] + ts_f[k+1])
        H = H_anh3 + sp_X(t_mid) * X3 + sp_Y(t_mid) * Y3
        U3 = exp(-im * dt_sub * H) * U3
    end
    Usub = U3[subspace_indices, subspace_indices]
    F_3 = abs2(tr(U_target' * Usub)) / 4
    L_3 = 1 - real(tr(Usub' * Usub)) / 2
    return F_3, L_3
end

# Load all trajectories
out_dir = joinpath(@__DIR__, "spectral_sweep_results")
files = sort(filter(f -> startswith(f, "traj_") && endswith(f, ".jld2"), readdir(out_dir)))

function var_sus_from_traj(traj)
    Δt = traj.Δt[1]
    T  = traj.N
    iso_dim = traj.dims[:Ũ⃗]
    var_state_end = traj[:var_Ũ⃗][:, end]
    Ũ⃗_end  = var_state_end[1:iso_dim]
    ∂Ũ⃗_end = var_state_end[(iso_dim+1):(2*iso_dim)]
    U  = iso_vec_to_operator(Ũ⃗_end)
    ∂U = iso_vec_to_operator(∂Ũ⃗_end)
    d  = size(U, 1)
    return abs(tr((U' * ∂U)' * (U' * ∂U))) / (T * Δt)^2 / d
end

leak_bounds = Float64[]
F_2lvls     = Float64[]
F_3lvls     = Float64[]
L_3lvls     = Float64[]
var_suss    = Float64[]

for f in files
    path = joinpath(out_dir, f)
    @load path traj leak_bound
    F2 = F_2lvl_rollout(traj)
    F3, L3 = FL_3lvl(traj)
    vs = var_sus_from_traj(traj)
    push!(leak_bounds, leak_bound)
    push!(F_2lvls, F2)
    push!(F_3lvls, F3)
    push!(L_3lvls, L3)
    push!(var_suss, vs)
    @printf("ε_max=%.2e   F₂=%.6f   F₃=%.6f   L₃=%.3e   χ_z=%.4e\n",
        leak_bound, F2, F3, L3, vs)
end

# Save aggregated data
@save joinpath(out_dir, "fidelity_sweep.jld2") leak_bounds F_2lvls F_3lvls L_3lvls var_suss

# Plot — 2×2 grid: (1) F, (2) 1−F log, (3) L_3lvl, (4) var_sus
fig = Figure(size = (1500, 1000), fontsize = 16)

ax_F = Axis(fig[1, 1];
    xlabel = "LEAK_BOUND  ( |Ω̂(η)|²  constraint )",
    ylabel = "Fidelity at gate end",
    xscale = log10,
    title = "F vs LEAK_BOUND")
scatter!(ax_F, leak_bounds, F_2lvls; markersize = 14, color = :crimson, label = "F_2lvl (rollout)")
lines!(ax_F,   leak_bounds, F_2lvls; color = :crimson, linewidth = 2.0)
scatter!(ax_F, leak_bounds, F_3lvls; markersize = 14, color = :forestgreen, label = "F_3lvl (subspace)")
lines!(ax_F,   leak_bounds, F_3lvls; color = :forestgreen, linewidth = 2.0)
hlines!(ax_F, [0.9999]; color = :gray, linestyle = :dash, linewidth = 1, label = "F threshold")
axislegend(ax_F; position = :lb, labelsize = 12)

ax_inf = Axis(fig[1, 2];
    xlabel = "LEAK_BOUND  ( |Ω̂(η)|²  constraint )",
    ylabel = "1 − F",
    xscale = log10, yscale = log10,
    title = "Infidelity vs LEAK_BOUND")
scatter!(ax_inf, leak_bounds, max.(1 .- F_2lvls, 1e-12); markersize = 14, color = :crimson, label = "1 − F_2lvl")
lines!(ax_inf,   leak_bounds, max.(1 .- F_2lvls, 1e-12); color = :crimson, linewidth = 2.0)
scatter!(ax_inf, leak_bounds, max.(1 .- F_3lvls, 1e-12); markersize = 14, color = :forestgreen, label = "1 − F_3lvl")
lines!(ax_inf,   leak_bounds, max.(1 .- F_3lvls, 1e-12); color = :forestgreen, linewidth = 2.0)
axislegend(ax_inf; position = :rt, labelsize = 12)

ax_L = Axis(fig[2, 1];
    xlabel = "LEAK_BOUND  ( |Ω̂(η)|²  constraint )",
    ylabel = "L_3lvl(T) = 1 − tr(Uᵤ⁺Uᵤ)/2",
    xscale = log10, yscale = log10,
    title = "3-lvl leakage at gate end")
scatter!(ax_L, leak_bounds, max.(L_3lvls, 1e-12); markersize = 14, color = :royalblue)
lines!(ax_L,   leak_bounds, max.(L_3lvls, 1e-12); color = :royalblue, linewidth = 2.0)
lines!(ax_L, leak_bounds, 2 .* leak_bounds;
    color = :black, linestyle = :dash, linewidth = 1.5,
    label = "2·LEAK_BOUND (proxy prediction)")
axislegend(ax_L; position = :rb, labelsize = 11)

ax_S = Axis(fig[2, 2];
    xlabel = "LEAK_BOUND  ( |Ω̂(η)|²  constraint )",
    ylabel = "var_sus (σ_z susceptibility)",
    xscale = log10, yscale = log10,
    title = "Robustness vs LEAK_BOUND")
scatter!(ax_S, leak_bounds, max.(var_suss, 1e-12); markersize = 14, color = :purple)
lines!(ax_S,   leak_bounds, max.(var_suss, 1e-12); color = :purple, linewidth = 2.0)

save(joinpath(out_dir, "fidelity_leakage_robustness_vs_LEAK_BOUND.png"), fig; px_per_unit = 4)

@printf("\nDone.\n")
@printf("All plots saved to: %s\n", out_dir)
@printf("  - fidelity_leakage_robustness_vs_LEAK_BOUND.png  (this new 2x2 plot)\n")
@printf("  - pareto_var_sus_vs_LEAK_BOUND.png               (from spectral_sweep_robust.jl)\n")
@printf("Trajectories: %d files (traj_*.jld2)\n", length(files))
@printf("Aggregated data: fidelity_sweep.jld2, sweep_summary.jld2\n")
