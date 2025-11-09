#!/usr/bin/env julia

# Parameter Sweep Analysis: Robustness vs Constraints
# This script performs two parameter sweeps and analyzes susceptibility
# with multiple seeds per configuration

import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate();
using PiccoloQuantumObjects
using QuantumCollocation
using LinearAlgebra
using NamedTrajectories
using Statistics
using CairoMakie
using Random
using Printf
using Dates
using JLD2

# ============================================================================
# Configuration
# ============================================================================

# Problem parameters
T      = 40
Δt     = 0.2
Δt_min = Δt
Δt_max = Δt
F = 0.9999
U_goal = GATES.H
H_drive = [PAULIS.X, PAULIS.Y, PAULIS.Z]
piccolo_opts = PiccoloOptions(verbose=false)

# System setup
sys = QuantumSystem(H_drive)
∂ₑHₑ = [PAULIS.X, PAULIS.Y, PAULIS.Z]
varsys = VariationalQuantumSystem(H_drive, ∂ₑHₑ)

# Sweep parameters
n_points = 25
seed_values = [42, 123, 456, 789, 1011, 2002, 2025, 3002, 3025, 9876]
n_seeds = length(seed_values)

# Sweep 1: vary dda_bound from -0.2 to 1, fix a_bound = 2.5
a_bound_fixed = 4.0
dda_bounds = 10 .^range(-0.2, 1, length=n_points)

# Sweep 2: vary a_bound from -0.2 to 1, fix dda_bound = 2.5
dda_bound_fixed = 4.0
a_bounds = 10 .^range(-0.2, 1, length=n_points)

# ============================================================================
# Utility Functions
# ============================================================================

function var_obj(traj::NamedTrajectory, H_drive::Vector{Matrix{ComplexF64}}, 
                 error_op::Matrix{ComplexF64})
    """Calculate variational objective (susceptibility measure)"""
    varsys_local = VariationalQuantumSystem(H_drive, [error_op])
    T = traj.T
    Δt = traj.Δt[1]
    ww = iso_vec_to_operator(variational_unitary_rollout(traj, varsys_local)[2][1][:,end])
    d = size(ww)[1]
    return norm(tr(ww'ww)) / (T * Δt)^2 / d
end

function calculate_susceptibility(traj::NamedTrajectory, H_drive::Vector{Matrix{ComplexF64}})
    """Calculate average susceptibility over all Pauli errors"""
    x_err = var_obj(traj, H_drive, PAULIS.X)
    y_err = var_obj(traj, H_drive, PAULIS.Y)
    z_err = var_obj(traj, H_drive, PAULIS.Z)
    return (x_err + y_err + z_err) / 3
end

function optimize_trajectory(a_bound::Float64, dda_bound::Float64, 
                            seed::Int64)
    """Run optimization with given parameters"""
    Random.seed!(seed)

    tog_prob = UnitaryToggleProblem(
        varsys, U_goal, T, Δt;
        a_bound=a_bound,
        dda_bound=dda_bound,
        Δt_max=Δt,
        Δt_min=Δt,
        Q=0.0,
        Q_t = 1.0,
        piccolo_options = PiccoloOptions(verbose=false)
    )
    push!(
        tog_prob.constraints, 
        FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, tog_prob.trajectory)
    );

    # Solve with two-stage optimization
    solve!(tog_prob, max_iter=2500, print_level=0, options=IpoptOptions(eval_hessian=false))
    
    return tog_prob
end

# ============================================================================
# Sweep 1: Vary dda_bound, fix a_bound (with multiple seeds)
# ============================================================================

println("="^80)
println("SWEEP 1: Varying dda_bound from -0.2 to 1 (a_bound = $a_bound_fixed)")
println("Using $n_seeds seeds: $seed_values")
println("="^80)

# Store results for all seeds
trajectories_dda = Array{Any}(undef, n_points, n_seeds)
susceptibilities_dda = zeros(n_points, n_seeds)
objective_values_dda = zeros(n_points, n_seeds)
fidelities_dda = zeros(n_points, n_seeds)

for (i, dda) in enumerate(dda_bounds)
    println("\n[$i/$n_points] dda_bound = $(Printf.@sprintf("%.3f", dda))")
    
    for (j, seed) in enumerate(seed_values)
        print("  Seed $seed ($j/$n_seeds)... ")
        
        prob = optimize_trajectory(a_bound_fixed, dda, seed)
        traj = prob.trajectory
        
        trajectories_dda[i, j] = traj
        susceptibilities_dda[i, j] = calculate_susceptibility(traj, H_drive)
        objective_values_dda[i, j] = prob.objective.L(vec(traj))
        fidelities_dda[i, j] = unitary_rollout_fidelity(traj, sys)
        
        println("Done")
    end
    
    # Report statistics across seeds
    println("  Susceptibility: $(Printf.@sprintf("%.6e", mean(susceptibilities_dda[i, :]))) ± $(Printf.@sprintf("%.6e", std(susceptibilities_dda[i, :])))")
    println("  Objective: $(Printf.@sprintf("%.6e", mean(objective_values_dda[i, :]))) ± $(Printf.@sprintf("%.6e", std(objective_values_dda[i, :])))")
    println("  Fidelity: $(Printf.@sprintf("%.6f", mean(fidelities_dda[i, :]))) ± $(Printf.@sprintf("%.6f", std(fidelities_dda[i, :])))")
end

# Compute statistics across seeds
susceptibilities_dda_mean = mean(susceptibilities_dda, dims=2)[:, 1]
susceptibilities_dda_std = std(susceptibilities_dda, dims=2)[:, 1]
objective_values_dda_mean = mean(objective_values_dda, dims=2)[:, 1]
objective_values_dda_std = std(objective_values_dda, dims=2)[:, 1]
fidelities_dda_mean = mean(fidelities_dda, dims=2)[:, 1]
fidelities_dda_std = std(fidelities_dda, dims=2)[:, 1]

# ============================================================================
# Sweep 2: Vary a_bound, fix dda_bound (with multiple seeds)
# ============================================================================

println("\n" * "="^80)
println("SWEEP 2: Varying a_bound from -0.2 to 1 (dda_bound = $dda_bound_fixed)")
println("Using $n_seeds seeds: $seed_values")
println("="^80)

# Store results for all seeds
trajectories_a = Array{Any}(undef, n_points, n_seeds)
susceptibilities_a = zeros(n_points, n_seeds)
objective_values_a = zeros(n_points, n_seeds)
fidelities_a = zeros(n_points, n_seeds)

for (i, a) in enumerate(a_bounds)
    println("\n[$i/$n_points] a_bound = $(Printf.@sprintf("%.3f", a))")
    
    for (j, seed) in enumerate(seed_values)
        print("  Seed $seed ($j/$n_seeds)... ")
        
        prob = optimize_trajectory(a, dda_bound_fixed, seed)
        traj = prob.trajectory
        
        trajectories_a[i, j] = traj
        susceptibilities_a[i, j] = calculate_susceptibility(traj, H_drive)
        objective_values_a[i, j] = prob.objective.L(vec(traj))
        fidelities_a[i, j] = unitary_rollout_fidelity(traj, sys)
        
        println("Done")
    end
    
    # Report statistics across seeds
    println("  Susceptibility: $(Printf.@sprintf("%.6e", mean(susceptibilities_a[i, :]))) ± $(Printf.@sprintf("%.6e", std(susceptibilities_a[i, :])))")
    println("  Objective: $(Printf.@sprintf("%.6e", mean(objective_values_a[i, :]))) ± $(Printf.@sprintf("%.6e", std(objective_values_a[i, :])))")
    println("  Fidelity: $(Printf.@sprintf("%.6f", mean(fidelities_a[i, :]))) ± $(Printf.@sprintf("%.6f", std(fidelities_a[i, :])))")
end

# Compute statistics across seeds
susceptibilities_a_mean = mean(susceptibilities_a, dims=2)[:, 1]
susceptibilities_a_std = std(susceptibilities_a, dims=2)[:, 1]
objective_values_a_mean = mean(objective_values_a, dims=2)[:, 1]
objective_values_a_std = std(objective_values_a, dims=2)[:, 1]
fidelities_a_mean = mean(fidelities_a, dims=2)[:, 1]
fidelities_a_std = std(fidelities_a, dims=2)[:, 1]

# ============================================================================
# Create Visualizations
# ============================================================================

println("\n" * "="^80)
println("Creating visualizations...")
println("="^80)

fig = Figure(size=(1800, 2400))

# Sweep 1: dda_bound plots
Label(fig[1, 1:3], "Sweep 1: Varying dda_bound (a_bound = $a_bound_fixed)", 
      fontsize=20, font="bold")

# Plot 1.1: Susceptibility vs dda_bound
ax1_1 = Axis(fig[2, 1],
    title="(Adjoint) Susceptibility vs dda_bound",
    xlabel="dda_bound",
    ylabel="Susceptibility (var_obj)"
)
lines!(ax1_1, collect(dda_bounds), susceptibilities_dda_mean, 
       color=:blue, linewidth=2, label="Mean Susceptibility")
band!(ax1_1, collect(dda_bounds), 
      susceptibilities_dda_mean .- susceptibilities_dda_std,
      susceptibilities_dda_mean .+ susceptibilities_dda_std,
      color=(:blue, 0.3), label="±1 std")
scatter!(ax1_1, collect(dda_bounds), susceptibilities_dda_mean, 
         color=:blue, markersize=8)
axislegend(ax1_1, position=:rt)

# Plot 1.2: Objective value vs dda_bound
ax1_2 = Axis(fig[2, 2],
    title="Objective Value vs dda_bound",
    xlabel="dda_bound",
    ylabel="Objective Value"
)
lines!(ax1_2, collect(dda_bounds), objective_values_dda_mean, 
       color=:red, linewidth=2, label="Mean Objective")
band!(ax1_2, collect(dda_bounds),
      objective_values_dda_mean .- objective_values_dda_std,
      objective_values_dda_mean .+ objective_values_dda_std,
      color=(:red, 0.3), label="±1 std")
scatter!(ax1_2, collect(dda_bounds), objective_values_dda_mean, 
         color=:red, markersize=8)
axislegend(ax1_2, position=:rt)

# Plot 1.3: Combined plot
ax1_3 = Axis(fig[2, 3],
    title="Combined: Susceptibility & Objective vs dda_bound",
    xlabel="dda_bound",
    ylabel="Value (normalized)"
)
# Normalize for comparison
susc_norm_dda = susceptibilities_dda_mean ./ maximum(susceptibilities_dda_mean)
obj_norm_dda = objective_values_dda_mean ./ maximum(objective_values_dda_mean)
lines!(ax1_3, collect(dda_bounds), susc_norm_dda, 
       color=:blue, linewidth=2, label="Susceptibility (norm)")
lines!(ax1_3, collect(dda_bounds), obj_norm_dda, 
       color=:red, linewidth=2, label="Objective (norm)")
scatter!(ax1_3, collect(dda_bounds), susc_norm_dda, color=:blue, markersize=6)
scatter!(ax1_3, collect(dda_bounds), obj_norm_dda, color=:red, markersize=6)
axislegend(ax1_3, position=:rt)

# Sweep 2: a_bound plots
Label(fig[3, 1:3], "Sweep 2: Varying a_bound (dda_bound = $dda_bound_fixed)", 
      fontsize=20, font="bold")

# Plot 2.1: Susceptibility vs a_bound
ax2_1 = Axis(fig[4, 1],
    title="Susceptibility vs a_bound",
    xlabel="a_bound",
    ylabel="Susceptibility (var_obj)"
)
lines!(ax2_1, collect(a_bounds), susceptibilities_a_mean, 
       color=:green, linewidth=2, label="Mean Susceptibility")
band!(ax2_1, collect(a_bounds),
      susceptibilities_a_mean .- susceptibilities_a_std,
      susceptibilities_a_mean .+ susceptibilities_a_std,
      color=(:green, 0.3), label="±1 std")
scatter!(ax2_1, collect(a_bounds), susceptibilities_a_mean, 
         color=:green, markersize=8)
axislegend(ax2_1, position=:rt)

# Plot 2.2: Objective value vs a_bound
ax2_2 = Axis(fig[4, 2],
    title="Objective Value vs a_bound",
    xlabel="a_bound",
    ylabel="Objective Value"
)
lines!(ax2_2, collect(a_bounds), objective_values_a_mean, 
       color=:purple, linewidth=2, label="Mean Objective")
band!(ax2_2, collect(a_bounds),
      objective_values_a_mean .- objective_values_a_std,
      objective_values_a_mean .+ objective_values_a_std,
      color=(:purple, 0.3), label="±1 std")
scatter!(ax2_2, collect(a_bounds), objective_values_a_mean, 
         color=:purple, markersize=8)
axislegend(ax2_2, position=:rt)

# Plot 2.3: Combined plot
ax2_3 = Axis(fig[4, 3],
    title="Combined: Susceptibility & Objective vs a_bound",
    xlabel="a_bound",
    ylabel="Value (normalized)"
)
# Normalize for comparison
susc_norm_a = susceptibilities_a_mean ./ maximum(susceptibilities_a_mean)
obj_norm_a = objective_values_a_mean ./ maximum(objective_values_a_mean)
lines!(ax2_3, collect(a_bounds), susc_norm_a, 
       color=:green, linewidth=2, label="Susceptibility (norm)")
lines!(ax2_3, collect(a_bounds), obj_norm_a, 
       color=:purple, linewidth=2, label="Objective (norm)")
scatter!(ax2_3, collect(a_bounds), susc_norm_a, color=:green, markersize=6)
scatter!(ax2_3, collect(a_bounds), obj_norm_a, color=:purple, markersize=6)
axislegend(ax2_3, position=:rt)

# Summary statistics
Label(fig[5, 1:3], "Summary Statistics", fontsize=18, font="bold")

ax_summary = Axis(fig[6, 1:3], title="")
hidedecorations!(ax_summary)
hidespines!(ax_summary)

summary_text = """
Number of seeds per configuration: $n_seeds
Seeds used: $seed_values

Sweep 1 (dda_bound variation):
  Susceptibility range: $(Printf.@sprintf("%.3e", minimum(susceptibilities_dda_mean))) to $(Printf.@sprintf("%.3e", maximum(susceptibilities_dda_mean)))
  Objective range: $(Printf.@sprintf("%.3e", minimum(objective_values_dda_mean))) to $(Printf.@sprintf("%.3e", maximum(objective_values_dda_mean)))
  Mean fidelity: $(Printf.@sprintf("%.6f", mean(fidelities_dda_mean))) ± $(Printf.@sprintf("%.6f", mean(fidelities_dda_std)))
  
Sweep 2 (a_bound variation):
  Susceptibility range: $(Printf.@sprintf("%.3e", minimum(susceptibilities_a_mean))) to $(Printf.@sprintf("%.3e", maximum(susceptibilities_a_mean)))
  Objective range: $(Printf.@sprintf("%.3e", minimum(objective_values_a_mean))) to $(Printf.@sprintf("%.3e", maximum(objective_values_a_mean)))
  Mean fidelity: $(Printf.@sprintf("%.6f", mean(fidelities_a_mean))) ± $(Printf.@sprintf("%.6f", mean(fidelities_a_std)))
"""

text!(ax_summary, 0.1, 0.5, text=summary_text,
      space=:relative, fontsize=14, font="monospace")

# ============================================================================
# Save Results
# ============================================================================

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
folder_name = "tog_sus_dda_sweep_$(timestamp)"
save_path = joinpath("artifacts", folder_name)
mkpath(save_path)

println("\nSaving results to: $save_path")

# Save figure
save(joinpath(save_path, "analysis_plots.png"), fig, px_per_unit=2)
println("✓ Saved plots")

# Save trajectories and data
@save joinpath(save_path, "tog_sweep1_dda.jld2") trajectories_dda susceptibilities_dda objective_values_dda fidelities_dda susceptibilities_dda_mean susceptibilities_dda_std objective_values_dda_mean objective_values_dda_std fidelities_dda_mean fidelities_dda_std dda_bounds a_bound_fixed seed_values
@save joinpath(save_path, "tog_sweep2_a.jld2") trajectories_a susceptibilities_a objective_values_a fidelities_a susceptibilities_a_mean susceptibilities_a_std objective_values_a_mean objective_values_a_std fidelities_a_mean fidelities_a_std a_bounds dda_bound_fixed seed_values
println("✓ Saved trajectory data")

# Save summary
open(joinpath(save_path, "summary.txt"), "w") do io
    println(io, "Constraint Sweep Analysis")
    println(io, "="^60)
    println(io, "Timestamp: $timestamp")
    println(io, "\nConfiguration:")
    println(io, "  Gate: Hadamard")
    println(io, "  Time horizon (T): $T")
    println(io, "  Time step range: $Δt_min to $Δt_max")
    println(io, "  Number of points per sweep: $n_points")
    println(io, "  Number of seeds: $n_seeds")
    println(io, "  Random seeds: $seed_values")
    println(io, "\nSweep 1 (dda_bound):")
    println(io, "  Fixed a_bound: $a_bound_fixed")
    println(io, "  dda_bound range: -0.2 to 1")
    println(io, "\nSweep 2 (a_bound):")
    println(io, "  Fixed dda_bound: $dda_bound_fixed")
    println(io, "  a_bound range: -0.2 to 1")
    println(io, "\n" * summary_text)
end
println("✓ Saved summary")

println("\n" * "="^80)
println("Analysis complete! Results saved to: $save_path")
println("="^80)