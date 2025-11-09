#!/usr/bin/env julia

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
# Load Data
# ============================================================================

println("Loading data...")


println("Loading Adjoint method data...")
adjoint_sweep1_path = "artifacts/adj_sus_dda_sweep_20251001_010151/sweep1_dda.jld2"
adjoint_sweep2_path = "artifacts/adj_sus_dda_sweep_20251001_010151/sweep2_a.jld2"

@load adjoint_sweep1_path trajectories_dda susceptibilities_dda objective_values_dda fidelities_dda dda_bounds a_bound_fixed
adjoint_dda_susc = susceptibilities_dda
adjoint_dda_bounds = dda_bounds

@load adjoint_sweep2_path trajectories_a susceptibilities_a objective_values_a fidelities_a a_bounds dda_bound_fixed
adjoint_a_susc = susceptibilities_a
adjoint_a_bounds = a_bounds

# Load Toggle method data (document 2)
println("Loading Toggle method data...")
toggle_sweep1_path = "artifacts/adj_sus_dda_sweep_20251001_031619/sweep1_dda.jld2"  # Update with actual path
toggle_sweep2_path = "artifacts/adj_sus_dda_sweep_20251001_031619/sweep2_a.jld2"    # Update with actual path

@load toggle_sweep1_path trajectories_dda susceptibilities_dda objective_values_dda fidelities_dda dda_bounds a_bound_fixed
toggle_dda_susc = susceptibilities_dda
toggle_dda_bounds = dda_bounds

@load toggle_sweep2_path trajectories_a susceptibilities_a objective_values_a fidelities_a a_bounds dda_bound_fixed
toggle_a_susc = susceptibilities_a
toggle_a_bounds = a_bounds

println("✓ Data loaded successfully")

# ============================================================================
# Create Comparison Plots
# ============================================================================

println("\nCreating comparison plots...")

fig = Figure(size=(1600, 800))

# Plot 1: Susceptibility vs dda_bound (both methods)
ax1 = Axis(fig[1, 1],
    title="Susceptibility vs dda_bound: Adjoint vs Toggle",
    xlabel="dda_bound",
    ylabel="Susceptibility (var_obj)",
    xscale=log10
)

# Adjoint method
lines!(ax1, collect(adjoint_dda_bounds), adjoint_dda_susc, 
       color=:blue, linewidth=2.5, label="Adjoint")
scatter!(ax1, collect(adjoint_dda_bounds), adjoint_dda_susc, 
         color=:blue, markersize=10)

# Toggle method
lines!(ax1, collect(toggle_dda_bounds), toggle_dda_susc, 
       color=:red, linewidth=2.5, label="Toggle", linestyle=:dash)
scatter!(ax1, collect(toggle_dda_bounds), toggle_dda_susc, 
         color=:red, markersize=10, marker=:diamond)

axislegend(ax1, position=:rt, framevisible=true)

# Plot 2: Susceptibility vs a_bound (both methods)
ax2 = Axis(fig[1, 2],
    title="Susceptibility vs a_bound: Adjoint vs Toggle",
    xlabel="a_bound",
    ylabel="Susceptibility (var_obj)",
    xscale=log10
)

# Adjoint method
lines!(ax2, collect(adjoint_a_bounds), adjoint_a_susc, 
       color=:blue, linewidth=2.5, label="Adjoint")
scatter!(ax2, collect(adjoint_a_bounds), adjoint_a_susc, 
         color=:blue, markersize=10)

# Toggle method
lines!(ax2, collect(toggle_a_bounds), toggle_a_susc, 
       color=:red, linewidth=2.5, label="Toggle", linestyle=:dash)
scatter!(ax2, collect(toggle_a_bounds), toggle_a_susc, 
         color=:red, markersize=10, marker=:diamond)

axislegend(ax2, position=:rt, framevisible=true)

# ============================================================================
# Create Additional Comparison Plot (Linear scale)
# ============================================================================

fig2 = Figure(size=(1600, 800))

# Plot 3: Linear scale comparison for dda_bound
ax3 = Axis(fig2[1, 1],
    title="Susceptibility vs dda_bound: Adjoint vs Toggle (Linear Scale)",
    xlabel="dda_bound",
    ylabel="Susceptibility (var_obj)"
)

lines!(ax3, collect(adjoint_dda_bounds), adjoint_dda_susc, 
       color=:blue, linewidth=2.5, label="Adjoint")
scatter!(ax3, collect(adjoint_dda_bounds), adjoint_dda_susc, 
         color=:blue, markersize=10)

lines!(ax3, collect(toggle_dda_bounds), toggle_dda_susc, 
       color=:red, linewidth=2.5, label="Toggle", linestyle=:dash)
scatter!(ax3, collect(toggle_dda_bounds), toggle_dda_susc, 
         color=:red, markersize=10, marker=:diamond)

axislegend(ax3, position=:rt, framevisible=true)

# Plot 4: Linear scale comparison for a_bound
ax4 = Axis(fig2[1, 2],
    title="Susceptibility vs a_bound: Adjoint vs Toggle (Linear Scale)",
    xlabel="a_bound",
    ylabel="Susceptibility (var_obj)"
)

lines!(ax4, collect(adjoint_a_bounds), adjoint_a_susc, 
       color=:blue, linewidth=2.5, label="Adjoint")
scatter!(ax4, collect(adjoint_a_bounds), adjoint_a_susc, 
         color=:blue, markersize=10)

lines!(ax4, collect(toggle_a_bounds), toggle_a_susc, 
       color=:red, linewidth=2.5, label="Toggle", linestyle=:dash)
scatter!(ax4, collect(toggle_a_bounds), toggle_a_susc, 
         color=:red, markersize=10, marker=:diamond)

axislegend(ax4, position=:rt, framevisible=true)

# ============================================================================
# Print Summary Statistics
# ============================================================================

println("\n" * "="^80)
println("COMPARISON SUMMARY")
println("="^80)

println("\nSweep 1 (dda_bound variation):")
println("  Adjoint susceptibility range: $(Printf.@sprintf("%.3e", minimum(adjoint_dda_susc))) to $(Printf.@sprintf("%.3e", maximum(adjoint_dda_susc)))")
println("  Toggle susceptibility range:  $(Printf.@sprintf("%.3e", minimum(toggle_dda_susc))) to $(Printf.@sprintf("%.3e", maximum(toggle_dda_susc)))")
println("  Adjoint mean: $(Printf.@sprintf("%.3e", mean(adjoint_dda_susc)))")
println("  Toggle mean:  $(Printf.@sprintf("%.3e", mean(toggle_dda_susc)))")

println("\nSweep 2 (a_bound variation):")
println("  Adjoint susceptibility range: $(Printf.@sprintf("%.3e", minimum(adjoint_a_susc))) to $(Printf.@sprintf("%.3e", maximum(adjoint_a_susc)))")
println("  Toggle susceptibility range:  $(Printf.@sprintf("%.3e", minimum(toggle_a_susc))) to $(Printf.@sprintf("%.3e", maximum(toggle_a_susc)))")
println("  Adjoint mean: $(Printf.@sprintf("%.3e", mean(adjoint_a_susc)))")
println("  Toggle mean:  $(Printf.@sprintf("%.3e", mean(toggle_a_susc)))")

# ============================================================================
# Save Comparison Plots
# ============================================================================

println("\n" * "="^80)
println("Saving comparison plots...")
println("="^80)

save_dir = "artifacts/method_comparison"
mkpath(save_dir)

save(joinpath(save_dir, "comparison_logscale.png"), fig, px_per_unit=2)
save(joinpath(save_dir, "comparison_linear.png"), fig2, px_per_unit=2)

println("✓ Saved comparison plots to: $save_dir")
println("\nAnalysis complete!")