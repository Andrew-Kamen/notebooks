# Time Step Study: Toggle-Frame Objective
# Studies the effect of increasing time steps T while keeping total duration Δt * T = 4.0 constant
import Pkg; Pkg.activate(@__DIR__)
Pkg.instantiate();
using PiccoloQuantumObjects
using QuantumCollocation
using LinearAlgebra
using Statistics
using CairoMakie
using Random
using Printf
using JLD2
using Dates

# Problem parameters
TOTAL_DURATION = 4.0
U_goal = GATES.H
H_drive = [PAULIS.X, PAULIS.Y, PAULIS.Z]
piccolo_opts = PiccoloOptions(verbose=false)
sys = QuantumSystem(H_drive)
a_bound = 1.0
dda_bound = 1.0
H_ϵ_add = a -> [PAULIS.X, PAULIS.Y, PAULIS.Z]

# Time step study parameters
T_values = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
Δt_values = [TOTAL_DURATION / T for T in T_values]
n_seeds = 2
seed_values = [42, 123]

println("Time Step Study: Toggle-Frame Objective")
println("="^70)
println("Total Duration: ", TOTAL_DURATION)
println("T values: ", T_values)
println("Δt values: ", Δt_values)
println("Seeds: ", seed_values)
println()

# Storage for results
n_T = length(T_values)
trajectories = Array{Any}(undef, n_T, n_seeds)
fidelities = zeros(n_T, n_seeds)
wall_times = zeros(n_T, n_seeds)

# Main optimization loop
for (i, (T, Δt)) in enumerate(zip(T_values, Δt_values))
    println("\n", "="^70)
    println("Study $i/$n_T: T=$T, Δt=$(Printf.@sprintf("%.4f", Δt)), Duration=$(Printf.@sprintf("%.2f", T*Δt))")
    println("="^70)
    
    for (seed_idx, seed) in enumerate(seed_values)
        println("\n  Seed $seed_idx (value=$seed):")
        Random.seed!(seed)
        
        # Create toggle-frame problem
        prob = UnitarySmoothPulseProblem(
            sys, U_goal, T, Δt;
            a_bound=a_bound,
            dda_bound=dda_bound,
            piccolo_options=piccolo_opts,
            activate_rob_loss=true,
            H_err=H_ϵ_add
        )
        
        # Solve with timing
        elapsed_time = @elapsed begin
            solve!(prob, max_iter=500, print_level=0, options=IpoptOptions(eval_hessian=false))
            solve!(prob, max_iter=50, print_level=0)
        end
        
        # Calculate fidelity
        fidelity = unitary_rollout_fidelity(prob.trajectory, sys)
        
        # Store results
        trajectories[i, seed_idx] = prob.trajectory
        fidelities[i, seed_idx] = fidelity
        wall_times[i, seed_idx] = elapsed_time
        
        println("    -> Fidelity: $(Printf.@sprintf("%.6f", fidelity))")
        println("    -> Wall time: $(Printf.@sprintf("%.2f", elapsed_time))s")
    end
end

# Compute statistics
mean_fidelities = mean(fidelities, dims=2)[:, 1]
std_fidelities = std(fidelities, dims=2)[:, 1]
mean_times = mean(wall_times, dims=2)[:, 1]

println("\n" * "="^70)
println("TOGGLE-FRAME TIME STEP STUDY COMPLETE")
println("="^70)
println("\nSummary Statistics:")
for (i, T) in enumerate(T_values)
    println("T=$T: Fidelity=$(Printf.@sprintf("%.6f", mean_fidelities[i])) ± $(Printf.@sprintf("%.6f", std_fidelities[i])), Time=$(Printf.@sprintf("%.2f", mean_times[i]))s")
end

# Create visualization
fig = Figure(size=(1600, 1200))

ax1 = Axis(fig[1, 1],
    title="Fidelity vs Time Steps (Toggle-Frame)",
    xlabel="Number of Time Steps (T)",
    ylabel="Fidelity"
)
lines!(ax1, T_values, mean_fidelities, color=:blue, linewidth=3, label="Mean")
band!(ax1, T_values, 
      mean_fidelities .- std_fidelities,
      mean_fidelities .+ std_fidelities,
      color=(:blue, 0.3), label="±1 Std")
for seed_idx in 1:n_seeds
    scatter!(ax1, T_values, fidelities[:, seed_idx], label="Seed $seed_idx", markersize=8)
end
axislegend(ax1, position=:rb)

ax2 = Axis(fig[1, 2],
    title="Computation Time vs Time Steps",
    xlabel="Number of Time Steps (T)",
    ylabel="Wall Time (s)"
)
lines!(ax2, T_values, mean_times, color=:red, linewidth=3, label="Mean")
for seed_idx in 1:n_seeds
    scatter!(ax2, T_values, wall_times[:, seed_idx], label="Seed $seed_idx", markersize=8)
end
axislegend(ax2, position=:lt)

ax3 = Axis(fig[2, 1],
    title="Fidelity vs Time Step Size",
    xlabel="Time Step Size (Δt)",
    ylabel="Fidelity"
)
lines!(ax3, Δt_values, mean_fidelities, color=:green, linewidth=3, label="Mean")
band!(ax3, Δt_values,
      mean_fidelities .- std_fidelities,
      mean_fidelities .+ std_fidelities,
      color=(:green, 0.3))
axislegend(ax3)

ax4 = Axis(fig[2, 2],
    title="Summary Statistics"
)
hidedecorations!(ax4)
hidespines!(ax4)

summary_text = """
Toggle-Frame Time Step Study
Total Duration: $(TOTAL_DURATION)

Best Fidelity: $(Printf.@sprintf("%.6f", maximum(mean_fidelities)))
  at T=$(T_values[argmax(mean_fidelities)])

Worst Fidelity: $(Printf.@sprintf("%.6f", minimum(mean_fidelities)))
  at T=$(T_values[argmin(mean_fidelities)])

Fastest Time: $(Printf.@sprintf("%.2f", minimum(mean_times)))s
  at T=$(T_values[argmin(mean_times)])

Slowest Time: $(Printf.@sprintf("%.2f", maximum(mean_times)))s
  at T=$(T_values[argmax(mean_times)])

Avg Fidelity: $(Printf.@sprintf("%.6f", mean(mean_fidelities)))
Avg Time: $(Printf.@sprintf("%.2f", mean(mean_times)))s
"""

text!(ax4, 0.1, 0.5, text=summary_text, space=:relative, fontsize=12, font="monospace")

Label(fig[0, :], "Toggle-Frame: Time Step Study (Fixed Duration = $(TOTAL_DURATION))",
      fontsize=20, font="bold")

display(fig)

# Save results
output_dir = joinpath(@__DIR__, "timestep_study_results", "toggle_frame")
mkpath(output_dir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

# Save figure
save(joinpath(output_dir, "timestep_analysis_$(timestamp).png"), fig, px_per_unit=2)

# Save data
data_file = joinpath(output_dir, "timestep_data_$(timestamp).jld2")
@save data_file T_values Δt_values trajectories fidelities wall_times TOTAL_DURATION seed_values a_bound dda_bound

# Save summary
summary_file = joinpath(output_dir, "summary_$(timestamp).txt")
open(summary_file, "w") do io
    println(io, "Toggle-Frame Time Step Study")
    println(io, "="^70)
    println(io, "Total Duration: ", TOTAL_DURATION)
    println(io, "a_bound: ", a_bound)
    println(io, "dda_bound: ", dda_bound)
    println(io, "Seeds: ", seed_values)
    println(io, "\nResults:")
    for (i, T) in enumerate(T_values)
        println(io, "T=$T, Δt=$(Printf.@sprintf("%.4f", Δt_values[i])): ")
        println(io, "  Fidelity: $(Printf.@sprintf("%.6f", mean_fidelities[i])) ± $(Printf.@sprintf("%.6f", std_fidelities[i]))")
        println(io, "  Time: $(Printf.@sprintf("%.2f", mean_times[i]))s")
    end
end

println("\nResults saved to: $(output_dir)")