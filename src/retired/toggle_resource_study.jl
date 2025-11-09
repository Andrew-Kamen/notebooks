# Time Step Study: Toggle-Frame Objective with Convergence Tracking
# Studies the effect of increasing time steps T while keeping total duration Δt * T = 4.0 constant
# Now tracks convergence time to target objective value
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

# Convergence tracking parameters
TARGET_OBJECTIVE = 1e-3  # Target objective value for convergence
MAX_CONVERGENCE_TIME = Inf  # Maximum time to wait for convergence

# Time step study parameters
T_values = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
Δt_values = [TOTAL_DURATION / T for T in T_values]
n_seeds = 2
seed_values = [42, 123]

println("Time Step Study: Toggle-Frame Objective with Convergence Tracking")
println("="^70)
println("Total Duration: ", TOTAL_DURATION)
println("Target Objective Value: ", TARGET_OBJECTIVE)
println("T values: ", T_values)
println("Δt values: ", Δt_values)
println("Seeds: ", seed_values)
println()

# Storage for results
n_T = length(T_values)
trajectories = Array{Any}(undef, n_T, n_seeds)
fidelities = zeros(n_T, n_seeds)
wall_times = zeros(n_T, n_seeds)
convergence_times = fill(Inf, n_T, n_seeds)  # Time to reach target objective
final_objectives = zeros(n_T, n_seeds)
converged_flags = falses(n_T, n_seeds)

# Callback to track convergence time
mutable struct ConvergenceTracker
    start_time::Float64
    target_objective::Float64
    convergence_time::Float64
    converged::Bool

    ConvergenceTracker(target) = new(time(), target, Inf, false)
end

function convergence_callback(tracker::ConvergenceTracker)
    return function(prob, iter, obj_val, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials)
        if !tracker.converged && obj_val <= tracker.target_objective
            tracker.convergence_time = time() - tracker.start_time
            tracker.converged = true
        end
        return true  # Continue optimization
    end
end

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

        # Setup convergence tracker
        tracker = ConvergenceTracker(TARGET_OBJECTIVE)

        # Solve with timing and convergence tracking
        tracker.start_time = time()
        elapsed_time = @elapsed begin
            # First pass with Hessian approximation
            solve!(prob, max_iter=500, print_level=0,
                   options=IpoptOptions(eval_hessian=false))

            # Reset tracker for second pass if not yet converged
            if !tracker.converged
                tracker.start_time = time()
            end

            # Second pass with full Hessian
            solve!(prob, max_iter=50, print_level=0)
        end

        # Calculate fidelity and final objective
        fidelity = unitary_rollout_fidelity(prob.trajectory, sys)
        final_obj = get_objective(prob)

        # Check if converged by end of optimization
        if !tracker.converged && final_obj <= TARGET_OBJECTIVE
            tracker.convergence_time = elapsed_time
            tracker.converged = true
        end

        # Store results
        trajectories[i, seed_idx] = prob.trajectory
        fidelities[i, seed_idx] = fidelity
        wall_times[i, seed_idx] = elapsed_time
        convergence_times[i, seed_idx] = tracker.convergence_time
        converged_flags[i, seed_idx] = tracker.converged
        final_objectives[i, seed_idx] = final_obj

        println("    -> Fidelity: $(Printf.@sprintf("%.6f", fidelity))")
        println("    -> Final Objective: $(Printf.@sprintf("%.3e", final_obj))")
        println("    -> Wall time: $(Printf.@sprintf("%.2f", elapsed_time))s")
        println("    -> Converged: $(tracker.converged)")
        if tracker.converged
            println("    -> Convergence time: $(Printf.@sprintf("%.2f", tracker.convergence_time))s")
        end
    end
end

# Compute statistics
mean_fidelities = mean(fidelities, dims=2)[:, 1]
std_fidelities = std(fidelities, dims=2)[:, 1]
mean_times = mean(wall_times, dims=2)[:, 1]

# Convergence statistics (only for converged runs)
mean_convergence_times = zeros(n_T)
std_convergence_times = zeros(n_T)
convergence_rates = zeros(n_T)

for i in 1:n_T
    converged_mask = converged_flags[i, :]
    convergence_rates[i] = sum(converged_mask) / n_seeds

    if any(converged_mask)
        conv_times = convergence_times[i, converged_mask]
        mean_convergence_times[i] = mean(conv_times)
        std_convergence_times[i] = length(conv_times) > 1 ? std(conv_times) : 0.0
    else
        mean_convergence_times[i] = NaN
        std_convergence_times[i] = NaN
    end
end

println("\n" * "="^70)
println("TOGGLE-FRAME TIME STEP STUDY COMPLETE")
println("="^70)
println("\nSummary Statistics:")
for (i, T) in enumerate(T_values)
    println("T=$T:")
    println("  Fidelity: $(Printf.@sprintf("%.6f", mean_fidelities[i])) ± $(Printf.@sprintf("%.6f", std_fidelities[i]))")
    println("  Wall Time: $(Printf.@sprintf("%.2f", mean_times[i]))s")
    println("  Convergence Rate: $(Printf.@sprintf("%.0f", convergence_rates[i]*100))%")
    if convergence_rates[i] > 0
        println("  Convergence Time: $(Printf.@sprintf("%.2f", mean_convergence_times[i]))s ± $(Printf.@sprintf("%.2f", std_convergence_times[i]))s")
    end
end

# Create visualization
fig = Figure(size=(1800, 1400))

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
    title="Total Computation Time vs Time Steps",
    xlabel="Number of Time Steps (T)",
    ylabel="Wall Time (s)"
)
lines!(ax2, T_values, mean_times, color=:red, linewidth=3, label="Mean")
for seed_idx in 1:n_seeds
    scatter!(ax2, T_values, wall_times[:, seed_idx], label="Seed $seed_idx", markersize=8)
end
axislegend(ax2, position=:lt)

ax3 = Axis(fig[2, 1],
    title="Convergence Time vs Time Steps",
    xlabel="Number of Time Steps (T)",
    ylabel="Time to Reach Target Objective (s)"
)
# Only plot converged runs
valid_indices = .!isnan.(mean_convergence_times)
if any(valid_indices)
    lines!(ax3, T_values[valid_indices], mean_convergence_times[valid_indices],
           color=:green, linewidth=3, label="Mean")
    band!(ax3, T_values[valid_indices],
          mean_convergence_times[valid_indices] .- std_convergence_times[valid_indices],
          mean_convergence_times[valid_indices] .+ std_convergence_times[valid_indices],
          color=(:green, 0.3))
    for seed_idx in 1:n_seeds
        conv_mask = converged_flags[:, seed_idx]
        if any(conv_mask)
            scatter!(ax3, T_values[conv_mask], convergence_times[conv_mask, seed_idx],
                    label="Seed $seed_idx", markersize=8)
        end
    end
end
axislegend(ax3, position=:lt)

ax4 = Axis(fig[2, 2],
    title="Convergence Rate vs Time Steps",
    xlabel="Number of Time Steps (T)",
    ylabel="Convergence Rate (%)"
)
barplot!(ax4, T_values, convergence_rates .* 100, color=:purple, alpha=0.7)
hlines!(ax4, [100], color=:black, linestyle=:dash, linewidth=2)

ax5 = Axis(fig[3, 1],
    title="Final Objective Value vs Time Steps",
    xlabel="Number of Time Steps (T)",
    ylabel="Final Objective Value (log scale)",
    yscale=log10
)
for seed_idx in 1:n_seeds
    scatter!(ax5, T_values, final_objectives[:, seed_idx],
            label="Seed $seed_idx", markersize=8)
end
hlines!(ax5, [TARGET_OBJECTIVE], color=:red, linestyle=:dash,
        linewidth=2, label="Target")
axislegend(ax5, position=:rt)

ax6 = Axis(fig[3, 2],
    title="Summary Statistics"
)
hidedecorations!(ax6)
hidespines!(ax6)

summary_text = """
Toggle-Frame Time Step Study
Total Duration: $(TOTAL_DURATION)
Target Objective: $(TARGET_OBJECTIVE)

Best Fidelity: $(Printf.@sprintf("%.6f", maximum(mean_fidelities)))
  at T=$(T_values[argmax(mean_fidelities)])

Overall Convergence Rate: $(Printf.@sprintf("%.0f", mean(convergence_rates)*100))%

Fastest Convergence: $(Printf.@sprintf("%.2f", minimum(mean_convergence_times[valid_indices])))s
  at T=$(T_values[valid_indices][argmin(mean_convergence_times[valid_indices])])

Avg Fidelity: $(Printf.@sprintf("%.6f", mean(mean_fidelities)))
Avg Wall Time: $(Printf.@sprintf("%.2f", mean(mean_times)))s
Avg Convergence Time: $(Printf.@sprintf("%.2f", mean(mean_convergence_times[valid_indices])))s
"""

text!(ax6, 0.1, 0.5, text=summary_text, space=:relative, fontsize=12, font="monospace")

Label(fig[0, :], "Toggle-Frame: Time Step Study with Convergence Tracking (Duration = $(TOTAL_DURATION))",
      fontsize=20, font="bold")

display(fig)

# Save results
output_dir = joinpath(@__DIR__, "timestep_study_results", "toggle_frame")
mkpath(output_dir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

# Save figure
save(joinpath(output_dir, "timestep_convergence_analysis_$(timestamp).png"), fig, px_per_unit=2)

# Save data
data_file = joinpath(output_dir, "timestep_convergence_data_$(timestamp).jld2")
@save data_file T_values Δt_values trajectories fidelities wall_times convergence_times converged_flags final_objectives TOTAL_DURATION TARGET_OBJECTIVE seed_values a_bound dda_bound

# Save summary
summary_file = joinpath(output_dir, "convergence_summary_$(timestamp).txt")
open(summary_file, "w") do io
    println(io, "Toggle-Frame Time Step Study with Convergence Tracking")
    println(io, "="^70)
    println(io, "Total Duration: ", TOTAL_DURATION)
    println(io, "Target Objective: ", TARGET_OBJECTIVE)
    println(io, "a_bound: ", a_bound)
    println(io, "dda_bound: ", dda_bound)
    println(io, "Seeds: ", seed_values)
    println(io, "\nResults:")
    for (i, T) in enumerate(T_values)
        println(io, "\nT=$T, Δt=$(Printf.@sprintf("%.4f", Δt_values[i])):")
        println(io, "  Fidelity: $(Printf.@sprintf("%.6f", mean_fidelities[i])) ± $(Printf.@sprintf("%.6f", std_fidelities[i]))")
        println(io, "  Wall Time: $(Printf.@sprintf("%.2f", mean_times[i]))s")
        println(io, "  Convergence Rate: $(Printf.@sprintf("%.0f", convergence_rates[i]*100))%")
        if convergence_rates[i] > 0
            println(io, "  Convergence Time: $(Printf.@sprintf("%.2f", mean_convergence_times[i]))s ± $(Printf.@sprintf("%.2f", std_convergence_times[i]))s")
        end
        for seed_idx in 1:n_seeds
            println(io, "    Seed $(seed_values[seed_idx]): Obj=$(Printf.@sprintf("%.3e", final_objectives[i, seed_idx])), Converged=$(converged_flags[i, seed_idx])")
        end
    end
end

println("\nResults saved to: $(output_dir)")