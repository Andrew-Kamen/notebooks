# Time Step Study: Comparison Across All Objectives
# Loads and compares results from Toggle-Frame, Variational (Adjoint), and Universal objectives
import Pkg; Pkg.activate(@__DIR__)
Pkg.instantiate();
using Statistics
using CairoMakie
using Printf
using JLD2
using Dates

base_dir = joinpath(@__DIR__, "timestep_study_results")

# Function to load most recent file from a directory
function load_latest_results(subdir::String)
    dir = joinpath(base_dir, subdir)
    files = filter(f -> startswith(f, "timestep_data_") && endswith(f, ".jld2"), readdir(dir))
    if isempty(files)
        error("No data files found in $dir")
    end
    latest_file = sort(files)[end]  # Get most recent by timestamp
    return load(joinpath(dir, latest_file))
end

println("Loading data from all three methods...")
toggle_data = load_latest_results("toggle_frame")
var_data = load_latest_results("variational")
uni_data = load_latest_results("universal")
println("Data loaded successfully!\n")

# Extract common parameters
T_values = toggle_data["T_values"]
Δt_values = toggle_data["Δt_values"]
TOTAL_DURATION = toggle_data["TOTAL_DURATION"]

# Compute statistics for each method
toggle_mean_fid = mean(toggle_data["fidelities"], dims=2)[:, 1]
toggle_std_fid = std(toggle_data["fidelities"], dims=2)[:, 1]
toggle_mean_time = mean(toggle_data["wall_times"], dims=2)[:, 1]

var_mean_fid = mean(var_data["fidelities"], dims=2)[:, 1]
var_std_fid = std(var_data["fidelities"], dims=2)[:, 1]
var_mean_time = mean(var_data["wall_times"], dims=2)[:, 1]

uni_mean_fid = mean(uni_data["fidelities"], dims=2)[:, 1]
uni_std_fid = std(uni_data["fidelities"], dims=2)[:, 1]
uni_mean_time = mean(uni_data["wall_times"], dims=2)[:, 1]

println("Comparison Statistics Computed\n")

# Create comprehensive comparison visualization
fig = Figure(size=(2000, 1400))

# Plot 1: Fidelity Comparison vs T
ax1 = Axis(fig[1, 1:2],
    title="Fidelity Comparison: All Objectives vs Time Steps",
    xlabel="Number of Time Steps (T)",
    ylabel="Fidelity"
)

# Toggle-Frame
lines!(ax1, T_values, toggle_mean_fid, color=:blue, linewidth=3, label="Toggle-Frame")
band!(ax1, T_values,
      toggle_mean_fid .- toggle_std_fid,
      toggle_mean_fid .+ toggle_std_fid,
      color=(:blue, 0.2))

# Variational
lines!(ax1, T_values, var_mean_fid, color=:purple, linewidth=3, label="Variational/Adjoint")
band!(ax1, T_values,
      var_mean_fid .- var_std_fid,
      var_mean_fid .+ var_std_fid,
      color=(:purple, 0.2))

# Universal
lines!(ax1, T_values, uni_mean_fid, color=:crimson, linewidth=3, label="Universal")
band!(ax1, T_values,
      uni_mean_fid .- uni_std_fid,
      uni_mean_fid .+ uni_std_fid,
      color=(:crimson, 0.2))

hlines!(ax1, [0.999], color=:black, linestyle=:dash, linewidth=1.5, label="0.999 threshold")
axislegend(ax1, position=:rb)

# Plot 2: Computation Time Comparison
ax2 = Axis(fig[1, 3],
    title="Computation Time vs T",
    xlabel="Number of Time Steps (T)",
    ylabel="Wall Time (s)",
    yscale=log10
)

lines!(ax2, T_values, toggle_mean_time, color=:blue, linewidth=3, label="Toggle-Frame")
lines!(ax2, T_values, var_mean_time, color=:purple, linewidth=3, label="Variational")
lines!(ax2, T_values, uni_mean_time, color=:crimson, linewidth=3, label="Universal")
axislegend(ax2, position=:lt)

# Plot 3: Fidelity vs Δt
ax3 = Axis(fig[2, 1],
    title="Fidelity vs Time Step Size",
    xlabel="Time Step Size (Δt)",
    ylabel="Fidelity"
)

lines!(ax3, Δt_values, toggle_mean_fid, color=:blue, linewidth=3, label="Toggle-Frame")
lines!(ax3, Δt_values, var_mean_fid, color=:purple, linewidth=3, label="Variational")
lines!(ax3, Δt_values, uni_mean_fid, color=:crimson, linewidth=3, label="Universal")
axislegend(ax3, position=:rb)

# Plot 4: Fidelity Difference from Toggle-Frame
ax4 = Axis(fig[2, 2],
    title="Fidelity Difference vs Toggle-Frame (Baseline)",
    xlabel="Number of Time Steps (T)",
    ylabel="Δ Fidelity"
)

lines!(ax4, T_values, var_mean_fid .- toggle_mean_fid, 
       color=:purple, linewidth=3, label="Variational - Toggle")
lines!(ax4, T_values, uni_mean_fid .- toggle_mean_fid, 
       color=:crimson, linewidth=3, label="Universal - Toggle")
hlines!(ax4, [0], color=:black, linestyle=:dash, linewidth=1)
axislegend(ax4, position=:rt)

# Plot 5: Time Efficiency (Fidelity per second)
ax5 = Axis(fig[2, 3],
    title="Fidelity per Second",
    xlabel="Number of Time Steps (T)",
    ylabel="Fidelity / Wall Time (1/s)"
)

lines!(ax5, T_values, toggle_mean_fid ./ toggle_mean_time, 
       color=:blue, linewidth=3, label="Toggle-Frame")
lines!(ax5, T_values, var_mean_fid ./ var_mean_time, 
       color=:purple, linewidth=3, label="Variational")
lines!(ax5, T_values, uni_mean_fid ./ uni_mean_time, 
       color=:crimson, linewidth=3, label="Universal")
axislegend(ax5, position=:rt)

# Plot 6: Summary Statistics
ax6 = Axis(fig[3, 1:3],
    title="Summary Statistics"
)
hidedecorations!(ax6)
hidespines!(ax6)

summary_text = """
Time Step Study Comparison (Fixed Duration = $(TOTAL_DURATION))
$(repeat("=", 80))

TOGGLE-FRAME:
  Best Fidelity: $(Printf.@sprintf("%.6f", maximum(toggle_mean_fid))) at T=$(T_values[argmax(toggle_mean_fid)])
  Avg Fidelity:  $(Printf.@sprintf("%.6f", mean(toggle_mean_fid)))
  Avg Time:      $(Printf.@sprintf("%.2f", mean(toggle_mean_time)))s
  T > 0.999:     $(sum(toggle_mean_fid .> 0.999)) / $(length(T_values))

VARIATIONAL (ADJOINT):
  Best Fidelity: $(Printf.@sprintf("%.6f", maximum(var_mean_fid))) at T=$(T_values[argmax(var_mean_fid)])
  Avg Fidelity:  $(Printf.@sprintf("%.6f", mean(var_mean_fid)))
  Avg Time:      $(Printf.@sprintf("%.2f", mean(var_mean_time)))s
  T > 0.999:     $(sum(var_mean_fid .> 0.999)) / $(length(T_values))

UNIVERSAL:
  Best Fidelity: $(Printf.@sprintf("%.6f", maximum(uni_mean_fid))) at T=$(T_values[argmax(uni_mean_fid)])
  Avg Fidelity:  $(Printf.@sprintf("%.6f", mean(uni_mean_fid)))
  Avg Time:      $(Printf.@sprintf("%.2f", mean(uni_mean_time)))s
  T > 0.999:     $(sum(uni_mean_fid .> 0.999)) / $(length(T_values))

RELATIVE PERFORMANCE (vs Toggle-Frame):
  Variational Fidelity Δ:  $(Printf.@sprintf("%+.6f", mean(var_mean_fid - toggle_mean_fid)))
  Universal Fidelity Δ:    $(Printf.@sprintf("%+.6f", mean(uni_mean_fid - toggle_mean_fid)))
  Variational Time Ratio:  $(Printf.@sprintf("%.2f", mean(var_mean_time) / mean(toggle_mean_time)))x
  Universal Time Ratio:    $(Printf.@sprintf("%.2f", mean(uni_mean_time) / mean(toggle_mean_time)))x
"""

text!(ax6, 0.05, 0.5, text=summary_text,
      space=:relative, fontsize=11, font="monospace")

Label(fig[0, :], "Time Step Study: Comparison Across All Objectives",
      fontsize=22, font="bold")

display(fig)

# Save comparison results
output_dir = joinpath(@__DIR__, "timestep_study_results", "comparison")
mkpath(output_dir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

# Save figure
save(joinpath(output_dir, "comparison_main_$(timestamp).png"), fig, px_per_unit=2)

# Save comparison summary
summary_file = joinpath(output_dir, "comparison_summary_$(timestamp).txt")
open(summary_file, "w") do io
    println(io, "Time Step Study Comparison")
    println(io, "="^80)
    println(io, "\nTotal Duration: ", TOTAL_DURATION)
    println(io, "T values: ", T_values)
    println(io, "\n" * "="^80)
    
    println(io, "\nTOGGLE-FRAME RESULTS:")
    for (i, T) in enumerate(T_values)
        println(io, "  T=$T: Fidelity=$(Printf.@sprintf("%.6f", toggle_mean_fid[i])) ± $(Printf.@sprintf("%.6f", toggle_std_fid[i])), Time=$(Printf.@sprintf("%.2f", toggle_mean_time[i]))s")
    end
    
    println(io, "\nVARIATIONAL (ADJOINT) RESULTS:")
    for (i, T) in enumerate(T_values)
        println(io, "  T=$T: Fidelity=$(Printf.@sprintf("%.6f", var_mean_fid[i])) ± $(Printf.@sprintf("%.6f", var_std_fid[i])), Time=$(Printf.@sprintf("%.2f", var_mean_time[i]))s")
    end
    
    println(io, "\nUNIVERSAL RESULTS:")
    for (i, T) in enumerate(T_values)
        println(io, "  T=$T: Fidelity=$(Printf.@sprintf("%.6f", uni_mean_fid[i])) ± $(Printf.@sprintf("%.6f", uni_std_fid[i])), Time=$(Printf.@sprintf("%.2f", uni_mean_time[i]))s")
    end
    
    println(io, "\n" * "="^80)
    println(io, "\nOVERALL COMPARISON:")
    println(io, "Best method by avg fidelity: ", 
           ["Toggle-Frame", "Variational", "Universal"][argmax([mean(toggle_mean_fid), mean(var_mean_fid), mean(uni_mean_fid)])])
    println(io, "Fastest method: ",
           ["Toggle-Frame", "Variational", "Universal"][argmin([mean(toggle_mean_time), mean(var_mean_time), mean(uni_mean_time)])])
    println(io, "Most consistent (lowest std): ",
           ["Toggle-Frame", "Variational", "Universal"][argmin([mean(toggle_std_fid), mean(var_std_fid), mean(uni_std_fid)])])
end

println("\n" * "="^80)
println("COMPARISON COMPLETE")
println("="^80)
println("\nResults saved to: $(output_dir)")