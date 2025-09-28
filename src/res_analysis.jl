# Resource Analysis of Toggling, Adjoint, and Universal Robustness Objectives
# for Constraints and Penalty Methods
# Extended Analysis with Time Horizon Scaling

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.develop(path="../../QuantumCollocation.jl")

using PiccoloQuantumObjects
using QuantumCollocation
using ForwardDiff
using LinearAlgebra
using Plots
using SparseArrays
using NamedTrajectories
using Statistics
using CairoMakie
using Random
using ColorSchemes
using Makie
using Printf

# Problem parameters with varying T
T_values = 20:2:40  # T from 20 to 40 in increments of 2
n_T_values = length(T_values)
Δt = 0.2
U_goal = GATES.H
H_drive = [PAULIS.X, PAULIS.Y, PAULIS.Z]
piccolo_opts = PiccoloOptions(verbose=false)
pretty_print(X::AbstractMatrix) = Base.show(stdout, "text/plain", X)

sys = QuantumSystem(H_drive)

∂ₑH₀ = [PAULIS.X, PAULIS.Y, PAULIS.Z]
var_sys = VariationalQuantumSystem(
    H_drive,
    ∂ₑH₀
)

# Enhanced Performance tracking structures
mutable struct SolverMetrics
    wall_times::Vector{Float64}
    iter_times::Vector{Float64}
    objectives::Vector{Float64}
    constraints::Vector{Float64}
    memory_samples::Vector{Float64}  # Multiple memory samples during execution
    iterations::Vector{Int}
    convergence_rates::Vector{Float64}
    objective_trajectory::Vector{Float64}  # Track objective across iterations
    SolverMetrics() = new([], [], [], [], [], [], [], [])
end

function track_solver_performance_enhanced(prob; max_iter=150, print_level=5)
    metrics = SolverMetrics()
    start_time = time()
    
    # Sample memory multiple times during execution
    memory_samples = Float64[]
    
    # Initial memory
    GC.gc()  # Force garbage collection for cleaner measurement
    push!(memory_samples, Base.gc_live_bytes() / 1024^2)
    
    # Track initial objective
    Z_vec = vec(prob.trajectory)
    initial_obj = prob.objective.L(Z_vec)
    push!(metrics.objective_trajectory, initial_obj)
    
    # Solve with intermediate tracking
    iter_count = 0
    for batch in 1:ceil(Int, max_iter/10)
        batch_iter = min(10, max_iter - (batch-1)*10)
        if batch_iter <= 0
            break
        end
        
        result = solve!(prob; max_iter=batch_iter, print_level=print_level)
        iter_count += batch_iter
        
        # Sample memory during execution
        push!(memory_samples, Base.gc_live_bytes() / 1024^2)
        
        # Track objective value
        Z_vec = vec(prob.trajectory)
        current_obj = prob.objective.L(Z_vec)
        push!(metrics.objective_trajectory, current_obj)
        
        # Check convergence
        if length(metrics.objective_trajectory) > 1
            obj_change = abs(metrics.objective_trajectory[end] - metrics.objective_trajectory[end-1])
            if obj_change < 1e-6
                break
            end
        end
    end
    
    end_time = time()
    total_time = end_time - start_time
    
    # Final memory sample
    push!(memory_samples, Base.gc_live_bytes() / 1024^2)
    
    push!(metrics.wall_times, total_time)
    metrics.memory_samples = memory_samples
    push!(metrics.iterations, iter_count)
    
    # Get final objective
    try
        Z_vec = vec(prob.trajectory)
        push!(metrics.objectives, prob.objective.L(Z_vec))
    catch
        push!(metrics.objectives, NaN)
    end
    
    # Compute convergence rate if we have enough data
    if length(metrics.objective_trajectory) > 2
        conv_rate = abs(metrics.objective_trajectory[end] - metrics.objective_trajectory[end-1]) / 
                    abs(metrics.objective_trajectory[2] - metrics.objective_trajectory[1])
        push!(metrics.convergence_rates, conv_rate)
    end
    
    return metrics, total_time, memory_samples
end

function analyze_convergence_enhanced(metrics::SolverMetrics)
    memory_stats = if !isempty(metrics.memory_samples)
        Dict(
            "avg_memory_mb" => mean(metrics.memory_samples),
            "peak_memory_mb" => maximum(metrics.memory_samples),
            "median_memory_mb" => median(metrics.memory_samples),
            "memory_std_mb" => std(metrics.memory_samples)
        )
    else
        Dict(
            "avg_memory_mb" => 0.0,
            "peak_memory_mb" => 0.0,
            "median_memory_mb" => 0.0,
            "memory_std_mb" => 0.0
        )
    end
    
    return Dict(
        "total_wall_time" => isempty(metrics.wall_times) ? 0.0 : metrics.wall_times[end],
        "final_objective" => isempty(metrics.objectives) ? NaN : metrics.objectives[end],
        "total_iterations" => isempty(metrics.iterations) ? 0 : metrics.iterations[end],
        "convergence_rate" => isempty(metrics.convergence_rates) ? NaN : metrics.convergence_rates[end],
        "objective_trajectory" => metrics.objective_trajectory,
        memory_stats...
    )
end

# Initialize seeds and problems for different T values
n_guesses = 3  # Reduced for computational efficiency with more T values
n_drives = sys.n_drives
var_n_drives = var_sys.n_drives
variational_scales = fill(1.0, length(var_sys.G_vars))

# Store seeds for each T value
seeds_by_T = Dict()

for T in T_values
    seeds = []
    for i in 1:n_guesses
        Random.seed!(1234+10*i+T)  # Include T in seed for variety
        a_bounds = fill(1.0, n_drives)
        da_bounds = fill(1.0, n_drives)
        dda_bounds = fill(10^(1.5-0.5*i), n_drives)
        control_bounds = (a_bounds, da_bounds, dda_bounds)
        traj = initialize_trajectory(
                        U_goal,
                        T,
                        Δt,
                        n_drives,
                        control_bounds;
                        system=sys
                    )
        push!(seeds, traj)
    end
    seeds_by_T[T] = seeds
end

# Define fidelity targets
a_vals = exp.(range(log(100), log(100000), length=5))  # Reduced for efficiency
final_fid_floor = 1 .- 1 ./ a_vals
n_fidelities = length(final_fid_floor)

# Initialize storage for all T values
init_probs_by_T = Dict(
    "default" => Dict(),
    "variational" => Dict(),
    "toggling" => Dict(),
    "universal" => Dict()
)

init_fids_by_T = Dict(
    "default" => Dict(),
    "variational" => Dict(),
    "toggling" => Dict(),
    "universal" => Dict()
)

H₀_add = a -> [PAULIS.X, PAULIS.Y, PAULIS.Z]
X_drive = sys.H.H_drives[1]
H₀_mult = a -> a[1] * X_drive

println("Initializing problems for T values: ", T_values)

# Initialize problems for each T
for T in T_values
    println("\nInitializing problems for T=$T")
    
    # Initialize storage matrices for this T
    init_probs_by_T["default"][T] = Matrix{Any}(undef, n_guesses, n_fidelities)
    init_probs_by_T["variational"][T] = Matrix{Any}(undef, n_guesses, n_fidelities)
    init_probs_by_T["toggling"][T] = Matrix{Any}(undef, n_guesses, n_fidelities)
    init_probs_by_T["universal"][T] = Matrix{Any}(undef, n_guesses, n_fidelities)
    
    init_fids_by_T["default"][T] = zeros(n_guesses, n_fidelities)
    init_fids_by_T["variational"][T] = zeros(n_guesses, n_fidelities)
    init_fids_by_T["toggling"][T] = zeros(n_guesses, n_fidelities)
    init_fids_by_T["universal"][T] = zeros(n_guesses, n_fidelities)
    
    for i in 1:n_guesses
        for j in 1:n_fidelities
            Random.seed!(1234+10*i+T+j)
            
            # default case (no robustness)
            default = UnitarySmoothPulseProblem(sys, U_goal, T, Δt; 
                init_trajectory=deepcopy(seeds_by_T[T][i]))
            init_probs_by_T["default"][T][i, j] = default
            init_fids_by_T["default"][T][i,j] = unitary_rollout_fidelity(default.trajectory, sys)

            # variational objective
            var_prob = UnitaryVariationalProblem(var_sys, U_goal, T, Δt; 
                init_trajectory=deepcopy(seeds_by_T[T][i]), 
                robust_times=[[T], [T], [T]], Q_r=0.1, 
                piccolo_options=piccolo_opts)
            init_probs_by_T["variational"][T][i, j] = var_prob
            init_fids_by_T["variational"][T][i,j] = unitary_rollout_fidelity(var_prob.trajectory, sys)

            # toggling objective
            tog_prob = UnitarySmoothPulseProblem(sys, U_goal, T, Δt; 
                init_trajectory=deepcopy(seeds_by_T[T][i]), 
                activate_rob_loss=true, H_err=H₀_add, Q_t=0.1)
            init_probs_by_T["toggling"][T][i, j] = tog_prob
            init_fids_by_T["toggling"][T][i,j] = unitary_rollout_fidelity(tog_prob.trajectory, sys)

            # universal objective
            uni_prob = UnitaryUniversalProblem(sys, U_goal, T, Δt; 
                init_trajectory=deepcopy(seeds_by_T[T][i]), 
                activate_hyperspeed=true)
            init_probs_by_T["universal"][T][i, j] = uni_prob
            init_fids_by_T["universal"][T][i,j] = unitary_rollout_fidelity(uni_prob.trajectory, sys)
        end
    end
end

println("\nInitialization complete for all T values")

# Performance tracking for constraint-based methods across T values
constraint_metrics_by_T = Dict(
    "variational" => Dict(),
    "toggling" => Dict(),
    "universal" => Dict()
)

final_probs_by_T = Dict(
    "variational" => Dict(),
    "toggling" => Dict(),
    "universal" => Dict()
)

println("\nSolving constraint-based methods with performance tracking across T values...")

for T in T_values
    println("\nProcessing T=$T")
    
    # Initialize storage for this T
    constraint_metrics_by_T["variational"][T] = Matrix{SolverMetrics}(undef, n_guesses, n_fidelities)
    constraint_metrics_by_T["toggling"][T] = Matrix{SolverMetrics}(undef, n_guesses, n_fidelities)
    constraint_metrics_by_T["universal"][T] = Matrix{SolverMetrics}(undef, n_guesses, n_fidelities)
    
    final_probs_by_T["variational"][T] = Matrix{Any}(undef, n_guesses, n_fidelities)
    final_probs_by_T["toggling"][T] = Matrix{Any}(undef, n_guesses, n_fidelities)
    final_probs_by_T["universal"][T] = Matrix{Any}(undef, n_guesses, n_fidelities)
    
    for i in 1:n_guesses
        for j in 1:n_fidelities
            # Variational
            var_prob = UnitaryVariationalProblem(
                var_sys, U_goal, T, Δt;
                robust_times=[[T], [T], [T]],
                Q=0.0, Q_r=1.0,
                init_trajectory=init_probs_by_T["variational"][T][i,j].trajectory,
                var_seed=false,
                piccolo_options=piccolo_opts
            )
            F = final_fid_floor[j]
            push!(var_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, var_prob.trajectory))
            metrics, time, mem = track_solver_performance_enhanced(var_prob; max_iter=30, print_level=1)
            constraint_metrics_by_T["variational"][T][i,j] = metrics
            final_probs_by_T["variational"][T][i,j] = var_prob
            
            # Toggling
            tog_prob = UnitaryMaxToggleProblem(
                init_probs_by_T["toggling"][T][i,j], U_goal, H₀_add;
                Q_t=1.0, final_fidelity=final_fid_floor[j],
                piccolo_options=piccolo_opts
            )
            metrics, time, mem = track_solver_performance_enhanced(tog_prob; max_iter=30, print_level=1)
            constraint_metrics_by_T["toggling"][T][i,j] = metrics
            final_probs_by_T["toggling"][T][i,j] = tog_prob
            
            # Universal
            uni_prob = UnitaryMaxUniversalProblem(
                init_probs_by_T["universal"][T][i,j], U_goal;
                Q_t=1.0, final_fidelity=final_fid_floor[j],
                piccolo_options=piccolo_opts
            )
            metrics, time, mem = track_solver_performance_enhanced(uni_prob; max_iter=30, print_level=1)
            constraint_metrics_by_T["universal"][T][i,j] = metrics
            final_probs_by_T["universal"][T][i,j] = uni_prob
        end
    end
end

println("\nConstraint-based optimization complete")

# Performance tracking for penalty methods across T values
sweep_rob_loss_λ = exp.(range(log(.1), log(1), length=5))
n_lambdas = length(sweep_rob_loss_λ)

penalty_metrics_by_T = Dict(
    "variational" => Dict(),
    "toggling" => Dict(),
    "universal" => Dict()
)

pen_probs_by_T = Dict(
    "variational" => Dict(),
    "toggling" => Dict(),
    "universal" => Dict()
)

println("\nSolving penalty-based methods with performance tracking across T values...")

for T in T_values
    println("\nProcessing T=$T")
    
    # Initialize storage for this T
    penalty_metrics_by_T["variational"][T] = Matrix{SolverMetrics}(undef, n_guesses, n_lambdas)
    penalty_metrics_by_T["toggling"][T] = Matrix{SolverMetrics}(undef, n_guesses, n_lambdas)
    penalty_metrics_by_T["universal"][T] = Matrix{SolverMetrics}(undef, n_guesses, n_lambdas)
    
    pen_probs_by_T["variational"][T] = Matrix{Any}(undef, n_guesses, n_lambdas)
    pen_probs_by_T["toggling"][T] = Matrix{Any}(undef, n_guesses, n_lambdas)
    pen_probs_by_T["universal"][T] = Matrix{Any}(undef, n_guesses, n_lambdas)
    
    for i in 1:n_guesses
        for (λ_idx, λ) in enumerate(sweep_rob_loss_λ)
            # Use appropriate index for initial trajectory (min of λ_idx and n_fidelities)
            init_idx = min(λ_idx, n_fidelities)
            
            # Variational
            var_prob = UnitaryVariationalProblem(
                var_sys, U_goal, T, Δt;
                init_trajectory=deepcopy(init_probs_by_T["variational"][T][i,init_idx].trajectory),
                piccolo_options=piccolo_opts,
                var_seed=false, Q_r=λ
            )
            metrics, time, mem = track_solver_performance_enhanced(var_prob; max_iter=30, print_level=1)
            penalty_metrics_by_T["variational"][T][i, λ_idx] = metrics
            pen_probs_by_T["variational"][T][i, λ_idx] = var_prob
            
            # Toggling
            tog_prob = UnitarySmoothPulseProblem(
                sys, U_goal, T, Δt;
                init_trajectory=deepcopy(seeds_by_T[T][i]),
                piccolo_options=piccolo_opts,
                activate_rob_loss=true, H_err=H₀_add, Q_t=λ
            )
            metrics, time, mem = track_solver_performance_enhanced(tog_prob; max_iter=30, print_level=1)
            penalty_metrics_by_T["toggling"][T][i, λ_idx] = metrics
            pen_probs_by_T["toggling"][T][i, λ_idx] = tog_prob
            
            # Universal
            uni_prob = UnitaryUniversalProblem(
                sys, U_goal, T, Δt;
                init_trajectory=deepcopy(seeds_by_T[T][i]),
                piccolo_options=piccolo_opts,
                activate_hyperspeed=true, Q_t=λ
            )
            metrics, time, mem = track_solver_performance_enhanced(uni_prob; max_iter=30, print_level=1)
            penalty_metrics_by_T["universal"][T][i, λ_idx] = metrics
            pen_probs_by_T["universal"][T][i, λ_idx] = uni_prob
        end
    end
end

println("\nPenalty-based optimization complete")

# Analyze performance vs T for each method
method_performance_vs_T = Dict()
method_colors = [:blue, :red, :green]

for method in ["variational", "toggling", "universal"]
    T_data = Float64[]
    time_means = Float64[]
    time_stds = Float64[]
    mem_means = Float64[]
    mem_stds = Float64[]
    iter_means = Float64[]
    obj_final_means = Float64[]
    
    for T in T_values
        times = Float64[]
        memories = Float64[]
        iterations = Float64[]
        final_objs = Float64[]
        
        # Collect from constraint problems
        for i in 1:n_guesses, j in 1:n_fidelities
            analysis = analyze_convergence_enhanced(constraint_metrics_by_T[method][T][i,j])
            push!(times, analysis["total_wall_time"])
            push!(memories, analysis["avg_memory_mb"])  # Using average memory
            push!(iterations, analysis["total_iterations"])
            push!(final_objs, analysis["final_objective"])
        end
        
        # Store aggregated metrics
        push!(T_data, T)
        push!(time_means, mean(times))
        push!(time_stds, std(times))
        push!(mem_means, mean(memories))
        push!(mem_stds, std(memories))
        push!(iter_means, mean(iterations))
        push!(obj_final_means, mean(filter(!isnan, final_objs)))
    end
    
    method_performance_vs_T[method] = Dict(
        "T_values" => T_data,
        "time_means" => time_means,
        "time_stds" => time_stds,
        "mem_means" => mem_means,
        "mem_stds" => mem_stds,
        "iter_means" => iter_means,
        "obj_finals" => obj_final_means
    )
end

# Analyze resource scaling patterns
println("\n=== Resource Scaling Analysis ===")

# Fit scaling models (linear regression in log space for power law)
for method in ["variational", "toggling", "universal"]
    data = method_performance_vs_T[method]
    
    # Log-transform for power law fitting: resource ~ T^α
    log_T = log.(data["T_values"])
    log_time = log.(data["time_means"])
    log_mem = log.(data["mem_means"])
    
    # Simple linear regression for scaling exponent
    X = [ones(length(log_T)) log_T]
    
    # Time scaling
    time_coeffs = X \ log_time
    time_exponent = time_coeffs[2]
    
    # Memory scaling
    mem_coeffs = X \ log_mem
    mem_exponent = mem_coeffs[2]
    
    println("\n$method:")
    println("  Time complexity: O(T^$(round(time_exponent, digits=2)))")
    println("  Memory complexity: O(T^$(round(mem_exponent, digits=2)))")
end

# Create final summary plot
fig = Figure(resolution=(1200, 400))

# Resource efficiency plot
ax = Axis(fig[1,1:2], 
    xlabel="Time Horizon T", ylabel="Resource Efficiency (Obj/Time)",
    title="Resource Efficiency Across Methods"
)

for (idx, method) in enumerate(["variational", "toggling", "universal"])
    data = method_performance_vs_T[method]
    efficiency = data["obj_finals"] ./ data["time_means"]
    lines!(ax, data["T_values"], efficiency, 
        color=method_colors[idx], linewidth=2, label=method)
end

axislegend(ax, position=:rt)
display(fig)

println("\n=== Analysis Complete ===")