import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate();
Pkg.develop(path="../../QuantumCollocation.jl")
using PiccoloQuantumObjects
using QuantumCollocation
using ForwardDiff
using LinearAlgebra
using Plots
using SparseArrays
using Statistics
using CairoMakie
using Random
using NamedTrajectories
using CairoMakie
const CM = CairoMakie

# Problem parameters
F = 0.9999
num_iter = 500

function SpaceCurve(traj::NamedTrajectory, U_goal::AbstractMatrix{<:Number}, H_err::AbstractMatrix{<:Number})
    T = traj.T
    first_order_terms = Vector{Matrix{ComplexF64}}(undef, T)
    first_order_integral = zeros(ComplexF64, size(U_goal))

    for i in 1:T
        U = iso_vec_to_operator(traj.Ũ⃗[:, i])
        first_order_integral += U' * H_err * U
        first_order_terms[i] = first_order_integral
    end
    d = size(U_goal)[1]
    space_curve = [[real(tr(PAULIS.X * first_order_terms[t] / (d * T))),
                    real(tr(PAULIS.Y * first_order_terms[t] / (d * T))),
                    real(tr(PAULIS.Z * first_order_terms[t] / (d * T)))] for t in 1:T] 
    return space_curve
end

# helper to avoid duplicate legend entries:
function scatter_with_line!(ax, x, y; color, label)
    CM.lines!(ax, x, y; color=color, linewidth=2)         # connect points
    CM.scatter!(ax, x, y; color=color, marker=:circle, 
                markersize=8, label=label)                 # overlay dots + legend
end

for idx in 1:1
    Random.seed!(10*idx)
    T = 40
    Δt = 0.2
    U_goal = GATES.H
    H_drive = [PAULIS.X, PAULIS.Y, PAULIS.Z]
    piccolo_opts = PiccoloOptions(verbose=false)
    pretty_print(X::AbstractMatrix) = Base.show(stdout, "text/plain", X);
    sys = QuantumSystem(H_drive)

    norm_G_var = []
    norm_G_uni = []
    norm_G_def = []
    norm_G_add = []
    norm_G_var_x = []
    norm_G_var_y = []
    norm_G_var_z = []
    norm_G_uni_x = []
    norm_G_uni_y = []
    norm_G_uni_z = []
    norm_G_def_x = []
    norm_G_def_y = []
    norm_G_def_z = []
    norm_G_add_x = []
    norm_G_add_y = []
    norm_G_add_z = []


    ä_vals = exp10.(range(-3, stop = 1, length = 30))
    for ä in ä_vals
        warm_uni_prob = UnitaryUniversalProblem(
            sys, U_goal, T, Δt;
            activate_hyperspeed=true, dda_bound=ä, 
            Q=1.0,
            piccolo_options=piccolo_opts
            )

        solve!(warm_uni_prob, max_iter=100, print_level=5, options=IpoptOptions(eval_hessian=false))

        t_uni_prob = UnitaryUniversalProblem(
            sys, U_goal, T, Δt;
            activate_hyperspeed=true, dda_bound=ä, 
            Q=0.0,
            piccolo_options=piccolo_opts
            )

        push!(t_uni_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, t_uni_prob.trajectory))

        solve!(t_uni_prob, max_iter=num_iter, print_level=5, options=IpoptOptions(eval_hessian=false))
        solve!(t_uni_prob, max_iter=20, print_level=5)
        t_uni_G_x = norm(SpaceCurve(t_uni_prob.trajectory, U_goal, PAULIS.X)[end])
        t_uni_G_y = norm(SpaceCurve(t_uni_prob.trajectory, U_goal, PAULIS.Y)[end])
        t_uni_G_z = norm(SpaceCurve(t_uni_prob.trajectory, U_goal, PAULIS.Z)[end])

        t_uni_G_avg = (t_uni_G_x + t_uni_G_y + t_uni_G_z) / 3
        push!(norm_G_uni, t_uni_G_avg)
        push!(norm_G_uni_x, t_uni_G_x)
        push!(norm_G_uni_y, t_uni_G_y)
        push!(norm_G_uni_z, t_uni_G_z)

        #Default
        def = UnitarySmoothPulseProblem(sys, U_goal, T, Δt, dda_bound=ä; Q_t=1.0)

        push!(def.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, def.trajectory))

        # solve!(def, max_iter=500, print_level=1, options=IpoptOptions(eval_hessian=false))
        solve!(def, max_iter=num_iter, print_level=5, options=IpoptOptions(eval_hessian=false))
        #solve!(def, max_iter=20, print_level=5)
        def_G_x = norm(SpaceCurve(def.trajectory, U_goal, PAULIS.X)[end])
        def_G_y = norm(SpaceCurve(def.trajectory, U_goal, PAULIS.Y)[end])
        def_G_z = norm(SpaceCurve(def.trajectory, U_goal, PAULIS.Z)[end])

        def_G_avg = (def_G_x + def_G_y + def_G_z) / 3
        push!(norm_G_def, def_G_avg)
        push!(norm_G_def_x, def_G_x)
        push!(norm_G_def_y, def_G_y)
        push!(norm_G_def_z, def_G_z)

        #Adjoint
        ∂ₑHₐ = PAULIS.X
        varsys_add = VariationalQuantumSystem(
            H_drive,
            [PAULIS.X, PAULIS.Y, PAULIS.Z],
        )

        varadd_prob = UnitaryVariationalProblem(
                varsys_add, U_goal, T, Δt;
                robust_times=[[T],[T],[T]],
                dda_bound = ä,
                Q=0.0,
                Q_s=0.0,
                Q_r=1.0,
                piccolo_options=piccolo_opts
            )
        
        push!(varadd_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, varadd_prob.trajectory))

        solve!(varadd_prob, max_iter=num_iter, print_level=5, options=IpoptOptions(eval_hessian=false))
        #solve!(varadd_prob, max_iter=100, print_level=5)
        varadd_G_x = norm(SpaceCurve(varadd_prob.trajectory, U_goal, PAULIS.X)[end])
        varadd_G_y = norm(SpaceCurve(varadd_prob.trajectory, U_goal, PAULIS.Y)[end])
        varadd_G_z = norm(SpaceCurve(varadd_prob.trajectory, U_goal, PAULIS.Z)[end])
        varadd_G_avg = (varadd_G_x + varadd_G_y + varadd_G_z) / 3
        push!(norm_G_var, varadd_G_avg)
        push!(norm_G_var_x, varadd_G_x)
        push!(norm_G_var_y, varadd_G_y)
        push!(norm_G_var_z, varadd_G_z)

        # Toggling
        Hₑ_add = a -> [PAULIS.X, PAULIS.Y, PAULIS.Z]
        add_prob = UnitarySmoothPulseProblem(
                    sys, U_goal, T, Δt;
                    dda_bound=ä,
                    piccolo_options=piccolo_opts,
                    activate_rob_loss=true,
                    H_err=Hₑ_add,
                    Q=0.0,
                    Q_t=1.0
                )

        push!(add_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, add_prob.trajectory))
        
        solve!(add_prob, max_iter=num_iter, print_level=5, options=IpoptOptions(eval_hessian=false))
        #solve!(add_prob, max_iter=20, print_level=5)
        add_G_x = norm(SpaceCurve(add_prob.trajectory, U_goal, PAULIS.X)[end])
        add_G_y = norm(SpaceCurve(add_prob.trajectory, U_goal, PAULIS.Y)[end])
        add_G_z = norm(SpaceCurve(add_prob.trajectory, U_goal, PAULIS.Z)[end])
        add_G_avg = (add_G_x + add_G_y + add_G_z) / 3
        push!(norm_G_add, add_G_avg)
        push!(norm_G_add_x, add_G_x)
        push!(norm_G_add_y, add_G_y)
        push!(norm_G_add_z, add_G_z)
        println("Iteration complete for ä = $ä")
    end

    fig = CM.Figure(size = (800, 600))
    ax = CM.Axis(fig[1, 1];
        xlabel = "ä constraint",
        ylabel = "(‖E(X)‖+‖E(Y)‖+‖E(Z)‖)/3",
        xscale = log10,
        yscale = log10,
        title  = "Universal robustness vs control acceleration",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, ä_vals, norm_G_def; color=colors[1], label="Default")
    scatter_with_line!(ax, ä_vals, norm_G_var; color=colors[2], label="Variational")
    scatter_with_line!(ax, ä_vals, norm_G_add; color=colors[3], label="Toggling")
    scatter_with_line!(ax, ä_vals, norm_G_uni; color=colors[4], label="Universal")

    CM.axislegend(ax; position = :rt)
    #display(fig)
    save("dda_constraint_uni_Q_seed_$idx.png", fig)

    fig = CM.Figure(size = (800, 600))
    ax = CM.Axis(fig[1, 1];
        xlabel = "ä constraint",
        ylabel = "‖E(X)‖",
        xscale = log10,
        yscale = log10,
        title  = "Universal robustness vs control acceleration",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, ä_vals, norm_G_def_x; color=colors[1], label="Default")
    scatter_with_line!(ax, ä_vals, norm_G_var_x; color=colors[2], label="Variational")
    scatter_with_line!(ax, ä_vals, norm_G_add_x; color=colors[3], label="Toggling")
    scatter_with_line!(ax, ä_vals, norm_G_uni_x; color=colors[4], label="Universal")

    CM.axislegend(ax; position = :rt)
    #display(fig)
    save("dda_constraint_uni_x_Q_seed_$idx.png", fig)

    fig = CM.Figure(size = (800, 600))
    ax = CM.Axis(fig[1, 1];
        xlabel = "ä constraint",
        ylabel = "‖E(Y)‖",
        xscale = log10,
        yscale = log10,
        title  = "Universal robustness vs control acceleration",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, ä_vals, norm_G_def_y; color=colors[1], label="Default")
    scatter_with_line!(ax, ä_vals, norm_G_var_y; color=colors[2], label="Variational")
    scatter_with_line!(ax, ä_vals, norm_G_add_y; color=colors[3], label="Toggling")
    scatter_with_line!(ax, ä_vals, norm_G_uni_y; color=colors[4], label="Universal")

    CM.axislegend(ax; position = :rt)
    #display(fig)
    save("dda_constraint_uni_y_Q_seed_$idx.png", fig)

    fig = CM.Figure(size = (800, 600))
    ax = CM.Axis(fig[1, 1];
        xlabel = "ä constraint",
        ylabel = "‖E(Z)‖",
        xscale = log10,
        yscale = log10,
        title  = "Universal robustness vs control acceleration",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, ä_vals, norm_G_def_z; color=colors[1], label="Default")
    scatter_with_line!(ax, ä_vals, norm_G_var_z; color=colors[2], label="Variational")
    scatter_with_line!(ax, ä_vals, norm_G_add_z; color=colors[3], label="Toggling")
    scatter_with_line!(ax, ä_vals, norm_G_uni_z; color=colors[4], label="Universal")

    CM.axislegend(ax; position = :rt)
    #display(fig)
    save("dda_constraint_uni_z_Q_seed_$idx.png", fig)
    using DataFrames, CSV

    df = DataFrame(
        ä_vals      = ä_vals,
        norm_G_uni   = norm_G_uni,
        norm_G_def   = norm_G_def,
        norm_G_add   = norm_G_add,
        norm_G_var = norm_G_var,
        norm_G_var_x = norm_G_var_x,
        norm_G_var_y = norm_G_var_y,
        norm_G_var_z = norm_G_var_z,
        norm_G_uni_x = norm_G_uni_x,
        norm_G_uni_y = norm_G_uni_y,
        norm_G_uni_z = norm_G_uni_z,
        norm_G_def_x = norm_G_def_x,
        norm_G_def_y = norm_G_def_y,
        norm_G_def_z = norm_G_def_z,
        norm_G_add_x = norm_G_add_x,
        norm_G_add_y = norm_G_add_y,
        norm_G_add_z = norm_G_add_z,
    )

    CSV.write("dda_constraint_uni_data_Q_seed_$idx.csv", df)
end