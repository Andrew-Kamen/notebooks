import Pkg; Pkg.activate(".."); Pkg.instantiate();
Pkg.develop(path="../../../QuantumCollocation.jl")
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
using DataFrames, CSV
const CM = CairoMakie

# Problem parameters
F = 0.9999
num_iter = 2000
hess_iter = 100

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

function var_obj(prob::DirectTrajOptProblem, H_drive::Vector{Matrix{ComplexF64}}, error_op::Matrix{ComplexF64})
    varsys = VariationalQuantumSystem(
        H_drive,
        [error_op]
    )
    traj = prob.trajectory
    T = traj.T
    Δt = traj.Δt[1]
    ww = iso_vec_to_operator(variational_unitary_rollout(prob.trajectory, varsys)[2][1][:,end])
    d = size(ww)[1]
    return norm(tr(ww'ww)) / (T * Δt)^2 / d
end

for idx in 2:10
    Random.seed!(idx)
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

    var_obj_tog_z = []
    var_obj_var_z = []
    var_obj_def_z = []
    var_obj_uni_z = []

    Q_r = 1.0
    ä_vals = exp10.(range(-3, stop = 0.301, length = 20))
    a_bound = 5.0
    for (j, ä) in enumerate(ä_vals)

        Random.seed!(idx)
        # Universal
        # baseline
        t_uni_prob = UnitaryUniversalProblem(
            sys, U_goal, T, Δt;
            activate_hyperspeed=true, a_bound=a_bound, dda_bound=ä, 
            Q=0.0,
            Q_t=Q_r,
            piccolo_options=piccolo_opts
            )
        push!(t_uni_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, t_uni_prob.trajectory))
        solve!(t_uni_prob, max_iter=num_iter, print_level=1, options=IpoptOptions(eval_hessian=false))
        solve!(t_uni_prob, max_iter=hess_iter, print_level=1)

        #solve!(t_uni_prob, max_iter=20, print_level=1)
        t_uni_G_x = norm(SpaceCurve(t_uni_prob.trajectory, U_goal, PAULIS.X)[end])
        t_uni_G_y = norm(SpaceCurve(t_uni_prob.trajectory, U_goal, PAULIS.Y)[end])
        t_uni_G_z = norm(SpaceCurve(t_uni_prob.trajectory, U_goal, PAULIS.Z)[end])

        t_uni_G_avg = (t_uni_G_x + t_uni_G_y + t_uni_G_z) / 3
        push!(norm_G_uni, t_uni_G_avg)
        push!(norm_G_uni_x, t_uni_G_x)
        push!(norm_G_uni_y, t_uni_G_y)
        push!(norm_G_uni_z, t_uni_G_z)

        Random.seed!(idx)
        #Default
        # solve!(def, max_iter=500, print_level=1, options=IpoptOptions(eval_hessian=false))
        def = UnitarySmoothPulseProblem(sys, U_goal, T, Δt, a_bound=a_bound, dda_bound=ä; Q_t=1.0)
        push!(def.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, def.trajectory))
        solve!(def, max_iter=num_iter, print_level=1, options=IpoptOptions(eval_hessian=false))
        solve!(def, max_iter=hess_iter, print_level=1)

        #solve!(def, max_iter=20, print_level=1)
        def_G_x = norm(SpaceCurve(def.trajectory, U_goal, PAULIS.X)[end])
        def_G_y = norm(SpaceCurve(def.trajectory, U_goal, PAULIS.Y)[end])
        def_G_z = norm(SpaceCurve(def.trajectory, U_goal, PAULIS.Z)[end])

        def_G_avg = (def_G_x + def_G_y + def_G_z) / 3
        push!(norm_G_def, def_G_avg)
        push!(norm_G_def_x, def_G_x)
        push!(norm_G_def_y, def_G_y)
        push!(norm_G_def_z, def_G_z)

        Random.seed!(idx)
        #Adjoint
        ∂ₑH = [PAULIS.Z]
        varsys_add = VariationalQuantumSystem(
            H_drive,
            ∂ₑH,
        )

        varadd_prob = UnitaryVariationalProblem(
                varsys_add, U_goal, T, Δt;
                robust_times=[[T]],
                a_bound=a_bound, 
                dda_bound = ä,
                Q=0.0,
                Q_s=0.0,
                Q_r=Q_r,
                piccolo_options=piccolo_opts
            )
        
        push!(varadd_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, varadd_prob.trajectory))
        solve!(varadd_prob, max_iter=num_iter, print_level=1, options=IpoptOptions(eval_hessian=false))
        solve!(varadd_prob, max_iter=hess_iter, print_level=1)
        #solve!(varadd_prob, max_iter=20, print_level=1)
        varadd_G_x = norm(SpaceCurve(varadd_prob.trajectory, U_goal, PAULIS.X)[end])
        varadd_G_y = norm(SpaceCurve(varadd_prob.trajectory, U_goal, PAULIS.Y)[end])
        varadd_G_z = norm(SpaceCurve(varadd_prob.trajectory, U_goal, PAULIS.Z)[end])
        varadd_G_avg = (varadd_G_x + varadd_G_y + varadd_G_z) / 3
        push!(norm_G_var, varadd_G_avg)
        push!(norm_G_var_x, varadd_G_x)
        push!(norm_G_var_y, varadd_G_y)
        push!(norm_G_var_z, varadd_G_z)

        # Toggling
        Random.seed!(idx)
        ∂ₑHₐ = [PAULIS.Z]
        varsys_bik = VariationalQuantumSystem(
            H_drive,
            ∂ₑHₐ
        )
        add_prob = UnitaryToggleProblem(
                    varsys_bik, U_goal, T, Δt;
                    a_bound=a_bound,
                    dda_bound=ä,
                    piccolo_options=piccolo_opts,
                    Q=0.0,
                    Q_t=Q_r
                )
        push!(add_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, add_prob.trajectory))
        solve!(add_prob, max_iter=num_iter, print_level=1, options=IpoptOptions(eval_hessian=false))
        solve!(varadd_prob, max_iter=hess_iter, print_level=1)


        #solve!(add_prob, max_iter=20, print_level=1)
        add_G_x = norm(SpaceCurve(add_prob.trajectory, U_goal, PAULIS.X)[end])
        add_G_y = norm(SpaceCurve(add_prob.trajectory, U_goal, PAULIS.Y)[end])
        add_G_z = norm(SpaceCurve(add_prob.trajectory, U_goal, PAULIS.Z)[end])
        add_G_avg = (add_G_x + add_G_y + add_G_z) / 3
        push!(norm_G_add, add_G_avg)
        push!(norm_G_add_x, add_G_x)
        push!(norm_G_add_y, add_G_y)
        push!(norm_G_add_z, add_G_z)

        error_op = GATES.Z
        push!(var_obj_def_z, var_obj(def, H_drive, error_op))
        push!(var_obj_tog_z, var_obj(add_prob, H_drive, error_op))
        push!(var_obj_uni_z, var_obj(t_uni_prob, H_drive, error_op))
        push!(var_obj_var_z, var_obj(varadd_prob, H_drive, error_op))

        println("Iteration complete for ä = $ä")
    end

    # fig = CM.Figure(size = (800, 600))
    # ax = CM.Axis(fig[1, 1];
    #     xlabel = "Qᵣ for variational",
    #     ylabel = "(‖E(X)‖+‖E(Y)‖+‖E(Z)‖)/3",
    #     xscale = log10,
    #     yscale = log10,
    #     title  = "Universal robustness vs control acceleration",
    # )

    # colors = Makie.wong_colors()

    # scatter_with_line!(ax, ä_vals, norm_G_def; color=colors[1], label="Default")
    # scatter_with_line!(ax, ä_vals, norm_G_var; color=colors[2], label="Variational")
    # scatter_with_line!(ax, ä_vals, norm_G_add; color=colors[3], label="Toggling")
    # scatter_with_line!(ax, ä_vals, norm_G_uni; color=colors[4], label="Universal")

    # CM.axislegend(ax; position = :rt)
    # #display(fig)
    # save("dda_constraint_uni_dephasing_Q_seed_$idx.png", fig)

    # fig = CM.Figure(size = (800, 600))
    # ax = CM.Axis(fig[1, 1];
    #     xlabel = "ä constraint",
    #     ylabel = "‖E(X)‖",
    #     xscale = log10,
    #     yscale = log10,
    #     title  = "Universal robustness vs control acceleration",
    # )

    # colors = Makie.wong_colors()

    # scatter_with_line!(ax, ä_vals, norm_G_def_x; color=colors[1], label="Default")
    # scatter_with_line!(ax, ä_vals, norm_G_var_x; color=colors[2], label="Variational")
    # scatter_with_line!(ax, ä_vals, norm_G_add_x; color=colors[3], label="Toggling")
    # scatter_with_line!(ax, ä_vals, norm_G_uni_x; color=colors[4], label="Universal")

    # CM.axislegend(ax; position = :rt)
    # #display(fig)
    # save("dda_constraint_uni_x_dephasing_Q_seed_$idx.png", fig)

    # fig = CM.Figure(size = (800, 600))
    # ax = CM.Axis(fig[1, 1];
    #     xlabel = "ä constraint",
    #     ylabel = "‖E(Y)‖",
    #     xscale = log10,
    #     yscale = log10,
    #     title  = "Universal robustness vs control acceleration",
    # )

    # colors = Makie.wong_colors()

    # scatter_with_line!(ax, Q_r_vals, norm_G_def_y; color=colors[1], label="Default")
    # scatter_with_line!(ax, Q_r_vals, norm_G_var_y; color=colors[2], label="Variational")
    # scatter_with_line!(ax, Q_r_vals, norm_G_add_y; color=colors[3], label="Toggling")
    # scatter_with_line!(ax, Q_r_vals, norm_G_uni_y; color=colors[4], label="Universal")

    # CM.axislegend(ax; position = :rt)
    # #display(fig)
    # save("dda_constraint_uni_y_dephasing_Q_seed_$idx.png", fig)


    fig = CM.Figure(size = (800, 600))
    ax = CM.Axis(fig[1, 1];
        xlabel = "ä values",
        ylabel = "Ɛ(Z)",
        xscale = log10,
        yscale = log10,
        title  = "Dephasing robustness vs Qᵣ, toggling objective",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, ä_vals, norm_G_def_z; color=colors[1], label="Default")
    scatter_with_line!(ax, ä_vals, norm_G_var_z; color=colors[2], label="Ɛᵥ(Z)")
    scatter_with_line!(ax, ä_vals, norm_G_add_z; color=colors[3], label="Ɛₜ(Z)")
    scatter_with_line!(ax, ä_vals, norm_G_uni_z; color=colors[4], label="Ɛᵤ(Z)")

    CM.axislegend(ax; position = :rt)
    display(fig)
    save("tog_obj_acceleration_sweep_dephasing_seed_$idx.png", fig)

    fig = CM.Figure(size = (800, 600))
    ax = CM.Axis(fig[1, 1];
        xlabel = "ä values",
        ylabel = "Ɛ(Z)",
        xscale = log10,
        yscale = log10,
        title  = "Dephasing robustness vs acceleration, variational objective",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, ä_vals, var_obj_def_z; color=colors[1], label="Default")
    scatter_with_line!(ax, ä_vals, var_obj_var_z; color=colors[2], label="Ɛᵥ(Z)")
    scatter_with_line!(ax, ä_vals, var_obj_tog_z; color=colors[3], label="Ɛₜ(Z)")
    scatter_with_line!(ax, ä_vals, var_obj_uni_z; color=colors[4], label="Ɛᵤ(Z)")

    CM.axislegend(ax; position = :rt)
    display(fig)
    save("var_obj_acceleration_sweep_dephasing_seed_$idx.png", fig)

    df = DataFrame(
        ä_vals      = ä_vals,
        norm_G_var_z = norm_G_var_z,
        norm_G_uni_z = norm_G_uni_z,
        norm_G_def_z = norm_G_def_z,
        norm_G_add_z = norm_G_add_z,
        var_obj_def_z = var_obj_def_z,
        var_obj_tog_z = var_obj_tog_z,
        var_obj_uni_z = var_obj_uni_z,
        var_obj_var_z = var_obj_var_z
    )

    CSV.write("acceleration_sweep_dephasing_seed_$idx.csv", df)
end