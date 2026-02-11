import Pkg; Pkg.activate(".."); Pkg.instantiate();
Pkg.develop(path="../../../QuantumCollocation.jl")
using PiccoloQuantumObjects
using QuantumCollocation
using ForwardDiff
using LinearAlgebra
using SparseArrays
using Statistics
using CairoMakie
using Random
using NamedTrajectories
using JLD2
⊗ = kron;

# variational objective metric
function var_obj(
    traj::NamedTrajectory, 
    H_drives::Vector{Matrix{ComplexF64}}, 
    H_errors::Vector{Matrix{ComplexF64}}
)
    Δt = traj.Δt[1]
    T = traj.T
    varsys = VariationalQuantumSystem(H_drives, H_errors)
    Ũ⃗, ∂Ũ⃗ = variational_unitary_rollout(traj, varsys)

    U = iso_vec_to_operator(Ũ⃗[:, end])
    # First error term
    ∂U = iso_vec_to_operator(∂Ũ⃗[1][:, end])

    d = size(U, 1)
    return abs(tr((U'*∂U)'*(U'*∂U))) / ((T-1) * Δt)^2 / d
end

# toggle objective metric
function tog_obj(
    traj::NamedTrajectory, 
    H_drives::Vector{Matrix{ComplexF64}},
    H_error::Matrix{ComplexF64}
)
    T = traj.T
    Δt = get_timesteps(traj)

    sys = QuantumSystem(H_drives)
    U = iso_vec_to_operator.(eachcol(unitary_rollout(traj, sys)))
    
    # Toggle integral
    H_ti = sum(Δt[i] .* U[i]' * H_error * U[i] for i=1:T-1)

    d₁ = size(U[1], 1)
    Δt₁ = Δt[1]
    metric = norm(tr(H_ti'H_ti)) / ((T-1) * Δt₁)^2 / d₁
    return metric
end

function scatter_with_line!(ax, x, y; color, label)
    lines!(ax, x, y; color=color, linewidth=2)         # connect points
    scatter!(ax, x, y; color=color, marker=:circle, 
                markersize=8, label=label)                 # overlay dots + legend
end

# Problem parameters
for seed in 1:1
    println("============|Starting random seed $seed|============")
    var_avg_vec = [] # ZZ, Z1, Z2 average
    def_avg_vec = []
    uni_avg_vec = []

    var_full_vec = []
    def_full_vec = []
    uni_full_vec = []

    var_avg_vec_tog = [] # ZZ, Z1, Z2 average
    def_avg_vec_tog = []
    uni_avg_vec_tog = []

    var_full_vec_tog = []
    def_full_vec_tog = []
    uni_full_vec_tog = []

    var_F_vec = []
    def_F_vec = []
    uni_F_vec = []

    ä_vals = exp10.(range(-2, stop = 3, length = 10))
    Q_vals = exp10.(range(0.0, stop = 3, length = 2)) # 2 points for initial check

    traj_uni = Vector{Any}(undef, length(Q_vals))
    traj_def = Vector{Any}(undef, length(Q_vals))
    traj_var = Vector{Any}(undef, length(Q_vals))

    for (iq, Q) in enumerate(Q_vals)
        F = 0.9999 # target fidelity
        ä = ä_vals[end]
        T = 50
        Δt = 0.2
        num_iter = 350
        iSWAP = exp(1.0im * π / 4 * (PAULIS.X ⊗ PAULIS.X + PAULIS.Y ⊗ PAULIS.Y))
        U_goal = iSWAP
        a_bound = 5.0
        println("============|Q is $Q|============")

        X1 = GATES.X ⊗ GATES.I
        Y1 = GATES.Y ⊗ GATES.I
        Z1 = GATES.Z ⊗ GATES.I
        X2 = GATES.I ⊗ GATES.X
        Y2 = GATES.I ⊗ GATES.Y
        Z2 = GATES.I ⊗ GATES.Z
        XX = GATES.X ⊗ GATES.X # transversal coupling
        YY = GATES.Y ⊗ GATES.Y # transversal coupling
        ZZ = GATES.Z ⊗ GATES.Z # transverse coupling, for error comparison
        XY = GATES.X ⊗ GATES.Y
        YX = GATES.Y ⊗ GATES.X
        XZ = GATES.X ⊗ GATES.Z
        ZX = GATES.Z ⊗ GATES.X
        YZ = GATES.Y ⊗ GATES.Z
        ZY = GATES.Z ⊗ GATES.Y

        pauli_strings = [X1, Y1, Z1, X2, Y2, Z2, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ]

        H_drive = [X1, Y1, Z1, X2, Y2, Z2, (XX + YY) / 2] # single qubit controls + tunable coupling
        piccolo_opts = PiccoloOptions(verbose=false)
        pretty_print(X::AbstractMatrix) = Base.show(stdout, "text/plain", X);
        sys = QuantumSystem(H_drive)
        # Universal 
        println("UNIVERSAL")
        Random.seed!(seed)
        uni_prob = UnitaryUniversalProblem(
            sys, U_goal, T, Δt, Δt_max=Δt, Δt_min=Δt, a_bound = a_bound, dda_bound=ä;
            activate_hyperspeed=true,
            Q=0.0,
            Q_t=Q,
            piccolo_options=piccolo_opts
            )
        push!(uni_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, uni_prob.trajectory))
        solve!(uni_prob, max_iter=num_iter, print_level=5, options=IpoptOptions(eval_hessian=false))
        #solve!(uni_prob, max_iter=50, print_level=5)
        uni_F = unitary_rollout_fidelity(uni_prob.trajectory, sys)
        println("Universal fidelity is $uni_F")
        push!(uni_F_vec, uni_F)

        # Default
        println("DEFAULT")
        Random.seed!(seed)
        def = UnitarySmoothPulseProblem(sys, U_goal, T, Δt; Δt_max=Δt, Δt_min=Δt, 
                                        a_bound = a_bound, dda_bound=ä, Q_t=1.0)
        push!(def.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, def.trajectory))
        solve!(def, max_iter=num_iter, print_level=5, options=IpoptOptions(eval_hessian=false))
        #solve!(def, max_iter=50, print_level=5)
        def_F = unitary_rollout_fidelity(def.trajectory, sys)
        println("Default fidelity is $def_F")
        push!(def_F_vec, def_F)

        ∂ₑHₐ = [Z1, Z2, ZZ];

        varsys_add = VariationalQuantumSystem(
            H_drive,
            ∂ₑHₐ
        )

        var_count = length(∂ₑHₐ)

        Random.seed!(seed)
        println("VARIATIONAL")
        var_prob = UnitaryVariationalProblem(
            varsys_add, U_goal, T, Δt;
            robust_times = [[T] for i in 1:var_count],
            Δt_max = Δt,
            Δt_min = Δt,
            a_bound = a_bound,
            dda_bound = ä,
            Q=0.0,
            Q_r = Q,
            Q_s = 0.0,
            piccolo_options = PiccoloOptions(verbose=false)
        )
        push!(var_prob.constraints, FinalUnitaryFidelityConstraint(U_goal, :Ũ⃗, F, var_prob.trajectory))
        solve!(var_prob, max_iter=num_iter, print_level=5, options=IpoptOptions(eval_hessian=false))
        var_F = unitary_rollout_fidelity(var_prob.trajectory, sys)
        println("Variational fidelity is $var_F")
        push!(var_F_vec, var_F)

        traj_uni[iq] = deepcopy(uni_prob.trajectory)
        traj_def[iq] = deepcopy(def.trajectory)
        traj_var[iq] = deepcopy(var_prob.trajectory)
        
        var_susc = []
        for P in pauli_strings
            push!(var_susc, var_obj(var_prob.trajectory, H_drive, [P]))
        end
        push!(var_full_vec, var_susc)

        def_susc = []
        for P in pauli_strings
            push!(def_susc, var_obj(def.trajectory, H_drive, [P]))
        end
        push!(def_full_vec, def_susc)

        uni_susc = []
        for P in pauli_strings
            push!(uni_susc, var_obj(uni_prob.trajectory, H_drive, [P]))
        end
        push!(uni_full_vec, uni_susc)
        
        var_zz = var_obj(var_prob.trajectory, H_drive, [ZZ])
        var_z1 = var_obj(var_prob.trajectory, H_drive, [Z1])
        var_z2 = var_obj(var_prob.trajectory, H_drive, [Z2])
        var_avg = (var_zz + var_z1 + var_z2) / 3
        push!(var_avg_vec, var_avg)

        def_zz = var_obj(def.trajectory, H_drive, [ZZ])
        def_z1 = var_obj(def.trajectory, H_drive, [Z1])
        def_z2 = var_obj(def.trajectory, H_drive, [Z2])
        def_avg = (def_zz + def_z1 + def_z2) / 3
        push!(def_avg_vec, def_avg)

        uni_zz = var_obj(uni_prob.trajectory, H_drive, [ZZ])
        uni_z1 = var_obj(uni_prob.trajectory, H_drive, [Z1])
        uni_z2 = var_obj(uni_prob.trajectory, H_drive, [Z2])
        uni_avg = (uni_zz + uni_z1 + uni_z2) / 3
        push!(uni_avg_vec, uni_avg)

        var_susc_tog = []
        for P in pauli_strings
            push!(var_susc_tog, tog_obj(var_prob.trajectory, H_drive, P))
        end
        push!(var_full_vec_tog, var_susc_tog)

        def_susc_tog = []
        for P in pauli_strings
            push!(def_susc_tog, tog_obj(def.trajectory, H_drive, P))
        end
        push!(def_full_vec_tog, def_susc_tog)

        uni_susc_tog = []
        for P in pauli_strings
            push!(uni_susc_tog, tog_obj(uni_prob.trajectory, H_drive, P))
        end
        push!(uni_full_vec_tog, uni_susc_tog)

        var_zz_tog = tog_obj(var_prob.trajectory, H_drive, ZZ)
        var_z1_tog = tog_obj(var_prob.trajectory, H_drive, Z1)
        var_z2_tog = tog_obj(var_prob.trajectory, H_drive, Z2)
        var_avg_tog = (var_zz_tog + var_z1_tog + var_z2_tog) / 3
        push!(var_avg_vec_tog, var_avg_tog)

        def_zz_tog = tog_obj(def.trajectory, H_drive, ZZ)
        def_z1_tog = tog_obj(def.trajectory, H_drive, Z1)
        def_z2_tog = tog_obj(def.trajectory, H_drive, Z2)
        def_avg_tog = (def_zz_tog + def_z1_tog + def_z2_tog) / 3
        push!(def_avg_vec_tog, def_avg_tog)

        uni_zz_tog = tog_obj(uni_prob.trajectory, H_drive, ZZ)
        uni_z1_tog = tog_obj(uni_prob.trajectory, H_drive, Z1)
        uni_z2_tog = tog_obj(uni_prob.trajectory, H_drive, Z2)
        uni_avg_tog = (uni_zz_tog + uni_z1_tog + uni_z2_tog) / 3
        push!(uni_avg_vec_tog, uni_avg_tog)

    end

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1];
        xlabel = "Q",
        ylabel = "‖Ɛ(ZZ)+Ɛ(Z₁)+Ɛ(Z₂)‖",
        xscale = log10,
        yscale = log10,
        title  = "Dephasing robustness vs Q (var metric)",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, Q_vals, def_avg_vec; color=colors[1], label="Default")
    scatter_with_line!(ax, Q_vals, var_avg_vec; color=colors[2], label="Variational")
    scatter_with_line!(ax, Q_vals, uni_avg_vec; color=colors[4], label="Universal")

    axislegend(ax; position = :rt)

    using DataFrames, CSV

    jldsave("trajectories_seed_$seed.jld2";
        seed = seed,
        Q_vals = Q_vals,
        traj_uni = traj_uni,
        traj_def = traj_def,
        traj_var = traj_var
    )

    df = DataFrame(
        Q_vals = Q_vals,
        def_avg_vec = def_avg_vec,
        uni_avg_vec = uni_avg_vec,
        var_avg_vec = var_avg_vec,
        def_F_vec = def_F_vec,
        var_F_vec = var_F_vec,
        uni_F_vec = uni_F_vec,
        var_full_vec = var_full_vec,
        def_full_vec = def_full_vec,
        uni_full_vec = uni_full_vec,
        var_full_vec_tog = var_full_vec_tog,
        def_full_vec_tog = def_full_vec_tog,
        uni_full_vec_tog = uni_full_vec_tog,
        def_avg_vec_tog = def_avg_vec_tog,
        uni_avg_vec_tog = uni_avg_vec_tog,
        var_avg_vec_tog = var_avg_vec_tog,
    )

    CSV.write("Q_sweep_iswap_T50_seed_$seed.csv", df)
    save("var_Q_sweep_iswap_T50_seed_$seed.png", fig)

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1];
        xlabel = "Q",
        ylabel = "‖Ɛ(ZZ)+Ɛ(Z₁)+Ɛ(Z₂)‖",
        xscale = log10,
        yscale = log10,
        title  = "Dephasing robustness vs Q (tog metric)",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, Q_vals, def_avg_vec_tog; color=colors[1], label="Default")
    scatter_with_line!(ax, Q_vals, var_avg_vec_tog; color=colors[2], label="Variational")
    scatter_with_line!(ax, Q_vals, uni_avg_vec_tog; color=colors[4], label="Universal")

    axislegend(ax; position = :rt)
    save("tog_Q_sweep_iswap_T50_seed_$seed.png", fig)

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1];
        xlabel = "Q",
        ylabel = "‖Ɛ(ZZ)+Ɛ(Z₁)+Ɛ(Z₂)‖",
        xscale = log10,
        yscale = log10,
        title  = "Gap for universal metric",
    )

    colors = Makie.wong_colors()

    scatter_with_line!(ax, Q_vals, uni_avg_vec_tog; color=colors[3], label="Tog metric")
    scatter_with_line!(ax, Q_vals, uni_avg_vec; color=colors[2], label="Var metric")

    axislegend(ax; position = :rt)
    save("gap_$seed.png", fig)
end
