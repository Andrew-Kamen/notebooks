# =============================================================================
# plot_stretched_results.jl
#
# Plotting / analysis script for the 300 ns stretched 3-level Duffing runs.
# Models on plot_results.jl but:
#   - Auto-detects parameters from parameters.txt (g_eff = 1 MHz for stretched)
#   - Default Gaussian-square baseline uses the run's g_eff (1 MHz here)
#   - Flat-region width chosen to give iSWAP under area assumptions
#   - Generates all the standard plots: combined, default_vs_robust (linear,
#     log, 6-panel), controls_full_gate, and ipopt iteration history
#
# Usage:
#   julia plot_stretched_results.jl <run_dir>
#
# Defaults to the stretched warmstart run if no arg given.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__)
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
Pkg.develop([
    Pkg.PackageSpec(path = piccolo_path),
    Pkg.PackageSpec(path = directtrajopt_path),
])
Pkg.add(["CairoMakie", "JLD2"])
Pkg.instantiate()

using Piccolo
using LinearAlgebra
using Printf
using CairoMakie
using JLD2

# =============================================================================
# Locate run directory
# =============================================================================
const DEFAULT_RUN_DIR = "robust_iswap_detuned_1MHz_260nsmw_10nsgauss_10nsbuf_3lvl_170MHzanh_stretched_Qr1000_seed42"

run_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, DEFAULT_RUN_DIR)
@assert isdir(run_dir) "Run directory not found: $run_dir"
@assert isfile(joinpath(run_dir, "trajectory.jld2")) "trajectory.jld2 missing in $run_dir"
println("Run dir: ", abspath(run_dir))

# =============================================================================
# Load trajectory + auto-detect Hilbert dim
# =============================================================================
traj_data = load(joinpath(run_dir, "trajectory.jld2"))
traj = traj_data["traj"]

iso_dim = traj.dims[:Ũ⃗]
D_total = isqrt(iso_dim ÷ 2)
n_lvl = isqrt(D_total)
@assert D_total == n_lvl^2 "Trajectory Ũ⃗ implies D=$D_total, not a perfect square"
@printf("Detected: %d-level per qubit (D = %d)\n", n_lvl, D_total)

# =============================================================================
# Parse run parameters
# =============================================================================
function _parse_params(path)
    p = Dict{String,Float64}()
    if !isfile(path); return p; end
    for line in readlines(path)
        m = match(r"^\s*([A-Za-zη_₁₂\s\(\)·]+)\s*=\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)", line)
        if m !== nothing
            k = strip(m.captures[1])
            try; p[k] = parse(Float64, m.captures[2]); catch; end
        end
    end
    return p
end
params = _parse_params(joinpath(run_dir, "parameters.txt"))

# Pull stretched defaults; fall back to original 5ns-buf params if missing
g_eff                = get(params, "g_eff",                2π * 0.001)
σ_rise               = get(params, "σ_rise",               2.5)
buffer_flat_duration = get(params, "buffer_flat_duration", 10.0)
T_total_gate_ns      = get(params, "T_total_gate_ns",      300.0)
a_bound              = get(params, "a_bound",              2π * 0.01)
η_anh                = get(params, "η_anh",                -2π * 0.170)
F_threshold          = get(params, "F_threshold",          0.9999)
buffer_duration      = 4 * σ_rise

@printf("Parsed params:  g_eff = %.4f rad/ns (%.2f MHz)\n", g_eff, g_eff/(2π)*1e3)
@printf("                σ_rise = %.3f ns, buffer_flat = %.1f ns, total = %.1f ns\n",
    σ_rise, buffer_flat_duration, T_total_gate_ns)
@printf("                η_anh = %.4f rad/ns (%.1f MHz)\n", η_anh, η_anh/(2π)*1e3)

# =============================================================================
# Build operators (2-level or 3-level Duffing depending on detected n_lvl)
# =============================================================================
function _build_ops(n_lvl)
    if n_lvl == 2
        return (;
            XX = operator_from_string("XX"), YY = operator_from_string("YY"),
            XI = operator_from_string("XI"), YI = operator_from_string("YI"),
            IX = operator_from_string("IX"), IY = operator_from_string("IY"),
            err_ops = [
                (operator_from_string("ZI"), "ZI"),
                (operator_from_string("IZ"), "IZ"),
                (operator_from_string("ZZ"), "ZZ"),
            ],
            drift = nothing,
            D = 4,
            subspace = [1, 2, 3, 4],
        )
    else
        b = zeros(ComplexF64, n_lvl, n_lvl)
        for j in 1:n_lvl-1; b[j, j+1] = sqrt(j); end
        b3 = copy(b); bd3 = b3'
        I3   = Matrix{ComplexF64}(I, n_lvl, n_lvl)
        I_sq = Matrix{ComplexF64}(I, n_lvl^2, n_lvl^2)
        X3 = b3 + bd3; Y3 = im * (bd3 - b3); n3 = bd3 * b3
        XI = kron(X3, I3); YI = kron(Y3, I3)
        IX = kron(I3, X3); IY = kron(I3, Y3)
        XX = kron(X3, X3); YY = kron(Y3, Y3)
        nI = kron(n3, I3); In_op = kron(I3, n3); nn = nI * In_op
        H_anh = (η_anh / 2) * (nI * (nI - I_sq) + In_op * (In_op - I_sq))
        return (; XX, YY, XI, YI, IX, IY,
                  err_ops = [(nI, "n_1"), (In_op, "n_2"), (nn, "n_1*n_2")],
                  drift = H_anh,
                  D = n_lvl^2,
                  subspace = [1, 2, 4, 5])
    end
end
ops = _build_ops(n_lvl)

# Microwave detunings (gate frame on-resonance)
const Δ_mw1 = 0.0
const Δ_mw2 = 0.0

function _gate_target_4x4()
    σx = ComplexF64[0.0 1.0; 1.0 0.0]
    σy = ComplexF64[0.0 -im; im 0.0]
    return exp(-im * π/4 * (kron(σx, σx) + kron(σy, σy)))
end
U_iswap_4x4 = _gate_target_4x4()
_comp_sub(U_full) = U_full[ops.subspace, ops.subspace]
function _fid_to_iswap(U_full)
    Usub = _comp_sub(U_full)
    return abs2(tr(U_iswap_4x4' * Usub)) / 16
end

# Gate-frame Hamiltonian
function H_gate_frame(u, t)
    uX1, uY1, uX2, uY2 = u
    H = g_eff * (ops.XX + ops.YY)
    if !isnothing(ops.drift); H = H + ops.drift; end
    c1 = cos(Δ_mw1 * t); s1 = sin(Δ_mw1 * t)
    H = H + uX1 * (ops.XI * c1 + ops.YI * s1)
    H = H + uY1 * (ops.YI * c1 - ops.XI * s1)
    c2 = cos(Δ_mw2 * t); s2 = sin(Δ_mw2 * t)
    H = H + uX2 * (ops.IX * c2 + ops.IY * s2)
    H = H + uY2 * (ops.IY * c2 - ops.IX * s2)
    return H
end
drive_bounds = fill(a_bound, 4)

# Edge & MW-region areas
let
    dt_fine = 0.01
    t_rise = collect(0:dt_fine:buffer_duration)
    g_rise = g_eff .* exp.(-(t_rise .- buffer_duration).^2 ./ (2 * σ_rise^2))
    global A_gauss  = sum(g_rise) * dt_fine
    global A_buffer = g_eff * buffer_flat_duration
    global A_pre    = A_gauss + A_buffer
    global A_post   = A_pre
end
θ_goal = π/4 - A_pre - A_post
@assert θ_goal > 0
V_rise = exp(-im * A_pre  * (ops.XX + ops.YY))
V_fall = exp(-im * A_post * (ops.XX + ops.YY))
@printf("A_pre = %.5f rad, θ_goal = %.5f rad\n", A_pre, θ_goal)

# =============================================================================
# Default Gaussian-square baseline at the run's g_eff
# =============================================================================
println("\nBuilding default Gaussian-square baseline (g_eff = $(round(g_eff/(2π)*1e3,digits=2)) MHz)...")

function _gaussian_square_area(flat_ns; dt = 0.01, trunc = 4)
    buf = trunc * σ_rise
    t_rise = collect(0:dt:buf)
    g_rise = g_eff .* exp.(-(t_rise .- buf).^2 ./ (2 * σ_rise^2))
    A_rise = sum(g_rise) * dt
    return 2 * A_rise + g_eff * flat_ns
end
flat_default = let target = π/4, lo = 0.0, hi = 1000.0
    for _ in 1:120
        mid = (lo + hi) / 2
        _gaussian_square_area(mid) < target ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end
@printf("Default flat width = %.3f ns  (total default gate = %.3f ns)\n",
    flat_default, 2*buffer_duration + flat_default)

const dt_default = 0.01
T_default = buffer_duration + flat_default + buffer_duration
t_default = collect(0:dt_default:T_default)
g_default = [
    t < buffer_duration ?
        g_eff * exp(-(t - buffer_duration)^2 / (2 * σ_rise^2)) :
    t < buffer_duration + flat_default ?
        g_eff :
        g_eff * exp(-(t - buffer_duration - flat_default)^2 / (2 * σ_rise^2))
    for t in t_default
]

# Rise / fall edges for the OPTIMIZED pulse (includes buffer flat at g_eff)
gauss_ns_fine = buffer_duration
buf_ns_fine   = buffer_flat_duration
edge_total    = gauss_ns_fine + buf_ns_fine
dt_edge = 0.01

t_rise_fine = collect(0:dt_edge:edge_total)
g_rise_fine = [
    t < gauss_ns_fine ?
        g_eff * exp(-(t - gauss_ns_fine)^2 / (2 * σ_rise^2)) :
        g_eff
    for t in t_rise_fine
]
t_fall_fine = collect(0:dt_edge:edge_total)
g_fall_fine = [
    t < buf_ns_fine ?
        g_eff :
        g_eff * exp(-(t - buf_ns_fine)^2 / (2 * σ_rise^2))
    for t in t_fall_fine
]

# Edge propagation with perturbation
function simulate_edge_unitary(g_pulse, t_pulse, dt, ε_op, ε_val)
    N = length(g_pulse)
    T_ns = t_pulse[end]
    function H_edge_err(u, t)
        idx = clamp(round(Int, t / dt) + 1, 1, N)
        H = g_pulse[idx] * (ops.XX + ops.YY) + ε_val * ε_op
        if !isnothing(ops.drift); H = H + ops.drift; end
        return H
    end
    edge_sys = QuantumSystem(H_edge_err, Float64[]; time_dependent = true)
    ts_prop = collect(LinRange(0.0, T_ns, min(N, 2000)))
    dummy_pulse = LinearSplinePulse(zeros(0, length(ts_prop)), ts_prop)
    qtraj = UnitaryTrajectory(edge_sys, dummy_pulse, Matrix{ComplexF64}(I, ops.D, ops.D))
    return qtraj(T_ns)
end

function simulate_robust_full(traj, ε_op, ε_val)
    V_rise_err = simulate_edge_unitary(g_rise_fine, t_rise_fine, dt_edge, ε_op, ε_val)
    V_fall_err = simulate_edge_unitary(g_fall_fine, t_fall_fine, dt_edge, ε_op, ε_val)
    ts  = vec(traj[:t]); us = traj[:u]; dus = traj[:du]
    function H_err(u, t); H_gate_frame(u, t) + ε_val * ε_op; end
    err_sys   = QuantumSystem(H_err, drive_bounds; time_dependent = true)
    nlp_pulse = CubicSplinePulse(us, dus, ts)
    qtraj     = UnitaryTrajectory(err_sys, nlp_pulse, Matrix{ComplexF64}(I, ops.D, ops.D))
    U_flat    = qtraj(ts[end])
    return _fid_to_iswap(V_fall_err * U_flat * V_rise_err)
end

function simulate_default_full(ε_op, ε_val)
    function H_def(u, t)
        idx = clamp(round(Int, t / dt_default) + 1, 1, length(g_default))
        H = g_default[idx] * (ops.XX + ops.YY) + ε_val * ε_op
        if !isnothing(ops.drift); H = H + ops.drift; end
        return H
    end
    def_sys = QuantumSystem(H_def, Float64[]; time_dependent = true)
    ts_prop = collect(LinRange(0.0, T_default, 2000))
    dummy_pulse = LinearSplinePulse(zeros(0, length(ts_prop)), ts_prop)
    qtraj = UnitaryTrajectory(def_sys, dummy_pulse, Matrix{ComplexF64}(I, ops.D, ops.D))
    return _fid_to_iswap(qtraj(T_default))
end

# =============================================================================
# ε-sweep both pulses
# =============================================================================
εs = collect(range(-0.02, 0.02, length = 101))
F_robust  = Dict{String,Vector{Float64}}()
F_default = Dict{String,Vector{Float64}}()

println("\nSweeping ε ∈ [-0.02, 0.02] (101 points) per error channel...")
for (err_op, err_name) in ops.err_ops
    print("  $err_name: robust...")
    F_robust[err_name]  = [simulate_robust_full(traj, err_op, ε) for ε in εs]
    print(" default...")
    F_default[err_name] = [simulate_default_full(err_op, ε)      for ε in εs]
    println(" done.")
end

# Save fidelity CSVs
for (_, err_name) in ops.err_ops
    open(joinpath(run_dir, "fidelity_$(err_name).csv"), "w") do io
        println(io, "epsilon,F_robust,F_default")
        for j in eachindex(εs)
            @printf(io, "%.10g,%.10g,%.10g\n",
                εs[j], F_robust[err_name][j], F_default[err_name][j])
        end
    end
end
println("Wrote fidelity_*.csv")

# =============================================================================
# Save full-gate pulse on absolute time axis
# =============================================================================
let
    dt = 0.05
    t_axis = collect(0:dt:T_total_gate_ns)
    g_env = similar(t_axis)
    mw    = zeros(4, length(t_axis))

    mw_start_t = buffer_duration + buffer_flat_duration
    mw_end_t   = mw_start_t + traj[:t][end]

    pulse_fine = CubicSplinePulse(traj[:u], traj[:du], vec(traj[:t]))
    for (i, t) in enumerate(t_axis)
        if t < buffer_duration
            g_env[i] = g_eff * exp(-(t - buffer_duration)^2 / (2 * σ_rise^2))
        elseif t < mw_end_t + buffer_flat_duration
            g_env[i] = g_eff
        else
            t_post = t - (mw_end_t + buffer_flat_duration)
            g_env[i] = g_eff * exp(-t_post^2 / (2 * σ_rise^2))
        end
        if mw_start_t <= t <= mw_end_t
            sampled = sample(pulse_fine, [t - mw_start_t])
            mw[:, i] = vec(sampled)
        end
    end

    open(joinpath(run_dir, "pulse_full_gate.csv"), "w") do io
        println(io, "time_ns,g_eff,u_X1,u_Y1,u_X2,u_Y2")
        for j in eachindex(t_axis)
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                t_axis[j], g_env[j], mw[1,j], mw[2,j], mw[3,j], mw[4,j])
        end
    end
end
println("Wrote pulse_full_gate.csv")

# =============================================================================
# Pulse layout helpers
# =============================================================================
function _spline_fine(traj)
    ts = vec(traj[:t]); us = traj[:u]; dus = traj[:du]
    p = CubicSplinePulse(us, dus, ts)
    ts_f = collect(LinRange(ts[1], ts[end], 2000))
    us_f = sample(p, ts_f)
    return ts, us, ts_f, us_f
end
ts_knots, us_knots, ts_fine, us_fine = _spline_fine(traj)
T_flat_opt = ts_fine[end]
mw_start = buffer_duration + buffer_flat_duration
mw_end   = mw_start + T_flat_opt
gate_end = mw_end + buffer_flat_duration + buffer_duration

dt_plot = 0.05
t_gauss_rise = collect(0:dt_plot:buffer_duration)
t_buf_pre    = collect(0:dt_plot:buffer_flat_duration) .+ buffer_duration
t_mw         = collect(0:dt_plot:T_flat_opt) .+ mw_start
t_buf_post   = collect(0:dt_plot:buffer_flat_duration) .+ mw_end
t_gauss_fall = collect(0:dt_plot:buffer_duration) .+ mw_end .+ buffer_flat_duration
g_gauss_rise = g_eff .* exp.(-(t_gauss_rise .- buffer_duration).^2 ./ (2 * σ_rise^2))
g_buf_pre    = fill(g_eff, length(t_buf_pre))
g_mw         = fill(g_eff, length(t_mw))
g_buf_post   = fill(g_eff, length(t_buf_post))
g_gauss_fall = g_eff .* exp.(-(t_gauss_fall .- mw_end .- buffer_flat_duration).^2 ./ (2 * σ_rise^2))

# =============================================================================
# Plots
# =============================================================================
mkpath(joinpath(run_dir, "figs"))
err_colors = [:red, :blue, :green]
mw_labels = ["u_X1", "u_Y1", "u_X2", "u_Y2"]
mw_colors = [:crimson, :orange, :forestgreen, :purple]

# (1) F vs ε — linear
fig = Figure(fontsize = 22, size = (1200, 400))
for (col, ((err_op, err_name), ecol)) in enumerate(zip(ops.err_ops, err_colors))
    ax = Axis(fig[1, col], xlabel = "ε (MHz)", ylabel = "Fidelity", title = "$err_name error")
    lines!(ax, εs .* 1000 ./ (2π), F_default[err_name];
        color = ecol, linewidth = 2, linestyle = :dash, label = "Default (GS)")
    lines!(ax, εs .* 1000 ./ (2π), F_robust[err_name];
        color = ecol, linewidth = 2, label = "Robust")
    hlines!(ax, [F_threshold]; linestyle = :dot, color = :black, linewidth = 0.5)
    if col == 1; axislegend(ax, position = :lb); end
end
save(joinpath(run_dir, "figs", "default_vs_robust.png"), fig)
println("Wrote figs/default_vs_robust.png")

# (2) 1-F vs ε — log scale
fig = Figure(fontsize = 22, size = (1200, 400))
for (col, ((err_op, err_name), ecol)) in enumerate(zip(ops.err_ops, err_colors))
    ax = Axis(fig[1, col], xlabel = "ε (MHz)", ylabel = "1 - Fidelity",
        title = "$err_name error", yscale = log10)
    lines!(ax, εs .* 1000 ./ (2π), max.(1 .- F_default[err_name], 1e-16);
        color = :black, linewidth = 2, linestyle = :dash, label = "Default")
    lines!(ax, εs .* 1000 ./ (2π), max.(1 .- F_robust[err_name], 1e-16);
        color = :black, linewidth = 2, label = "Robust")
    ylims!(ax, 1e-4, 1.0)
    if col == 1; axislegend(ax, position = :lb); end
end
save(joinpath(run_dir, "figs", "default_vs_robust_log.png"), fig)
println("Wrote figs/default_vs_robust_log.png")

# (3) 6-panel (linear top, log bottom)
fig = Figure(fontsize = 22, size = (1200, 800))
for (col, ((err_op, err_name), ecol)) in enumerate(zip(ops.err_ops, err_colors))
    ax_lin = Axis(fig[1, col], xlabel = "ε (MHz)", ylabel = "Fidelity", title = "$err_name error")
    lines!(ax_lin, εs .* 1000 ./ (2π), F_default[err_name];
        color = ecol, linewidth = 2, linestyle = :dash, label = "Default (GS)")
    lines!(ax_lin, εs .* 1000 ./ (2π), F_robust[err_name];
        color = ecol, linewidth = 2, label = "Robust")
    hlines!(ax_lin, [F_threshold]; linestyle = :dot, color = :black, linewidth = 0.5)
    if col == 1; axislegend(ax_lin, position = :lb); end

    ax_log = Axis(fig[2, col], xlabel = "ε (MHz)", ylabel = "1 − Fidelity", yscale = log10)
    lines!(ax_log, εs .* 1000 ./ (2π), max.(1 .- F_default[err_name], 1e-16);
        color = ecol, linewidth = 2, linestyle = :dash, label = "Default (GS)")
    lines!(ax_log, εs .* 1000 ./ (2π), max.(1 .- F_robust[err_name], 1e-16);
        color = ecol, linewidth = 2, label = "Robust")
    if col == 1; axislegend(ax_log, position = :lb); end
end
for col in 1:length(ops.err_ops)
    linkxaxes!(contents(fig[1, col])[1], contents(fig[2, col])[1])
end
save(joinpath(run_dir, "figs", "default_vs_robust_6panel.png"), fig)
println("Wrote figs/default_vs_robust_6panel.png")

# (4) Pulse layout — g(t) + microwaves on absolute time axis
fig = Figure(size = (1500, 800), fontsize = 26)
ax1 = Axis(fig[1, 1], ylabel = "g_eff (rad/ns)", title = "Pulse layout (full gate)")
lines!(ax1, t_gauss_rise, g_gauss_rise; color = :black, linewidth = 2)
lines!(ax1, t_buf_pre,    g_buf_pre;    color = :black, linewidth = 2)
lines!(ax1, t_mw,         g_mw;         color = :black, linewidth = 2)
lines!(ax1, t_buf_post,   g_buf_post;   color = :black, linewidth = 2)
lines!(ax1, t_gauss_fall, g_gauss_fall; color = :black, linewidth = 2)
vspan!(ax1, [0],                             [buffer_duration];                              color = (:gray, 0.1))
vspan!(ax1, [buffer_duration],               [mw_start];                                     color = (:goldenrod, 0.15))
vspan!(ax1, [mw_end],                        [mw_end + buffer_flat_duration];                color = (:goldenrod, 0.15))
vspan!(ax1, [mw_end + buffer_flat_duration], [gate_end];                                     color = (:gray, 0.1))

ax2 = Axis(fig[2, 1], xlabel = "t (ns)", ylabel = "u (rad/ns)")
for (i, (lbl, c)) in enumerate(zip(mw_labels, mw_colors))
    lines!(ax2, ts_fine .+ mw_start, vec(us_fine[i, :]); label = lbl, color = c, linewidth = 2)
    scatter!(ax2, ts_knots .+ mw_start, vec(us_knots[i, :]); color = :black, markersize = 6)
end
hlines!(ax2, [a_bound, -a_bound]; linestyle = :dash, color = :gray)
vspan!(ax2, [0],                             [buffer_duration];                              color = (:gray, 0.1))
vspan!(ax2, [buffer_duration],               [mw_start];                                     color = (:goldenrod, 0.15))
vspan!(ax2, [mw_end],                        [mw_end + buffer_flat_duration];                color = (:goldenrod, 0.15))
vspan!(ax2, [mw_end + buffer_flat_duration], [gate_end];                                     color = (:gray, 0.1))
linkxaxes!(ax1, ax2)
axislegend(ax2, position = :rb)
save(joinpath(run_dir, "figs", "controls_full_gate.png"), fig)
println("Wrote figs/controls_full_gate.png")

# (5) Combined money plot
FS = 30
fig = Figure(size = (1500, 1200), fontsize = FS)
for (col, ((err_op, err_name), ecol)) in enumerate(zip(ops.err_ops, err_colors))
    ax = Axis(fig[1, col], xlabel = "ε (MHz)", ylabel = "1 - Fidelity",
        title = "$err_name error", yscale = log10)
    lines!(ax, εs .* 1000 ./ (2π), max.(1 .- F_default[err_name], 1e-16);
        color = :black, linewidth = 2, linestyle = :dash, label = "Default")
    lines!(ax, εs .* 1000 ./ (2π), max.(1 .- F_robust[err_name], 1e-16);
        color = :black, linewidth = 2, label = "Robust")
    ylims!(ax, 1e-4, 1.0)
    if col == 1; axislegend(ax, position = :lb, labelsize = FS); end
end
ax_g = Axis(fig[2, 1:3], ylabel = "g_eff (rad/ns)")
lines!(ax_g, t_gauss_rise, g_gauss_rise; color = :black, linewidth = 2)
lines!(ax_g, t_buf_pre,    g_buf_pre;    color = :black, linewidth = 2)
lines!(ax_g, t_mw,         g_mw;         color = :black, linewidth = 2)
lines!(ax_g, t_buf_post,   g_buf_post;   color = :black, linewidth = 2)
lines!(ax_g, t_gauss_fall, g_gauss_fall; color = :black, linewidth = 2)
vspan!(ax_g, [0],                             [buffer_duration];                              color = (:gray, 0.1))
vspan!(ax_g, [buffer_duration],               [mw_start];                                     color = (:goldenrod, 0.15))
vspan!(ax_g, [mw_end],                        [mw_end + buffer_flat_duration];                color = (:goldenrod, 0.15))
vspan!(ax_g, [mw_end + buffer_flat_duration], [gate_end];                                     color = (:gray, 0.1))
ax_mw = Axis(fig[3, 1:3], xlabel = "t (ns)", ylabel = "u (rad/ns)")
for (i, (lbl, c)) in enumerate(zip(mw_labels, mw_colors))
    lines!(ax_mw, ts_fine .+ mw_start, vec(us_fine[i, :]); label = lbl, color = c, linewidth = 2)
    scatter!(ax_mw, ts_knots .+ mw_start, vec(us_knots[i, :]); color = :black, markersize = 6)
end
hlines!(ax_mw, [a_bound, -a_bound]; linestyle = :dash, color = :gray)
vspan!(ax_mw, [0],                             [buffer_duration];                              color = (:gray, 0.1))
vspan!(ax_mw, [buffer_duration],               [mw_start];                                     color = (:goldenrod, 0.15))
vspan!(ax_mw, [mw_end],                        [mw_end + buffer_flat_duration];                color = (:goldenrod, 0.15))
vspan!(ax_mw, [mw_end + buffer_flat_duration], [gate_end];                                     color = (:gray, 0.1))
linkxaxes!(ax_g, ax_mw)
axislegend(ax_mw, position = :rb, labelsize = FS)
save(joinpath(run_dir, "figs", "combined.png"), fig)
println("Wrote figs/combined.png")

# (6) Ipopt iteration history (if log present)
let
    log_candidates = filter(f -> startswith(f, "ipopt_") && endswith(f, ".log"),
        readdir(@__DIR__))
    matching = filter(f -> occursin(basename(run_dir)[length("robust_iswap_detuned_")+1:end], f),
        log_candidates)
    if !isempty(matching)
        log_path = joinpath(@__DIR__, first(matching))
        iters = Int[]; objs = Float64[]; inf_prs = Float64[]; inf_dus = Float64[]
        open(log_path, "r") do io
            for line in eachline(io)
                stripped = lstrip(line)
                startswith(stripped, "iter ") && continue
                isempty(stripped) && continue
                toks = split(stripped)
                length(toks) < 4 && continue
                i = tryparse(Int, toks[1]); i === nothing && continue
                obj = tryparse(Float64, toks[2]); obj === nothing && continue
                ip  = tryparse(Float64, toks[3]); ip === nothing && continue
                id  = tryparse(Float64, toks[4]); id === nothing && continue
                push!(iters, i); push!(objs, obj); push!(inf_prs, ip); push!(inf_dus, id)
            end
        end
        if !isempty(iters)
            fig = Figure(size = (1100, 800), fontsize = 18)
            ax1 = Axis(fig[1, 1], ylabel = "objective", yscale = log10,
                title = "Ipopt iteration history — $(basename(run_dir))")
            lines!(ax1, iters, max.(objs, 1e-12); color = :crimson, linewidth = 2)
            hidexdecorations!(ax1, grid = false)
            ax2 = Axis(fig[2, 1], ylabel = "inf_pr", yscale = log10)
            lines!(ax2, iters, max.(inf_prs, 1e-12); color = :forestgreen, linewidth = 2)
            hlines!(ax2, [1e-8]; color = :black, linestyle = :dash, linewidth = 1.5)
            hidexdecorations!(ax2, grid = false)
            ax3 = Axis(fig[3, 1], xlabel = "iter", ylabel = "inf_du", yscale = log10)
            lines!(ax3, iters, max.(inf_dus, 1e-12); color = :purple, linewidth = 2)
            linkxaxes!(ax1, ax2, ax3)
            save(joinpath(run_dir, "figs", "ipopt_iter_history.png"), fig)
            println("Wrote figs/ipopt_iter_history.png (parsed $(length(iters)) iters from $log_path)")
        end
    else
        println("(No matching ipopt_*.log found for iteration-history plot)")
    end
end

println("\nAll outputs saved under: ", abspath(run_dir))
