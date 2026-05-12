# =============================================================================
# plot_results.jl
#
# Post-process a saved rollout run — runs the robustness sweep (which the new
# rollout scripts deliberately skip to keep solve-time clean) and produces
# the full set of plots + CSV outputs that the original
# `robust_iswap_detuned_2MHz_150ns_5nsbuf.jl` (10kiter) script produced.
#
# Works on output dirs from EITHER:
#   - robust_iswap_detuned_2MHz_150ns_5nsbuf_rollout_1kiter.jl    (2-level Pauli)
#   - robust_iswap_detuned_2MHz_150ns_5nsbuf_3level_rollout.jl    (3-level Duffing)
#
# Auto-detects 2-level vs 3-level from the trajectory's :Ũ⃗ dimension.
#
# Usage:
#   julia plot_results.jl <run_dir>
#
# e.g.
#   julia plot_results.jl robust_iswap_detuned_2MHz_130nsmw_5nsgauss_5nsbuf_rollout_1kiter_seed42
#
# Outputs (all into <run_dir>/figs/ and <run_dir>/):
#   - controls_full_gate.png        full pulse layout (g_eff envelope + microwaves)
#   - default_vs_robust.png         F(ε) for ZI/IZ/ZZ or n̂_1/n̂_2/n̂_1·n̂_2
#   - default_vs_robust_log.png     1-F(ε) on log scale
#   - default_vs_robust_6panel.png  both rows combined
#   - combined.png                  pulse + infidelity (the "money plot")
#   - fidelity_<err>.csv            F vs ε per error channel
#   - pulse_full_gate.csv           absolute-time g_eff(t) + u(t)
# =============================================================================

import Pkg
Pkg.activate(@__DIR__)
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
Pkg.develop([
    Pkg.PackageSpec(path = piccolo_path),
    Pkg.PackageSpec(path = directtrajopt_path),
])
Pkg.instantiate()

using Piccolo
using LinearAlgebra
using Random
using Printf
using CairoMakie
using JLD2

# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
length(ARGS) ≥ 1 || error("usage: julia plot_results.jl <run_dir>")
run_dir = ARGS[1]
isdir(run_dir) || error("not a directory: $run_dir")
isfile(joinpath(run_dir, "trajectory.jld2")) || error("no trajectory.jld2 in $run_dir")
mkpath(joinpath(run_dir, "figs"))
println("Processing: $run_dir")

# -----------------------------------------------------------------------------
# Load trajectory + detect level count
# -----------------------------------------------------------------------------
traj_data = load(joinpath(run_dir, "trajectory.jld2"))
traj = traj_data["traj"]

iso_dim = traj.dims[:Ũ⃗]           # 2·D² where D is Hilbert dim
D_total = isqrt(iso_dim ÷ 2)
n_lvl = isqrt(D_total)
@assert D_total == n_lvl^2 "Trajectory Ũ⃗ dim implies 2-qubit Hilbert with D=$D_total but $D_total is not a perfect square"
@printf("Detected: %d-level per qubit (D = %d, iso_dim = %d)\n", n_lvl, D_total, iso_dim)

# -----------------------------------------------------------------------------
# Parameters (parse parameters.txt for the constants we need)
# -----------------------------------------------------------------------------
function _parse_params(path)
    p = Dict{String,Float64}()
    if !isfile(path); return p; end
    for line in readlines(path)
        m = match(r"^\s*([A-Za-zη_₁₂\s\(\)]+)\s*=\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)", line)
        if m !== nothing
            k = strip(m.captures[1])
            try; p[k] = parse(Float64, m.captures[2]); catch; end
        end
    end
    return p
end
params = _parse_params(joinpath(run_dir, "parameters.txt"))

# Defaults match the 5nsbuf layout used by both 2-level and 3-level scripts
g_eff                = get(params, "g_eff",                2π * 0.002)
σ_rise               = get(params, "σ_rise",               1.25)
buffer_flat_duration = get(params, "buffer_flat_duration", 5.0)
T_total_gate_ns      = get(params, "T_total_gate_ns",      150.0)
a_bound              = get(params, "a_bound",              2π * 0.01)
η_anh                = get(params, "η_anh",                -2π * 0.170)
F_threshold          = get(params, "F_threshold",          0.9999)

buffer_duration = 4 * σ_rise

# -----------------------------------------------------------------------------
# Build operators + Hamiltonian for 2-level (Pauli) or 3-level (Duffing)
# -----------------------------------------------------------------------------
function _build_ops(n_lvl)
    if n_lvl == 2
        XX = operator_from_string("XX")
        YY = operator_from_string("YY")
        XI = operator_from_string("XI")
        YI = operator_from_string("YI")
        IX = operator_from_string("IX")
        IY = operator_from_string("IY")
        ZI = operator_from_string("ZI")
        IZ = operator_from_string("IZ")
        ZZ = operator_from_string("ZZ")
        return (; XX, YY, XI, YI, IX, IY,
                  err_ops = [(ZI, "ZI"), (IZ, "IZ"), (ZZ, "ZZ")],
                  drift = nothing,    # no anharmonicity at 2-level
                  D = 4,
                  subspace = [1, 2, 3, 4])
    else
        b = zeros(ComplexF64, n_lvl, n_lvl)
        for j in 1:n_lvl-1; b[j, j+1] = sqrt(j); end
        b3, bd3 = copy(b), b'
        I3 = Matrix{ComplexF64}(I, n_lvl, n_lvl)
        I_sq = Matrix{ComplexF64}(I, n_lvl^2, n_lvl^2)
        X3 = b3 + bd3
        Y3 = im * (bd3 - b3)
        n3 = bd3 * b3
        XI = kron(X3, I3); YI = kron(Y3, I3)
        IX = kron(I3, X3); IY = kron(I3, Y3)
        XX = kron(X3, X3); YY = kron(Y3, Y3)
        nI = kron(n3, I3); In_op = kron(I3, n3); nn = nI * In_op
        H_anh = (η_anh / 2) * (nI * (nI - I_sq) + In_op * (In_op - I_sq))
        return (; XX, YY, XI, YI, IX, IY,
                  err_ops = [(nI, "n_1"), (In_op, "n_2"), (nn, "n_1*n_2")],
                  drift = H_anh,
                  D = n_lvl^2,
                  subspace = [1, 2, 4, 5])   # comp subspace for 3-level 2-qubit
    end
end

ops = _build_ops(n_lvl)

# Microwave detunings (gate frame on-resonance)
const Δ_mw1 = 0.0
const Δ_mw2 = 0.0

# Gate target — same for both: π/4 rotation of (XX+YY) in computational subspace
function _gate_target_4x4()
    σx = ComplexF64[0.0 1.0; 1.0 0.0]
    σy = ComplexF64[0.0 -im; im 0.0]
    return exp(-im * π/4 * (kron(σx, σx) + kron(σy, σy)))
end
U_iswap_4x4 = _gate_target_4x4()

# Embed/ extract for comp subspace fidelity
_comp_sub(U_full) = U_full[ops.subspace, ops.subspace]

function _fid_to_iswap(U_full)
    Usub = _comp_sub(U_full)
    return abs2(tr(U_iswap_4x4' * Usub)) / 16
end

# -----------------------------------------------------------------------------
# Gate-frame Hamiltonian (same shape for both 2-level and 3-level — only the
# operator types differ; drift gets the anharmonicity when present)
# -----------------------------------------------------------------------------
function H_gate_frame(u, t)
    uX1, uY1, uX2, uY2 = u
    H = g_eff * (ops.XX + ops.YY)
    if !isnothing(ops.drift)
        H = H + ops.drift
    end
    c1 = cos(Δ_mw1 * t); s1 = sin(Δ_mw1 * t)
    H = H + uX1 * (ops.XI * c1 + ops.YI * s1)
    H = H + uY1 * (ops.YI * c1 - ops.XI * s1)
    c2 = cos(Δ_mw2 * t); s2 = sin(Δ_mw2 * t)
    H = H + uX2 * (ops.IX * c2 + ops.IY * s2)
    H = H + uY2 * (ops.IY * c2 - ops.IX * s2)
    return H
end
drive_bounds = fill(a_bound, 4)

# -----------------------------------------------------------------------------
# Recover V_rise, V_fall, U_goal — same as in the rollout scripts
# -----------------------------------------------------------------------------
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
@printf("θ_goal = %.6f rad, A_pre = %.6f rad\n", θ_goal, A_pre)

# -----------------------------------------------------------------------------
# Robustness sweep — for each error generator, sweep ε ∈ [-0.02, 0.02] and
# compute F(U_full(ε), iSWAP) for (a) the optimized robust pulse and (b) the
# default Gaussian-square baseline.
# -----------------------------------------------------------------------------
println("\nBuilding default Gaussian-square baseline...")

function _gaussian_square_area(flat_ns; dt = 0.01, trunc = 4)
    buf = trunc * σ_rise
    t_rise = collect(0:dt:buf)
    g_rise = g_eff .* exp.(-(t_rise .- buf).^2 ./ (2 * σ_rise^2))
    A_rise = sum(g_rise) * dt
    return 2 * A_rise + g_eff * flat_ns
end
flat_default = let target = π/4, lo = 0.0, hi = 400.0
    for _ in 1:100
        mid = (lo + hi) / 2
        _gaussian_square_area(mid) < target ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end
@printf("Default Gaussian-square flat width = %.4f ns\n", flat_default)

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

# Build the rise/fall side pulses (15 ns each): Gaussian rise + flat buffer + flat buffer + Gaussian fall
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

# Edge propagation: integrate (g(t) (XX+YY) + drift + ε·err_op) over the edge,
# returning the full D×D unitary. For 3-level, the drift includes H_anh, which
# vanishes on the comp subspace but is needed for correct leakage on (n>1)
# states.
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

# Robust pulse with error: full flat-top with (H_gate_frame(u, t) + ε·err_op)
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

# Default Gaussian-square with error: no microwaves, just g(t)·(XX+YY) + drift + ε·err
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

εs = collect(range(-0.02, 0.02, length = 101))
F_robust  = Dict{String,Vector{Float64}}()
F_default = Dict{String,Vector{Float64}}()

println("\nSweeping ε ∈ [-0.02, 0.02] (101 points) per error channel...")
for (err_op, err_name) in ops.err_ops
    print("  $err_name: robust...")
    F_robust[err_name]  = [simulate_robust_full(traj, err_op, ε)  for ε in εs]
    print(" default...")
    F_default[err_name] = [simulate_default_full(err_op, ε)        for ε in εs]
    println(" done.")
end

# -----------------------------------------------------------------------------
# Save fidelity CSVs
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Save full-gate pulse on absolute time axis (for AWG-emulating sim)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
err_colors = [:red, :blue, :green]

# (1) F vs ε — linear scale
fig = Figure(fontsize = 22, size = (1200, 400))
for (col, ((err_op, err_name), ecol)) in enumerate(zip(ops.err_ops, err_colors))
    ax = Axis(fig[1, col], xlabel = "ε (MHz)", ylabel = "Fidelity",
        title = "$err_name error")
    lines!(ax, εs .* 1000 ./ (2π), F_default[err_name];
        color = ecol, linewidth = 2, linestyle = :dash, label = "Default (GS)")
    lines!(ax, εs .* 1000 ./ (2π), F_robust[err_name];
        color = ecol, linewidth = 2, label = "Robust")
    hlines!(ax, [F_threshold]; linestyle = :dot, color = :black, linewidth = 0.5)
    if col == 1; axislegend(ax, position = :lb); end
end
save(joinpath(run_dir, "figs", "default_vs_robust.png"), fig)
println("Wrote figs/default_vs_robust.png")

# (2) 1-F vs ε — log scale (infidelity)
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

# (3) 2-row variant (linear top, log bottom)
fig = Figure(fontsize = 22, size = (1200, 800))
for (col, ((err_op, err_name), ecol)) in enumerate(zip(ops.err_ops, err_colors))
    ax_lin = Axis(fig[1, col], xlabel = "ε (MHz)", ylabel = "Fidelity",
        title = "$err_name error")
    lines!(ax_lin, εs .* 1000 ./ (2π), F_default[err_name];
        color = ecol, linewidth = 2, linestyle = :dash, label = "Default (GS)")
    lines!(ax_lin, εs .* 1000 ./ (2π), F_robust[err_name];
        color = ecol, linewidth = 2, label = "Robust")
    hlines!(ax_lin, [F_threshold]; linestyle = :dot, color = :black, linewidth = 0.5)
    if col == 1; axislegend(ax_lin, position = :lb); end

    ax_log = Axis(fig[2, col], xlabel = "ε (MHz)", ylabel = "1 − Fidelity",
        yscale = log10)
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

# (4) Pulse layout — g_eff envelope + microwaves on absolute time axis
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

FS = 26
fig = Figure(size = (1500, 800), fontsize = FS)
ax1 = Axis(fig[1, 1], ylabel = "g_eff (rad/ns)", title = "Pulse layout (full gate)")
lines!(ax1, t_gauss_rise, g_gauss_rise; color = :black, linewidth = 2)
lines!(ax1, t_buf_pre,    g_buf_pre;    color = :black, linewidth = 2)
lines!(ax1, t_mw,         g_mw;         color = :black, linewidth = 2)
lines!(ax1, t_buf_post,   g_buf_post;   color = :black, linewidth = 2)
lines!(ax1, t_gauss_fall, g_gauss_fall; color = :black, linewidth = 2)
vspan!(ax1, [0],                                 [buffer_duration];                              color = (:gray, 0.1))
vspan!(ax1, [buffer_duration],                   [mw_start];                                     color = (:goldenrod, 0.15))
vspan!(ax1, [mw_end],                            [mw_end + buffer_flat_duration];                color = (:goldenrod, 0.15))
vspan!(ax1, [mw_end + buffer_flat_duration],     [gate_end];                                     color = (:gray, 0.1))

ax2 = Axis(fig[2, 1], xlabel = "t (ns)", ylabel = "u (rad/ns)")
mw_labels = ["u_X1", "u_Y1", "u_X2", "u_Y2"]
mw_colors = [:crimson, :orange, :forestgreen, :purple]
for (i, (lbl, c)) in enumerate(zip(mw_labels, mw_colors))
    lines!(ax2, ts_fine .+ mw_start, vec(us_fine[i, :]); label = lbl, color = c, linewidth = 2)
    scatter!(ax2, ts_knots .+ mw_start, vec(us_knots[i, :]); color = :black, markersize = 6)
end
hlines!(ax2, [a_bound, -a_bound]; linestyle = :dash, color = :gray)
vspan!(ax2, [0],                                 [buffer_duration];                              color = (:gray, 0.1))
vspan!(ax2, [buffer_duration],                   [mw_start];                                     color = (:goldenrod, 0.15))
vspan!(ax2, [mw_end],                            [mw_end + buffer_flat_duration];                color = (:goldenrod, 0.15))
vspan!(ax2, [mw_end + buffer_flat_duration],     [gate_end];                                     color = (:gray, 0.1))
linkxaxes!(ax1, ax2)
axislegend(ax2, position = :rb)
save(joinpath(run_dir, "figs", "controls_full_gate.png"), fig)
println("Wrote figs/controls_full_gate.png")

# (5) Combined "money plot" — infidelity sweeps + pulse layout in one figure
FS = 30
fig = Figure(size = (1500, 1200), fontsize = FS)
# Top row: infidelity panels
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
# Middle row: g_eff envelope
ax_g = Axis(fig[2, 1:3], ylabel = "g_eff (rad/ns)")
lines!(ax_g, t_gauss_rise, g_gauss_rise; color = :black, linewidth = 2)
lines!(ax_g, t_buf_pre,    g_buf_pre;    color = :black, linewidth = 2)
lines!(ax_g, t_mw,         g_mw;         color = :black, linewidth = 2)
lines!(ax_g, t_buf_post,   g_buf_post;   color = :black, linewidth = 2)
lines!(ax_g, t_gauss_fall, g_gauss_fall; color = :black, linewidth = 2)
vspan!(ax_g, [0],                                 [buffer_duration];                              color = (:gray, 0.1))
vspan!(ax_g, [buffer_duration],                   [mw_start];                                     color = (:goldenrod, 0.15))
vspan!(ax_g, [mw_end],                            [mw_end + buffer_flat_duration];                color = (:goldenrod, 0.15))
vspan!(ax_g, [mw_end + buffer_flat_duration],     [gate_end];                                     color = (:gray, 0.1))
# Bottom row: microwaves
ax_mw = Axis(fig[3, 1:3], xlabel = "t (ns)", ylabel = "u (rad/ns)")
for (i, (lbl, c)) in enumerate(zip(mw_labels, mw_colors))
    lines!(ax_mw, ts_fine .+ mw_start, vec(us_fine[i, :]); label = lbl, color = c, linewidth = 2)
    scatter!(ax_mw, ts_knots .+ mw_start, vec(us_knots[i, :]); color = :black, markersize = 6)
end
hlines!(ax_mw, [a_bound, -a_bound]; linestyle = :dash, color = :gray)
vspan!(ax_mw, [0],                                 [buffer_duration];                              color = (:gray, 0.1))
vspan!(ax_mw, [buffer_duration],                   [mw_start];                                     color = (:goldenrod, 0.15))
vspan!(ax_mw, [mw_end],                            [mw_end + buffer_flat_duration];                color = (:goldenrod, 0.15))
vspan!(ax_mw, [mw_end + buffer_flat_duration],     [gate_end];                                     color = (:gray, 0.1))
linkxaxes!(ax_g, ax_mw)
axislegend(ax_mw, position = :rb, labelsize = FS)
save(joinpath(run_dir, "figs", "combined.png"), fig)
println("Wrote figs/combined.png")

println("\nAll outputs saved under: ", abspath(run_dir))
