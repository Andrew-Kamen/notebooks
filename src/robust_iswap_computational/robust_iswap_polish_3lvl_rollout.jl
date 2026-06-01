# robust_iswap_polish_3lvl_rollout.jl
# A/B variant of robust_iswap_polish_3lvl.jl using VariationalRolloutProblem
# (indirect/rollout) instead of VariationalSplinePulseProblem (collocation).
# All other params (warm-start, constraints, robustness, n_path_samples, bounds,
# Q_r, ε_max, F_threshold, MAX_ITER, knot count, Δt, η) held identical.

import Pkg
Pkg.activate(@__DIR__)
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
Pkg.develop([
    Pkg.PackageSpec(path = piccolo_path),
    Pkg.PackageSpec(path = directtrajopt_path),
])
Pkg.add(["CairoMakie", "Ipopt", "FFTW", "JLD2"])
Pkg.instantiate()

using Piccolo, FFTW, LinearAlgebra, Printf, Random, SparseArrays, JLD2, CairoMakie
using DirectTrajOpt.Constraints: AbstractNonlinearConstraint
using DirectTrajOpt.CommonInterface

# === Notebook-matching constants ===
const T_TOTAL_NS     = 250.0
const T_PULSE_NS     = 200.0
const T_RISE         = 25.0
const T_FALL         = T_RISE + T_PULSE_NS
const T_EDGE_MARK    = 10.0
const T_FLAT_NS      = T_FALL - T_RISE - 2 * T_EDGE_MARK   # 180 ns
const N_KNOTS        = 15
const Δt_FLAT        = T_FLAT_NS / (N_KNOTS - 1)

const η_MHz          = 170.0
const g_max_MHz      = 2.0
const g_max_rad_per_ns = 2π * g_max_MHz * 1e-3
const g_flat         = g_max_rad_per_ns
const η_rad_per_ns   = 2π * η_MHz * 1e-3
const η_rad_neg      = -2π * η_MHz * 1e-3                   # transmon (η < 0)

const a_bound_MW     = 2π * 0.01
const F_THRESHOLD    = 0.9999
const Q_R            = 1e2
const ε_MAX          = 1e-2
const MAX_ITER       = 300
const PROFILE_ITERS  = 300

# Envelope filter constants (match notebook)
const B_AWG_MHz      = 250.0
const σ_f_anh_MHz    = 100.0
const σ_f_AWG_MHz    = B_AWG_MHz / sqrt(log(2))
const g_max_env      = 1.0                              # normalized envelope amplitude
const N_FINE_ENV     = 16384
const N_EDGE_GRID    = 800

# ε-sweep parameters
const εs_MHz_2q = collect(LinRange(-5.0, 5.0, 101))
const εs_rad    = 2π .* εs_MHz_2q .* 1e-3

const NOTEBOOK_RUN_TAG = "robust_2lvl_4dim_geff2MHz_200ns_eta170"
const TRAJ_PATH        = joinpath(@__DIR__, "traj_$(NOTEBOOK_RUN_TAG).jld2")
const POLISH_TAG       = "polish_3lvl_rollout_$(NOTEBOOK_RUN_TAG)"
const PROFILE_LOG      = joinpath(@__DIR__, "ipopt_$(POLISH_TAG).log")
const PROFILE_TRAJ     = joinpath(@__DIR__, "traj_$(POLISH_TAG).jld2")

# === 4-dim Paulis (for U_goal lift) ===
const XX = operator_from_string("XX")
const YY = operator_from_string("YY")

# === 9-dim operators (3 levels per qubit) ===
const I_9     = Matrix{ComplexF64}(I, 9, 9)
const I_3lvl  = Matrix{ComplexF64}(I, 3, 3)
const a_3     = ComplexF64[0 1 0; 0 0 sqrt(2); 0 0 0]
const a1_9    = kron(a_3, I_3lvl)
const a2_9    = kron(I_3lvl, a_3)
const n1_9    = a1_9' * a1_9
const n2_9    = a2_9' * a2_9
const XI_9    = a1_9 + a1_9'
const YI_9    = im * (a1_9' - a1_9)
const IX_9    = a2_9 + a2_9'
const IY_9    = im * (a2_9' - a2_9)
const exch_9  = XI_9 * IX_9 + YI_9 * IY_9
const H_anh_9 = (η_rad_neg / 2) * (n1_9 * (n1_9 - I_9) + n2_9 * (n2_9 - I_9))
const QSUB    = [1, 2, 4, 5]

# === Spectral leakage constraint (per qubit IQ pair) ===
struct SpectralLeakageConstraintIQ <: AbstractNonlinearConstraint
    name::Symbol
    i_X::Int
    i_Y::Int
    ω::Float64
    ε_max::Float64
    times_t::Vector{Float64}
    Δts::Vector{Float64}
    cosωt::Vector{Float64}
    sinωt::Vector{Float64}
    dim::Int
    equality::Bool
end

function SpectralLeakageConstraintIQ(name::Symbol, i_X::Int, i_Y::Int,
                                     ω::Float64, ε_max::Float64, traj)
    times = get_times(traj)
    Δts   = get_timesteps(traj)
    return SpectralLeakageConstraintIQ(name, i_X, i_Y, ω, ε_max,
        Vector{Float64}(times), Vector{Float64}(Δts),
        cos.(ω .* times), sin.(ω .* times),
        1, false)
end

function _ReIm_IQ(c::SpectralLeakageConstraintIQ, u::AbstractMatrix)
    Re_sum = zero(eltype(u))
    Im_sum = zero(eltype(u))
    @inbounds for k in 1:length(c.times_t)
        cos_k = c.cosωt[k]; sin_k = c.sinωt[k]; Δt_k = c.Δts[k]
        uX = u[c.i_X, k]; uY = u[c.i_Y, k]
        Re_sum += Δt_k * (uX * cos_k + uY * sin_k)
        Im_sum += Δt_k * (uY * cos_k - uX * sin_k)
    end
    return Re_sum, Im_sum
end

function CommonInterface.evaluate!(values::AbstractVector,
                                   c::SpectralLeakageConstraintIQ, traj)
    u = traj[c.name]
    Re_sum, Im_sum = _ReIm_IQ(c, u)
    values[1] = Re_sum^2 + Im_sum^2 - c.ε_max
    return nothing
end

function CommonInterface.eval_jacobian(c::SpectralLeakageConstraintIQ, traj)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂g    = spzeros(c.dim, Z_dim)
    u     = traj[c.name]
    Re_sum, Im_sum = _ReIm_IQ(c, u)
    comps = traj.components[c.name]
    @inbounds for k in 1:traj.N
        cos_k = c.cosωt[k]; sin_k = c.sinωt[k]; Δt_k = c.Δts[k]
        idxX = traj.dim * (k - 1) + comps[c.i_X]
        idxY = traj.dim * (k - 1) + comps[c.i_Y]
        ∂g[1, idxX] = 2 * Δt_k * (Re_sum * cos_k - Im_sum * sin_k)
        ∂g[1, idxY] = 2 * Δt_k * (Re_sum * sin_k + Im_sum * cos_k)
    end
    return ∂g
end

function CommonInterface.eval_hessian_of_lagrangian(c::SpectralLeakageConstraintIQ,
                                                    traj, μ::AbstractVector)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²g   = spzeros(Z_dim, Z_dim)
    comps = traj.components[c.name]
    a = zeros(Z_dim); b = zeros(Z_dim)
    @inbounds for k in 1:traj.N
        idxX = traj.dim * (k - 1) + comps[c.i_X]
        idxY = traj.dim * (k - 1) + comps[c.i_Y]
        a[idxX] = c.Δts[k] * c.cosωt[k]
        a[idxY] = c.Δts[k] * c.sinωt[k]
        b[idxX] = -c.Δts[k] * c.sinωt[k]
        b[idxY] =  c.Δts[k] * c.cosωt[k]
    end
    nzs = findall(!iszero, a) ∪ findall(!iszero, b)
    for i in nzs, j in nzs
        v = 2 * μ[1] * (a[i]*a[j] + b[i]*b[j])
        if v != 0.0
            ∂²g[i, j] = v
        end
    end
    return ∂²g
end

# === Load warm-start trajectory from notebook solve ===
println("Loading warm-start from: ", TRAJ_PATH)
@load TRAJ_PATH traj_robust U_goal U_iSWAP V_rise V_fall A_rise A_fall

# === Build 9-dim VariationalQuantumSystem with n̂ robustness generators ===
function H_fn_9(u, t)
    H = H_anh_9 + g_flat * exch_9
    H += u[1] * XI_9 + u[2] * YI_9
    H += u[3] * IX_9 + u[4] * IY_9
    return H
end

H_vars_9 = Function[
    (u, t) -> n1_9,                      # qubit-1 dephasing (transmon-physical)
    (u, t) -> n2_9,                      # qubit-2 dephasing
    (u, t) -> n1_9 * n2_9,               # cross-Kerr (physical ZZ)
]

drive_bounds = fill(a_bound_MW, 4)

varsys_9 = VariationalQuantumSystem(
    H_fn_9, H_vars_9, 4, drive_bounds; time_dependent = true)

println("9-dim VariationalQuantumSystem built.")
println("  levels = ", varsys_9.levels,
        "   n_drives = ", varsys_9.n_drives,
        "   n_vars = ", length(varsys_9.G_vars))

# === Embedded operator target ===
embedded_target = EmbeddedOperator(U_goal, QSUB, 9)
println("EmbeddedOperator: U_goal (4×4) embedded into 9-dim at QSUB = ", QSUB)

# === Warm-start spline pulse from saved trajectory ===
warm_controls = traj_robust[:u]
warm_du       = traj_robust[:du]
warm_times    = vec(traj_robust[:t])
pulse_init    = CubicSplinePulse(warm_controls, warm_du, warm_times)

@printf("Warm-start: %d knots over %.4f ns\n", size(warm_controls, 2), warm_times[end])

# === Build VariationalRolloutProblem (indirect/rollout method) ===
# All params held identical to the collocation version for a clean A/B compare.
qcp = VariationalRolloutProblem(
    varsys_9, pulse_init, embedded_target, N_KNOTS;
    Q                     = 0.0,
    Q_r                   = Q_R,
    R                     = 1e-3,
    du_bound              = Inf,
    Δt_bounds             = (Δt_FLAT, Δt_FLAT),
    dynamics_spline_order = 3,
    n_path_samples        = 3,
)

traj_3lvl = get_trajectory(qcp)

push!(qcp.prob.constraints,
    FinalUnitaryFidelityConstraint(embedded_target, :Ũ⃗, F_THRESHOLD, traj_3lvl)
)
push!(qcp.prob.constraints,
    SpectralLeakageConstraintIQ(:u, 1, 2, η_rad_per_ns, ε_MAX, traj_3lvl)
)
push!(qcp.prob.constraints,
    SpectralLeakageConstraintIQ(:u, 3, 4, η_rad_per_ns, ε_MAX, traj_3lvl)
)

# === Initial Ũ⃗ check (rollout method auto-initializes via UnitaryTrajectory) ===
# No manual var_Ũ⃗ patching needed — rollout problem has no :var_Ũ⃗ state.
let
    traj = get_trajectory(qcp)
    Ũ⃗_init = traj[:Ũ⃗][:, end]
    U_init = iso_vec_to_operator(Ũ⃗_init)
    U_sub_init = U_init[QSUB, QSUB]
    F0 = abs2(tr(U_goal' * U_sub_init)) / 16
    @printf("Initial F (warm-start, rollout-initialized): %.6f   (1−F = %.3e)\n", F0, 1 - F0)
end

# === Profile: PROFILE_ITERS iterations and extrapolate ===
@printf("\n=== Profile: %d iterations ===\n", PROFILE_ITERS)
t0 = time()
try
    solve!(qcp; max_iter = PROFILE_ITERS, print_level = 5,
        options = IpoptOptions(
            eval_hessian    = false,
            output_file     = PROFILE_LOG,
            constr_viol_tol = 1e-8,
            tol             = 1e-8,
            acceptable_tol  = 1e-8,
        ))
catch e
    e isa InterruptException || rethrow(e)
    println("Interrupted")
end
elapsed = time() - t0

@printf("\nProfile wall time:  %.2f s for %d iterations\n", elapsed, PROFILE_ITERS)
@printf("Per-iter cost:      %.2f s\n", elapsed / PROFILE_ITERS)
@printf("Estimated %d-iter:  %.1f minutes (%.1f hours)\n",
    MAX_ITER, MAX_ITER * elapsed / PROFILE_ITERS / 60,
    MAX_ITER * elapsed / PROFILE_ITERS / 3600)

# Save the partial trajectory
traj_polish = get_trajectory(qcp)
@save PROFILE_TRAJ traj_polish U_goal U_iSWAP V_rise V_fall A_rise A_fall

println("\nTrajectory saved to: $PROFILE_TRAJ")
println("IPOPT log:           $PROFILE_LOG")

# ============================================================================
# Post-solve diagnostics: control shapes (time + freq) + ε-sweep fidelity plot
# ============================================================================

# --- Build filtered g_eff envelope (for pre/post drift in full-gate sim) ---
function gaussian_lowpass(g_t::AbstractVector, dt_ns::Float64, σ_MHz::Float64)
    N      = length(g_t)
    G      = fft(g_t)
    fs_MHz = fftfreq(N, 1.0 / dt_ns) .* 1e3
    mask   = exp.(-0.5 .* (fs_MHz ./ σ_MHz).^2)
    return real.(ifft(G .* mask))
end

ts_env       = collect(LinRange(0.0, T_TOTAL_NS, N_FINE_ENV))
dt_env       = ts_env[2] - ts_env[1]
g_square     = Float64[(T_RISE ≤ t ≤ T_FALL) ? g_max_env : 0.0 for t in ts_env]
g_after_anh  = gaussian_lowpass(g_square,    dt_env, σ_f_anh_MHz)
g_after_AWG  = gaussian_lowpass(g_after_anh, dt_env, σ_f_AWG_MHz)
g_phys_env   = g_after_AWG .* g_max_rad_per_ns

T_MW_NS    = Float64(traj_polish[:t][end])
MW_END_ABS = T_RISE + T_EDGE_MARK + T_MW_NS

# --- Default envelope: Gaussian-square calibrated to area = π/4 ---
function build_default_env(W_flat::Float64)
    ts_loc = collect(LinRange(0.0, T_TOTAL_NS, N_FINE_ENV))
    dt_loc = ts_loc[2] - ts_loc[1]
    t_c    = T_TOTAL_NS / 2
    g_sq   = [(t_c - W_flat/2 ≤ t ≤ t_c + W_flat/2) ? g_max_env : 0.0 for t in ts_loc]
    g_a    = gaussian_lowpass(g_sq, dt_loc, σ_f_anh_MHz)
    g_b    = gaussian_lowpass(g_a,  dt_loc, σ_f_AWG_MHz)
    return ts_loc, g_b
end
function default_area(W_flat::Float64)
    ts_loc, g_b = build_default_env(W_flat)
    dt_loc = ts_loc[2] - ts_loc[1]
    return sum(g_b) * dt_loc * g_max_rad_per_ns
end
let lo = 0.0, hi = T_TOTAL_NS * 0.9
    for _ in 1:80
        mid = (lo + hi) / 2
        default_area(mid) < π/4 ? (lo = mid) : (hi = mid)
    end
    global W_DEF = (lo + hi) / 2
end
ts_def, g_def_norm = build_default_env(W_DEF)
g_def_phys = g_def_norm .* g_max_rad_per_ns

# --- 9-dim full-gate propagator (robust pulse with edges + flat + ε) ---
function propagate_robust_3lvl_full(ε_val::Real, ε_op_9::AbstractMatrix; N_fine = 4000)
    ts_pre = collect(LinRange(0.0, T_RISE + T_EDGE_MARK, N_EDGE_GRID))
    g_pre  = [g_after_AWG[argmin(abs.(ts_env .- t))] * g_max_rad_per_ns for t in ts_pre]
    U_pre  = Matrix{ComplexF64}(I, 9, 9)
    @inbounds for k in 1:(length(ts_pre) - 1)
        dt_k  = ts_pre[k+1] - ts_pre[k]
        g_mid = 0.5 * (g_pre[k] + g_pre[k+1])
        H = H_anh_9 + g_mid * exch_9 + ε_val * ε_op_9
        U_pre = exp(-im * dt_k * H) * U_pre
    end

    ts_flat_grid = collect(LinRange(0.0, T_MW_NS, N_fine))
    pulse_local  = CubicSplinePulse(traj_polish[:u], traj_polish[:du], vec(traj_polish[:t]))
    us_flat      = sample(pulse_local, ts_flat_grid)
    U_flat = Matrix{ComplexF64}(I, 9, 9)
    @inbounds for k in 1:(length(ts_flat_grid) - 1)
        dt_k  = ts_flat_grid[k+1] - ts_flat_grid[k]
        u_mid = 0.5 .* (us_flat[:, k] .+ us_flat[:, k+1])
        H = H_anh_9 + g_flat * exch_9 + u_mid[1] * XI_9 + u_mid[2] * YI_9 + u_mid[3] * IX_9 + u_mid[4] * IY_9 + ε_val * ε_op_9
        U_flat = exp(-im * dt_k * H) * U_flat
    end

    ts_post = collect(LinRange(MW_END_ABS, T_TOTAL_NS, N_EDGE_GRID))
    g_post  = [g_after_AWG[argmin(abs.(ts_env .- t))] * g_max_rad_per_ns for t in ts_post]
    U_post  = Matrix{ComplexF64}(I, 9, 9)
    @inbounds for k in 1:(length(ts_post) - 1)
        dt_k  = ts_post[k+1] - ts_post[k]
        g_mid = 0.5 * (g_post[k] + g_post[k+1])
        H = H_anh_9 + g_mid * exch_9 + ε_val * ε_op_9
        U_post = exp(-im * dt_k * H) * U_post
    end
    return U_post * U_flat * U_pre
end

# --- 9-dim default propagator (calibrated Gaussian-square, no MW) ---
function propagate_default_3lvl_full(ε_val::Real, ε_op_9::AbstractMatrix)
    U = Matrix{ComplexF64}(I, 9, 9)
    @inbounds for k in 1:(length(ts_def) - 1)
        dt_k  = ts_def[k+1] - ts_def[k]
        g_mid = 0.5 * (g_def_phys[k] + g_def_phys[k+1])
        H = H_anh_9 + g_mid * exch_9 + ε_val * ε_op_9
        U = exp(-im * dt_k * H) * U
    end
    return U
end

# --- ε-sweep ---
err_channels = [(:n̂1, n1_9), (:n̂2, n2_9), (:n̂1n̂2, n1_9 * n2_9)]
F_rob = Dict{Symbol, Vector{Float64}}()
F_def = Dict{Symbol, Vector{Float64}}()
L_rob = Dict{Symbol, Vector{Float64}}()
L_def = Dict{Symbol, Vector{Float64}}()

println("\n=== ε-sweep (3-lvl, full gate, no virtual Z) ===")
for (name, op) in err_channels
    Fr = Float64[]; Lr = Float64[]
    Fd = Float64[]; Ld = Float64[]
    for ε in εs_rad
        Ur = propagate_robust_3lvl_full(ε, op)
        Ud = propagate_default_3lvl_full(ε, op)
        Ur_sub = Ur[QSUB, QSUB]
        Ud_sub = Ud[QSUB, QSUB]
        push!(Fr, abs2(tr(U_iSWAP' * Ur_sub)) / 16)
        push!(Lr, 1 - real(tr(Ur_sub' * Ur_sub)) / 4)
        push!(Fd, abs2(tr(U_iSWAP' * Ud_sub)) / 16)
        push!(Ld, 1 - real(tr(Ud_sub' * Ud_sub)) / 4)
    end
    F_rob[name] = Fr; L_rob[name] = Lr
    F_def[name] = Fd; L_def[name] = Ld
    i0 = argmin(abs.(εs_MHz_2q))
    @printf("%s @ ε=0:   robust F=%.6f L=%.3e    default F=%.6f L=%.3e\n",
        string(name), Fr[i0], Lr[i0], Fd[i0], Ld[i0])
end

# --- Fidelity plot: 3 panels (one per error channel), robust vs default raw ---
ch_colors = Dict(:n̂1 => :crimson, :n̂2 => :forestgreen, :n̂1n̂2 => :royalblue)
fig_F = Figure(size = (1500, 500), fontsize = 14)
ax_n1   = Axis(fig_F[1, 1]; xlabel = "ε (MHz)", ylabel = "1 − F_3", yscale = log10, title = "n̂_1")
ax_n2   = Axis(fig_F[1, 2]; xlabel = "ε (MHz)", ylabel = "1 − F_3", yscale = log10, title = "n̂_2")
ax_n1n2 = Axis(fig_F[1, 3]; xlabel = "ε (MHz)", ylabel = "1 − F_3", yscale = log10, title = "n̂_1·n̂_2")
axes_F = Dict(:n̂1 => ax_n1, :n̂2 => ax_n2, :n̂1n̂2 => ax_n1n2)
for (name, _) in err_channels
    ax = axes_F[name]; c = ch_colors[name]
    lines!(ax, εs_MHz_2q, max.(1 .- F_rob[name], 1e-12); color = c, linewidth = 2.5, label = "robust")
    lines!(ax, εs_MHz_2q, max.(1 .- F_def[name], 1e-12); color = c, linewidth = 1.5, linestyle = :dash, label = "default")
    axislegend(ax; position = :lb, labelsize = 10)
end
save(joinpath(@__DIR__, "fidelity_sweep_$(POLISH_TAG).png"), fig_F; px_per_unit = 4)
println("Fidelity sweep saved: fidelity_sweep_$(POLISH_TAG).png")

# --- Control shapes: time domain (4 MW channels) ---
N_dense_mw = 2000
ts_dense   = collect(LinRange(0.0, T_MW_NS, N_dense_mw))
pulse_dense = CubicSplinePulse(traj_polish[:u], traj_polish[:du], vec(traj_polish[:t]))
us_dense = sample(pulse_dense, ts_dense)

fig_T = Figure(size = (1200, 500), fontsize = 14)
ax_T = Axis(fig_T[1, 1]; xlabel = "t (ns)", ylabel = "MW amplitude (rad/ns)",
    title = "Polished MW controls (time domain)")
labels_mw = ["u_X1", "u_Y1", "u_X2", "u_Y2"]
colors_mw = [:crimson, :orange, :forestgreen, :purple]
for i in 1:4
    lines!(ax_T, ts_dense, vec(us_dense[i, :]); color = colors_mw[i], linewidth = 2.0, label = labels_mw[i])
    scatter!(ax_T, vec(traj_polish[:t]), vec(traj_polish[:u][i, :]); color = :black, markersize = 5)
end
hlines!(ax_T, [a_bound_MW, -a_bound_MW]; color = :gray, linestyle = :dash, linewidth = 1.0, label = "±a_bound")
axislegend(ax_T; position = :rt, labelsize = 10)
save(joinpath(@__DIR__, "controls_time_$(POLISH_TAG).png"), fig_T; px_per_unit = 4)
println("Controls (time)  saved: controls_time_$(POLISH_TAG).png")

# --- Control shapes: frequency domain ---
N_fft = 16384
ts_fft = collect(LinRange(0.0, T_MW_NS, N_fft))
dt_fft = ts_fft[2] - ts_fft[1]
us_fft = sample(pulse_dense, ts_fft)
Ω_1 = us_fft[1, :] .+ im .* us_fft[2, :]
Ω_2 = us_fft[3, :] .+ im .* us_fft[4, :]
Ω_1_hat = fftshift(fft(Ω_1)) .* dt_fft
Ω_2_hat = fftshift(fft(Ω_2)) .* dt_fft
fs_MHz  = fftshift(fftfreq(N_fft, 1.0 / dt_fft)) .* 1e3

fig_W = Figure(size = (1200, 500), fontsize = 14)
ax_W = Axis(fig_W[1, 1]; xlabel = "f (MHz)", ylabel = "|Ω̂(f)|²",
    yscale = log10, title = "Polished MW spectrum (power)")
lines!(ax_W, fs_MHz, max.(abs2.(Ω_1_hat), 1e-14); color = :crimson, linewidth = 2.0, label = "Ω̂₁")
lines!(ax_W, fs_MHz, max.(abs2.(Ω_2_hat), 1e-14); color = :royalblue, linewidth = 2.0, label = "Ω̂₂")
vlines!(ax_W, [η_MHz, -η_MHz]; color = :black, linestyle = :dash, linewidth = 1.5, label = "±|η|/2π")
hlines!(ax_W, [ε_MAX]; color = :gray, linestyle = :dot, linewidth = 1.5, label = "ε_max")
xlims!(ax_W, -400, 400)
axislegend(ax_W; position = :lb, labelsize = 10)
save(joinpath(@__DIR__, "controls_freq_$(POLISH_TAG).png"), fig_W; px_per_unit = 4)
println("Controls (freq)  saved: controls_freq_$(POLISH_TAG).png")

# --- Save ε-sweep numerical data too ---
@save joinpath(@__DIR__, "eps_sweep_$(POLISH_TAG).jld2") εs_MHz_2q F_rob F_def L_rob L_def
println("ε-sweep data saved: eps_sweep_$(POLISH_TAG).jld2")
println("\nAll diagnostics complete.")
