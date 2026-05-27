import Pkg
Pkg.activate(@__DIR__)

using Piccolo, JLD2, LinearAlgebra, Printf
using DataInterpolations: CubicHermiteSpline

const T_NS = 50.0
const a_bound = 2π * 0.01
const η_anh = -2π * 0.170
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const U_target = (1/sqrt(2)) * ComplexF64[1.0 1.0; 1.0 -1.0]

const I3 = Matrix{ComplexF64}(I, 3, 3)
const a3 = ComplexF64[0 1 0; 0 0 sqrt(2); 0 0 0]
const X3 = a3 + adjoint(a3)
const Y3 = im * (adjoint(a3) - a3)
const n3 = adjoint(a3) * a3
const H_anh3 = (η_anh / 2) * n3 * (n3 - I3)
const subspace_indices = [1, 2]

out_dir = joinpath(@__DIR__, "spectral_sweep_results")
files = readdir(out_dir)
traj_files = sort(filter(f -> startswith(f, "traj_") && endswith(f, ".jld2"), files))
tightest_file = traj_files[end]
@printf("Loading: %s\n", tightest_file)
@load joinpath(out_dir, tightest_file) traj leak_bound vs achieved_omega_sq

@printf("\n=== Trajectory metadata ===\n")
@printf("ε_max                     = %.3e\n", leak_bound)
@printf("achieved |Ω̂(η)|²          = %.3e\n", achieved_omega_sq)
@printf("χ_z (from sweep)          = %.4e\n", vs)
@printf("ratio (achieved/ε_max)    = %.3f\n", achieved_omega_sq / leak_bound)

# === 2-lvl honest rollout ===
sys_2lvl = QuantumSystem(zeros(ComplexF64, 2, 2), [σx, σy], fill(a_bound, 2))
Ũ⃗ = unitary_rollout(traj, sys_2lvl;
    interpolation = :cubic_hermite, abstol = 1e-12, reltol = 1e-12)
U_T_2lvl = iso_vec_to_operator(Ũ⃗[:, end])
F_2lvl = abs2(tr(U_target' * U_T_2lvl)) / 4

@printf("\n=== 2-lvl fidelity ===\n")
@printf("F₂                        = %.8f\n", F_2lvl)
@printf("1 − F₂                    = %.3e\n", 1 - F_2lvl)

# === 3-lvl honest rollout (combined-envelope cubic-Hermite spline) ===
us = traj[:u]; dus = traj[:du]
ts_knots = collect(range(0.0, T_NS, length = size(us, 2)))
sp_X = CubicHermiteSpline(dus[1, :], us[1, :], ts_knots)
sp_Y = CubicHermiteSpline(dus[2, :], us[2, :], ts_knots)

function propagate_3lvl(sp_X, sp_Y; N_fine = 2000)
    ts_f = collect(range(0.0, T_NS, length = N_fine))
    dt_sub = ts_f[2] - ts_f[1]
    U3 = Matrix{ComplexF64}(I, 3, 3)
    for k in 1:N_fine - 1
        t_mid = 0.5 * (ts_f[k] + ts_f[k+1])
        H = H_anh3 + sp_X(t_mid) * X3 + sp_Y(t_mid) * Y3
        U3 = exp(-im * dt_sub * H) * U3
    end
    return U3
end

U3 = propagate_3lvl(sp_X, sp_Y)
Usub = U3[subspace_indices, subspace_indices]
F_3lvl = abs2(tr(U_target' * Usub)) / 4
L_3lvl = 1 - real(tr(Usub' * Usub)) / 2

@printf("\n=== 3-lvl honest verification ===\n")
@printf("F₃ (subspace overlap)     = %.8f\n", F_3lvl)
@printf("1 − F₃                    = %.3e\n", 1 - F_3lvl)
@printf("L₃(T)                     = %.3e\n", L_3lvl)
