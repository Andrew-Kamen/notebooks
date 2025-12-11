using PiccoloQuantumObjects
using DirectTrajOpt
using QuantumCollocation
using ForwardDiff
using LinearAlgebra
using Plots
using SparseArrays
using Statistics
using CairoMakie
using Random
using ExponentialAction
using NamedTrajectories
const ⊗ = kron

############################  Helpers  ######################################


#### Convert Betweened Flattened Matrices and Full Matrices ###
function vec_to_components(controls_vec, n_controls)
    n_timesteps = length(controls_vec) ÷ n_controls 
    controls  = reshape(controls_vec, (n_timesteps,n_controls))'
    return controls
end


### Forward Differentiation, with Piccolo-style Padding ###
function derivative_stencil(n_timesteps, n_controls)
    D = spzeros(n_timesteps, n_timesteps)
    for t in 1:n_timesteps-1
        D[t, t]   = -1
        D[t, t+1] =  1
    end
    D[end, end-1] = -1
    D[end, end]   =  1
    return I(n_controls) ⊗ D
end

function second_derivative_stencil(n_timesteps,n_controls)
    D = spzeros(n_timesteps, n_timesteps)
    for t in 1:n_timesteps-2
        D[t, t]   = 1
        D[t, t+1] =  -2
        D[t, t+2] =  1
    end
    return I(n_controls) ⊗ D
end

function first_derivative(n_controls, datavec)
    return 1/datavec[end]  * derivative_stencil((length(datavec) - 1) ÷ n_controls, n_controls) * datavec[1:end-1]
end 

function first_derivative_jac(n_controls, datavec)
    N = length(datavec)
    n_timesteps = (N - 1) ÷ n_controls
    Δt = datavec[end]
    jac  = spzeros(N - 1, N )
    D = derivative_stencil(n_timesteps, n_controls)
    jac[1:N-1, 1:N-1] = D/Δt 
    jac[:,end] = -1/Δt^2 * D * datavec[1:N-1]
    return jac
end

function second_derivative(n_controls, datavec)
    return 1/datavec[end]^2  * second_derivative_stencil((length(datavec) - 1) ÷ n_controls, n_controls) * datavec[1:end-1]
end 

function second_derivative_jac(n_controls, datavec)
    N = length(datavec)
    n_timesteps = (N - 1) ÷ n_controls
    Δt = datavec[end]
    jac  = spzeros(N - 1, N )
    D = second_derivative_stencil(n_timesteps, n_controls)
    jac[1:N-1, 1:N-1] = D/Δt^2
    jac[:,end] = -2/Δt^3 * D * datavec[1:N-1]
    return jac
end

#### Regularization ###
quadratic_reg = (vec, Δt) ->  1/2 * sum(vec.^2) * Δt^2
quadratic_reg_grad = (vec, Δt) ->  [vec * Δt^2, sum(vec.^2) * Δt]


### Unitary Rollouts ###
expm = M -> expv(1, M,I(size(M)[1]))

function prop(control_slice, G_func, Δt)
    return expm(G_func(control_slice) * Δt)
end 

### Use ForwardDiff to avoid the Commutator - Integral 
function prop_control_grad(control_slice, G_func, Δt)
    U = prop(control_slice, G_func, Δt)

    full_jac  = ForwardDiff.jacobian(v -> expm(G_func(v) * Δt), control_slice)
    return [reshape(full_jac[:,c], size(U)) for c in 1:length(control_slice)]
end

### Time Derivative is Easy!
function prop_time_grad(control_slice, G_func, Δt)
    return prop(control_slice, G_func, Δt) * G_func(control_slice)
end

function rollout(controls_vec, n_controls, Δt, G_func, t_final)
    controls = vec_to_components(controls_vec, n_controls)
    n_timesteps = length(controls_vec) ÷ n_controls 
    
    # Ensure t_final is within valid bounds
    t_final = min(t_final, n_timesteps)

    id = I(size(G_func(zeros(n_controls)))[1] ÷ 2)
    U_vec = operator_to_iso_vec(id)

    for t in 1:t_final - 1
        U_vec = (prop(controls[:, t], G_func, Δt)) * iso_vec_to_iso_operator(U_vec)
        U_vec = iso_operator_to_iso_vec(U_vec)
    end
    return U_vec 
end

function rollout_grad(controls_vec, n_controls, Δt, G_func, t_final)
    controls = vec_to_components(controls_vec, n_controls)
    n_timesteps = length(controls_vec) ÷ n_controls 
    
    # Ensure t_final is within valid bounds
    t_final = min(t_final, n_timesteps)

    id = I(size(G_func(zeros(n_controls)), 1) ÷ 2)
    iso_init = operator_to_iso_vec(id)

    d = size(G_func(zeros(n_controls)), 1)
    id2 = I(d)

    # Precompute propagators and derivatives up to t_final-1
    prop_list       = [prop(controls[:, t], G_func, Δt) for t in 1:t_final-1]
    ∂prop_control   = [prop_control_grad(controls[:, t], G_func, Δt) for t in 1:t_final-1]
    ∂prop_time_list = [prop_time_grad(controls[:, t], G_func, Δt) for t in 1:t_final-1]

    # Precompute prefix and suffix products up to t_final
    prefix = Vector{Matrix}(undef, t_final)
    suffix = Vector{Matrix}(undef, t_final)
    prefix[1] = id2
    for i in 2:t_final
        prefix[i] = prop_list[i-1] * prefix[i-1]
    end
    suffix[end] = id2
    for i in t_final-1:-1:1
        suffix[i] = suffix[i+1] * prop_list[i]
    end

    # Control Jacobian - only compute for timesteps up to t_final-1
    control_jac = zeros(length(iso_init), length(controls_vec))
    for i in 1:t_final-1
        left, right = prefix[i], suffix[i+1]
        for (c, diff) in enumerate(∂prop_control[i])
            idx = (c-1) * n_timesteps + i
            control_jac[:, idx] = iso_operator_to_iso_vec(right * diff * left * iso_vec_to_iso_operator(iso_init))
        end 
    end

    # Time derivative - only sum over timesteps up to t_final-1
    time_grad = zeros(size(iso_init))
    for i in 1:t_final-1
        left, right = prefix[i], suffix[i+1]
        time_grad += iso_operator_to_iso_vec(right * ∂prop_time_list[i] * left * iso_vec_to_iso_operator(iso_init))
    end

    return control_jac, time_grad
end

#### Objectives ####


### Regularizers ###
function control_regularization_objective(n_controls, R_a, R_da, R_dda)
    function obj(datavec)
        controls, Δt = datavec[1:end-1], datavec[end]
        da = first_derivative(n_controls,datavec)
        dda = second_derivative(n_controls,datavec)
        return R_a * quadratic_reg(controls, Δt) + 
               R_da * quadratic_reg(da, Δt) + 
               R_dda * quadratic_reg(dda, Δt) 
    end
end


function control_regularization_grad(n_controls, R_a, R_da, R_dda)
    function grad(datavec)
        controls, Δt = datavec[1:end-1], datavec[end]
        da = first_derivative(n_controls,datavec)
        dda = second_derivative(n_controls,datavec)
        J_da = first_derivative_jac(n_controls, datavec)
        J_dda = second_derivative_jac(n_controls, datavec)
        
        grad = zeros(length(datavec))
        
        Da, Dt = quadratic_reg_grad(controls, Δt)
        grad[1:end-1] += R_a * Da
        grad[end] += R_a * Dt

        Da, Dt = quadratic_reg_grad(da, Δt)
        grad[1:end-1] += R_da * (J_da[:, 1:end-1]'Da)
        grad[end] += R_da * (Da' * J_da[:, end]) + R_da * Dt  

        Da, Dt = quadratic_reg_grad(dda, Δt)
        grad[1:end-1] += R_dda * (J_dda[:, 1:end-1]'Da)
        grad[end] += R_dda * (Da' * J_dda[:, end]) + R_dda * Dt 

        return grad
    end
end

### Fidelity ###
function iso_perm_indices(n)
    dim = 2n^2
    perm  = Vector{Int}(undef, dim)
    permT = Vector{Int}(undef, dim)

    for k in 1:n, j in 1:2, i in 1:n
        p = i + (j-1)*n + (k-1)*2n       # original index
        q = i + (k-1)*n + (j-1)*n^2      # permuted index
        perm[q] = p
        permT[p] = q
    end

    return perm, permT
end

function unitary_infidelity(iso_U, U_targ)
    n =  size(U_targ)[1]
    return abs2(tr(iso_vec_to_operator(iso_U)'U_targ))/n^2
end 

function unitary_infidelity_grad(iso_U, U_targ)
    n = size(U_targ)[1]
    iso_targ = operator_to_iso_vec(U_targ)
    perm, permT = iso_perm_indices(n)

    U_vec = iso_U[perm]
    U_goal_vec = iso_targ[perm]

    U_real = U_vec[1:n^2]
    U_imag = U_vec[n^2+1:end]
    U_goal_real = U_goal_vec[1:n^2]
    U_goal_imag = U_goal_vec[n^2+1:end]

    z_real = sum(U_real .* U_goal_real + U_imag .* U_goal_imag)
    z_imag = sum(U_imag .* U_goal_real - U_real .* U_goal_imag)

    grad = ((2/n^2) * [z_real * U_goal_real - z_imag * U_goal_imag;
                     z_real * U_goal_imag + z_imag * U_goal_real])[permT]
    return grad
end 

function unitary_infidelity_objective(n_controls, sys, Q, U_targ)
    function obj(datavec)
        controls, Δt = datavec[1:end-1], datavec[end]
        G_func = a -> sys.G(a)
        U_final = rollout(controls, n_controls, Δt, G_func, n_timesteps)
        return Q * (1-unitary_infidelity(U_final, U_targ))
    end
end 

function unitary_infidelity_grad(n_controls, sys, Q, U_targ)
    function grad(datavec)
        controls, Δt = datavec[1:end-1], datavec[end]
        G_func = a -> sys.G(a)
        ∂ₐU, ∂ₜU = rollout_grad(controls, n_controls, Δt, G_func, n_timesteps)
        U_final = rollout(controls, n_controls, Δt, G_func, n_timesteps)
        ∂F = unitary_infidelity_grad(U_final, U_targ)
        grad = zeros(length(datavec))
        grad[1:end-1] = ∂F'∂ₐU
        grad[end] = ∂F'∂ₜU
        return  - Q * grad
    end
end 

### Toggling ###
function toggle_term(H_err, a, U_vec) 
    U = iso_vec_to_operator(U_vec)
    He = H_err(a)
    return operator_to_iso_vec(U' * He * U)
end 

function toggle_term_grad(H_err, ∂H_err, a, U_vec)
    n = Int(sqrt(length(U_vec) ÷ 2))
    He = H_err(a)
    ∂He = ∂H_err(a)
    U = iso_vec_to_operator(U_vec)

    ∂ₐ =  reduce(hcat, [operator_to_iso_vec(U' * ∂a * U) for ∂a in ∂He])

    ∂ᵤ = zeros(2 * n^2, 2 * n^2)

    for idx in 1:2 * n^2
        i = (idx % n) == 0 ? n : idx % n
        j =(idx - i) ÷ (2 * n) + 1
        re = (idx % (2*n) == 0 ? 2*n : idx%(2*n)) ≤ n
        d = spzeros(ComplexF64, (n,n))
        d[i,j] = re ? 1 : 1im 
        ∂ᵤ[:,idx] = operator_to_iso_vec(U' * He * d + d' * He * U)
    end 

    return ∂ₐ, ∂ᵤ
end

function full_toggle_term(H_err, controls_vec, n_controls, Δt, G_func)
    controls = vec_to_components(controls_vec, n_controls)
    n_timesteps = length(controls_vec) ÷ n_controls
    
    # Compute all rollouts cumulatively in one pass
    id = I(size(G_func(zeros(n_controls)))[1] ÷ 2)
    U_current = operator_to_iso_vec(id)
    
    out = zeros(2 * size(H_err(zeros(n_controls)))[1]^2)
    
    for t in 1:n_timesteps -1 
        # Add toggle term for current timestep
        out += toggle_term(H_err, controls[:, t], U_current)
        # Update rollout for next iteration (if not last timestep)
        if t < n_timesteps
            U_current = (prop(controls[:, t], G_func, Δt)) * iso_vec_to_iso_operator(U_current)
            U_current = iso_operator_to_iso_vec(U_current)
        end
    end
    
    return norm(vec(out), 2)^2
end

function full_toggle_term_grad(H_err, ∂H_err, controls_vec, n_controls, Δt, G_func)
    controls = vec_to_components(controls_vec, n_controls)
    n_timesteps = length(controls_vec) ÷ n_controls
    
    id = I(size(G_func(zeros(n_controls)), 1) ÷ 2)
    iso_init = operator_to_iso_vec(id)
    d = size(G_func(zeros(n_controls)), 1)
    id2 = I(d)
    
    # Precompute all propagators and derivatives once
    prop_list = [prop(controls[:, t], G_func, Δt) for t in 1:n_timesteps]
    ∂prop_control = [prop_control_grad(controls[:, t], G_func, Δt) for t in 1:n_timesteps]
    ∂prop_time_list = [prop_time_grad(controls[:, t], G_func, Δt) for t in 1:n_timesteps]
    
    # Compute cumulative rollouts (U at each timestep)
    U_rollouts = Vector{Vector{Float64}}(undef, n_timesteps)
    U_rollouts[1] = iso_init
    for t in 2:n_timesteps
        U_rollouts[t] = iso_operator_to_iso_vec(
            prop_list[t-1] * iso_vec_to_iso_operator(U_rollouts[t-1])
        )
    end
    
    # Precompute prefix products for rollout gradients
    # prefix[i] = prop[1] * ... * prop[i-1]
    prefix = Vector{Matrix}(undef, n_timesteps + 1)
    prefix[1] = id2
    for i in 2:n_timesteps + 1
        prefix[i] = prop_list[i-1] * prefix[i-1]
    end
    
    # Initialize output
    out = zeros(2 * size(H_err(zeros(n_controls)))[1]^2)
    ∂ₐo = zeros(length(out), length(controls_vec))
    ∂ₜo = zeros(length(out))
    
    # Compute toggle terms and gradients
    for t in 1:n_timesteps - 1
        U = U_rollouts[t]
        
        # Add toggle term
        out += toggle_term(H_err, controls[:, t], U)
        
        # Compute toggle term gradients
        ∂ₐT, ∂ᵤT = toggle_term_grad(H_err, ∂H_err, controls[:, t], U)
        
        # Direct contribution: derivative of toggle_term w.r.t. controls at time t
        ∂ₐT_full = spzeros(length(out), length(controls_vec))
        for c in 1:n_controls
            ∂ₐT_full[:, t + n_timesteps * (c-1)] = ∂ₐT[:, c]
        end
        ∂ₐo += ∂ₐT_full
        
        # Indirect contribution: derivative through U_t
        # We need ∂U_t/∂controls[i] for i < t and ∂U_t/∂Δt
        
        # Build the suffix for this specific t (from timestep 1 to t-1)
        suffix_t = Vector{Matrix}(undef, t)
        suffix_t[t] = id2
        for i in (t-1):-1:1
            suffix_t[i] = suffix_t[i+1] * prop_list[i]
        end

        # Gradient of U_t w.r.t. controls at earlier timesteps
        for i in 1:t-1
            left = prefix[i]
            right = suffix_t[i+1]
            
            for (c, diff) in enumerate(∂prop_control[i])
                idx = (c-1) * n_timesteps + i
                ∂U_∂aᵢ = iso_operator_to_iso_vec(
                    right * diff * left * iso_vec_to_iso_operator(iso_init)
                )
                ∂ₐo[:, idx] += ∂ᵤT * ∂U_∂aᵢ
            end
        end
        
        # Time gradient contribution from U_t
        for i in 1:t-1
            left = prefix[i]
            right = suffix_t[i+1]
            
            ∂U_∂t = iso_operator_to_iso_vec(
                right * ∂prop_time_list[i] * left * iso_vec_to_iso_operator(iso_init)
            )
            ∂ₜo += ∂ᵤT * ∂U_∂t
        end
    end
    return Δt^2 *2 * ∂ₐo' * out, Δt^2 * 2 * ∂ₜo' * out + 2 * Δt * norm(vec(out), 2)^2
end

function toggle_objective(n_controls, varsys, Q_t)
    H_err = a -> [1im * iso_operator_to_operator(H(a)) for H in varsys.G_vars]
    n = size(H_err(zeros(n_controls))[1], 1)
    num_err = length(H_err(zeros(n_controls)))
    
    function obj(datavec)
        controls, Δt = datavec[1:end-1], datavec[end]
        G_func = a -> varsys.G(a)
        sum = 0
        
        for n in 1:num_err
            H_err_n = a -> H_err(a)[n]
            sum += full_toggle_term(H_err_n, controls, n_controls, Δt, G_func)
        end
        
        return Q_t * sum * Δt^2/n
    end
end

function toggle_grad(n_controls, varsys, Q_t)
    H_err = a -> [1im * iso_operator_to_operator(H(a)) for H in varsys.G_vars]
    ∂H_err = a -> [1im * [iso_operator_to_operator(v) for v in H(a)] for H in varsys.∂G_vars]
    num_err = length(H_err(zeros(n_controls)))
    n = size(H_err(zeros(n_controls))[1], 1)
    function grad(datavec)
        controls, Δt = datavec[1:end-1], datavec[end]
        G_func = a -> varsys.G(a)
        grad = zeros(length(datavec))
        
        for n in 1:num_err
            H_err_n = a -> H_err(a)[n]
            ∂H_err_n = a -> ∂H_err(a)[n]
            ∂ₐ, ∂ₜ = full_toggle_term_grad(H_err_n, ∂H_err_n, controls, n_controls, Δt, G_func)
            grad[1:end-1] += ∂ₐ
            grad[end] += ∂ₜ
        end
        return Q_t * grad/n 
    end
end

### Adjoint Rollouts ###
function sensitivity(v)
    return sum(v.^2)/ sqrt(length(v)/2)
end

function sensitivity_grad(v)
    return 2 *v /sqrt(length(v)/2)
end

function var_G(G, G_vars)
    n, _ = size(G)
    v = length(G_vars)
    
    # Let the output matrix have the same element type as the input
    T = eltype(G)  # This will be Float64 normally, or Dual during AD
    
    G_0 = kron(I(v + 1), G)
    G_V = zeros(T, (v + 1) * n, (v + 1) * n)  # Use the same type as input
    
    for i = eachindex(G_vars)
        G_V[i * n + 1:(i + 1) * n, 1:n] = G_vars[i]
    end
    
    return G_0 + G_V
end

### Small helper: stack iso-operators for U + v senses ###
# returns a (n*(v+1))×n matrix where top block is U, subsequent are senses
@inline function stack_operator_and_senses(Uop, senses)
    blocks = (Uop, senses...)
    return vcat(blocks...)
end

@inline function unpack_stacked(stacked, n, v)
    # returns Uop, vector of v sense matrices
    Uop = stacked[1:n, :]
    senses = [stacked[k*n+1 : (k+1)*n, :] for k in 1:v]
    return Uop, senses
end

function final_variational_rollout(controls_vec, n_controls, Δt, G_func, var_G_funcs)
    G_eff = a -> var_G(G_func(a), [vG(a) for vG in var_G_funcs])
    v = length(var_G_funcs)

    controls = vec_to_components(controls_vec, n_controls)
    n_timesteps = length(controls_vec) ÷ n_controls

    id_small = I(size(G_func(zeros(n_controls)), 1) ÷ 2)
    n = size(id_small, 1) * 2         
    U_vec = operator_to_iso_vec(id_small)

    # initial stacked operator (U + v sense vectors)
    Uop_init = iso_vec_to_iso_operator(U_vec)
    sense_ops = [zeros(size(Uop_init)) for _ in 1:v]        # each is n×n
    stacked = stack_operator_and_senses(Uop_init, sense_ops) # (n*(v+1)) × n

    for t in 1:(n_timesteps - 1)
        P = prop(controls[:, t], G_eff, Δt)                # (n*(v+1))×(n*(v+1))
        stacked = P * stacked
    end

    Uop_final, senses_final = unpack_stacked(stacked, n, v)
    return iso_operator_to_iso_vec(Uop_final), [iso_operator_to_iso_vec(s) for s in senses_final]
end

function final_variational_rollout_grad(controls_vec, n_controls, Δt, G_func, var_G_funcs)
    G_eff = a -> var_G(G_func(a), [vG(a) for vG in var_G_funcs])
    v = length(var_G_funcs)
    controls = vec_to_components(controls_vec, n_controls)
    n_timesteps = length(controls_vec) ÷ n_controls 

    id = I(size(G_func(zeros(n_controls)))[1] ÷ 2)
    n = size(id, 1) * 2
    U_init = operator_to_iso_vec(id)
    sense_init = [zeros(length(U_init)) for _ ∈ 1:v]
    init = [iso_vec_to_iso_operator(U_init); vcat([iso_vec_to_iso_operator(s) for s in sense_init]...)] 

    prop_list = [prop(controls[:, t], G_eff, Δt) for t ∈ 1:n_timesteps-1]
    ∂prop_control_list = [prop_control_grad(controls[:, t], G_eff, Δt) for t ∈ 1:n_timesteps-1]
    ∂prop_time_list = [prop_time_grad(controls[:, t], G_eff, Δt) for t ∈ 1:n_timesteps-1]

    ### Compute the Control Jacobian ###
    control_U_jac = zeros(length(U_init), length(controls_vec))
    control_sense_jac = [zeros(length(U_init), length(controls_vec)) for _ ∈ 1:v]

    for i in 1:n_timesteps - 1
        left = (i > 1) ? reduce(*, reverse(prop_list[1:i-1])) : I(n * (v+1))
        right = (i < n_timesteps - 1) ? reduce(*, reverse(prop_list[i+1:end])) : I(n * (v+1))

        for (c, diff) in enumerate(∂prop_control_list[i])
            full = right * diff * left * init
            Uop, senses = unpack_stacked(full, n, v)
            
            control_U_jac[:, (c-1) * n_timesteps + i] =
                iso_operator_to_iso_vec(Uop)
            for k in 1:v
                control_sense_jac[k][:, (c-1) * n_timesteps + i] =
                    iso_operator_to_iso_vec(senses[k])
            end
        end 
    end

    ### Compute the Time Derivative ###
    total = zeros(size(U_init))
    sense_totals = [zeros(size(U_init)) for _ ∈ 1:v]

    for i in 1:n_timesteps - 1
        left = (i > 1) ? reduce(*, reverse(prop_list[1:i-1])) : I(n * (v+1))
        right = (i < n_timesteps - 1) ? reduce(*, reverse(prop_list[i+1:end])) : I(n * (v+1))

        full = right * ∂prop_time_list[i] * left * init
        Uop, senses = unpack_stacked(full, n, v)
        
        total += iso_operator_to_iso_vec(Uop)
        for k in 1:v
            sense_totals[k] += iso_operator_to_iso_vec(senses[k])
        end
    end

    return control_U_jac, control_sense_jac, total, sense_totals
end

function variational_infidelity_sensitivity_objective(n_controls, varsys, Q, Q_r, U_targ)
    function obj(datavec)
        controls, Δt = datavec[1:end-1], datavec[end]
        G_func = a -> varsys.G(a)
        var_G_funcs = [a -> g(a) for g in varsys.G_vars]

        U_final, final_sense = final_variational_rollout(controls, n_controls, Δt, G_func, var_G_funcs)
        return Q * (1-unitary_infidelity(U_final, U_targ)) + Q_r * sum([sensitivity(s) for s in final_sense])
    end
end 

function variational_infidelity_sensitivity_grad(n_controls, varsys, Q, Q_r, U_targ)
    function grad(datavec)
        controls, Δt = datavec[1:end-1], datavec[end]
        G_func = a -> sys.G(a)
        var_G_funcs = [a -> g(a) for g in varsys.G_vars]

        ∂ₐU, ∂ₐS, ∂ₜU, ∂ₜS = final_variational_rollout_grad(controls, n_controls, Δt, G_func, var_G_funcs)
        U_final, final_sense = final_variational_rollout(controls, n_controls, Δt, G_func, var_G_funcs)

        ∂F = unitary_infidelity_grad(U_final, U_targ)
        ∂S = [sensitivity_grad(s) for s in final_sense]
        
        grad = zeros(length(datavec))

        
        grad[1:end-1] = -Q * ∂F'∂ₐU
        grad[end] = -Q * ∂F'∂ₜU

        for (∂s, ∂ₐ, ∂ₜ) in zip(∂S, ∂ₐS, ∂ₜS)
            
            grad[1:end-1] += Q_r * (∂s'∂ₐ)'
            grad[end] += Q_r * ∂s'∂ₜ
        end
        return grad
    end
end 



function unitary_smooth_objective(n_controls, sys, U_goal, R_a, R_da, R_dda, Q)
    reg_obj = control_regularization_objective(n_controls, R_a, R_da, R_dda)
    fid_obj = unitary_infidelity_objective(n_controls, sys, Q, U_goal)
    return v -> reg_obj(v) + fid_obj(v)
end

function unitary_smooth_grad(n_controls, sys, U_goal, R_a, R_da, R_dda, Q)
    reg_grad = control_regularization_grad(n_controls, R_a, R_da, R_dda)
    fid_grad = unitary_infidelity_grad(n_controls, sys, Q, U_goal)
    return v -> reg_grad(v) + fid_grad(v)
end

function unitary_toggle_objective(n_controls, varsys, U_goal, R_a, R_da, R_dda, Q, Q_t)
    reg_obj = control_regularization_objective(n_controls, R_a, R_da, R_dda)
    fid_obj = unitary_infidelity_objective(n_controls, varsys, Q, U_goal)
    tog_obj = toggle_objective(n_controls, varsys, Q_t)
    return v -> reg_obj(v) + fid_obj(v) + tog_obj(v)
end

function unitary_toggle_grad(n_controls, varsys, U_goal, R_a, R_da, R_dda, Q, Q_t)
    reg_grad = control_regularization_grad(n_controls, R_a, R_da, R_dda)
    fid_grad = unitary_infidelity_grad(n_controls, varsys, Q, U_goal)
    tog_grad = toggle_grad(n_controls, varsys, Q_t)
    return v -> reg_grad(v) + fid_grad(v) + tog_grad(v)
end

function unitary_variational_objective(n_controls, varsys, U_goal, R_a, R_da, R_dda, Q, Q_r)
    reg_obj = control_regularization_objective(n_controls, R_a, R_da, R_dda)
    fid_obj = variational_infidelity_sensitivity_objective(n_controls, varsys, Q, Q_r, U_goal)
    return v -> reg_obj(v) + fid_obj(v)
end

function unitary_variational_grad(n_controls, varsys, U_goal, R_a, R_da, R_dda, Q, Q_r)
    reg_grad = control_regularization_grad(n_controls, R_a, R_da, R_dda)
    fid_grad = variational_infidelity_sensitivity_grad(n_controls, varsys, Q, Q_r, U_goal)
    return v -> reg_grad(v) + fid_grad(v)
end


### Optimization ###
using JuMP, Ipopt

function indirect_unitary_smooth_opt(
    Δt::Real, 
    U_goal::AbstractMatrix, 
    sys::QuantumSystem, 
    n_controls::Int,
    a_init::AbstractMatrix;
    iters::Int = 5000,
    a_bound::Union{Nothing, Real} = 1.0,
    da_bound::Union{Nothing, Real} = nothing, 
    dda_bound::Union{Nothing, Real} = nothing, 
    Δt_max::Real = 1.5 * Δt,
    Δt_min::Real = 0.5 * Δt,
    Q::Real = 500.0,
    R_a::Real = 1e-2, 
    R_da::Real = 1e-2,
    R_dda::Real = 1e-2 
)
    ### Initial Datavector ###
    datavec = [vec(a_init'); Δt]
    n = length(datavec)

    ### Objective ###
    obj_func = unitary_smooth_objective(n_controls, sys, U_goal, R_a, R_da, R_dda, Q)
    obj_grad = unitary_smooth_grad(n_controls, sys, U_goal, R_a, R_da, R_dda, Q)

    ### model and callback
    model = Model(Ipopt.Optimizer)
    hist  = []

    function my_intermediate_cb(args...)
        # grab the current iterate for ALL variables
        xval = [callback_value(model, xi) for xi in x]

        # push a *copy* so later changes don't overwrite history
        push!(hist, copy(xval))
        return true  # return false to stop
    end

    function my_objective(x...)
        return obj_func(collect(x))
    end

    function my_gradient(g::AbstractVector, x...)
        grad = obj_grad(collect(x))
        copyto!(g, grad)  # More efficient than loop
        return nothing
    end

    model = Model(Ipopt.Optimizer)

    # Ipopt settings for better performance
    set_optimizer_attribute(model, "print_level", 3)          # Moderate verbosity
    set_optimizer_attribute(model, "max_iter", iters)          # More iterations
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    MOI.set(model, MOI.RawOptimizerAttribute("nlp_scaling_method"), "gradient-based")
    MOI.set(model, Ipopt.CallbackFunction(), my_intermediate_cb)


        
    # Variables with bounds and better starting points
    @variable(model, x[i=1:n])

    for i in 1:n
        set_start_value(x[i], datavec[i])
    end

    # Add bounds on controls (not timestep)
    for i in 1:(n-1)  # All except last (timestep)
        if (i % n_timesteps == 1 || i % n_timesteps == 0)
            set_lower_bound(x[i], 0)
            set_upper_bound(x[i], 0)
        else
            set_lower_bound(x[i], -a_bound)
            set_upper_bound(x[i], a_bound)
        end
    end

    # Bound timestep to reasonable values
    set_lower_bound(x[n], Δt_min)   # Minimum timestep
    set_upper_bound(x[n], Δt_max)    # Maximum timestep

    # Register functions with correct signatures
    register(model, :my_obj, n, my_objective, my_gradient; autodiff = false)


    # Objective
    @NLobjective(model, Min, my_obj(x...))


    if(!isnothing(da_bound))
        function d(x1, x2, t)
            return (x1 - x2)/t 
        end 

        function ∇d(g, x1, x2, t)
            g[1] = 1/t
            g[2] = -1/t
            g[3] = -(x1 - x2)/t^2
        end 

        register(model, :d, 3, d, ∇d; autodiff = false)
        println("-------------------------------------Applying d Bound-------------------------------------")
        for n in 1:n_controls 
            for i in 1:n_timesteps-1
                @NLconstraint(model, -da_bound <= d(x[i + (n-1) * n_timesteps+1],
                                                    x[i + (n-1) * n_timesteps],
                                                    x[end]) <= da_bound)
                
            end 
        end
    end

    if(!isnothing(dda_bound))
        function dd(x1, x2, x3, t)
            return (x1 - 2 * x2 + x3)/t^2 
        end 

        function ∇dd(g, x1, x2, x3, t)
            g[1] = 1/t^2
            g[2] = -2/t^2
            g[3] = 1/t^2
            g[4] = -2 * (x1 - 2*x2 + x3) / t^3
        end

        register(model, :dd, 4, dd, ∇dd; autodiff = false)
        println("-------------------------------------Applying dda Bound-------------------------------------")
        for n in 1:n_controls 
            for i in 1:n_timesteps-2
                @NLconstraint(model, -dda_bound <= dd(x[i + (n-1) * n_timesteps+2],
                                                    x[i + (n-1) * n_timesteps+1],
                                                    x[i + (n-1) * n_timesteps],
                                                    x[end]) <= dda_bound)
                
            end 
        end
    end

    JuMP.optimize!(model)
    solution = value.(x)
    push!(hist, solution)
    a = vec_to_components(solution[1:end-1], n_controls)

    return obj_func, a, hist 
end 


function indirect_unitary_toggle_opt(
    Δt::Real, 
    U_goal::AbstractMatrix, 
    varsys::VariationalQuantumSystem, 
    n_controls::Int,
    a_init::AbstractMatrix;
    iters::Int = 5000,
    a_bound::Union{Nothing, Real} = 1.0,
    da_bound::Union{Nothing, Real} = nothing, 
    dda_bound::Union{Nothing, Real} = nothing, 
    Δt_max::Real = 1.5 * Δt,
    Δt_min::Real = 0.5 * Δt,
    Q::Real = 100.0,
    Q_t::Real = 1.0,
    R_a::Real = 1e-2, 
    R_da::Real = 1e-2,
    R_dda::Real = 1e-2,
    target_fidelity::Union{Nothing, Real} = nothing
)
    ### Initial Datavector ###
    datavec = [vec(a_init'); Δt]
    n = length(datavec)

    ### Objective ###
    obj_func = unitary_toggle_objective(n_controls, varsys, U_goal, R_a, R_da, R_dda, Q, Q_t)
    obj_grad = unitary_toggle_grad(n_controls, varsys, U_goal, R_a, R_da, R_dda, Q, Q_t)

    ### model and callback
    model = Model(Ipopt.Optimizer)
    hist  = []

    function my_intermediate_cb(args...)
        # grab the current iterate for ALL variables
        xval = [callback_value(model, xi) for xi in x]

        # push a *copy* so later changes don't overwrite history
        push!(hist, copy(xval))
        return true  # return false to stop
    end

    function my_objective(x...)
        return obj_func(collect(x))
    end

    function my_gradient(g::AbstractVector, x...)
        grad = obj_grad(collect(x))
        copyto!(g, grad)  # More efficient than loop
        return nothing
    end

    model = Model(Ipopt.Optimizer)

    # Ipopt settings for better performance
    set_optimizer_attribute(model, "print_level", 3)          # Moderate verbosity
    set_optimizer_attribute(model, "max_iter", iters)          # More iterations
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    MOI.set(model, MOI.RawOptimizerAttribute("nlp_scaling_method"), "gradient-based")
    MOI.set(model, Ipopt.CallbackFunction(), my_intermediate_cb)


        
    # Variables with bounds and better starting points
    @variable(model, x[i=1:n])

    for i in 1:n
        set_start_value(x[i], datavec[i])
    end

    # Add bounds on controls (not timestep)
    for i in 1:(n-1)  # All except last (timestep)
        if (i % n_timesteps == 1 || i % n_timesteps == 0)
            set_lower_bound(x[i], 0)
            set_upper_bound(x[i], 0)
        else
            set_lower_bound(x[i], -a_bound)
            set_upper_bound(x[i], a_bound)
        end
    end

    # Bound timestep to reasonable values
    set_lower_bound(x[n], Δt_min)   # Minimum timestep
    set_upper_bound(x[n], Δt_max)    # Maximum timestep

    # Register functions with correct signatures
    register(model, :my_obj, n, my_objective, my_gradient; autodiff = false)


    # Objective
    @NLobjective(model, Min, my_obj(x...))


    if(!isnothing(da_bound))
        function d(x1, x2, t)
            return (x1 - x2)/t 
        end 

        function ∇d(g, x1, x2, t)
            g[1] = 1/t
            g[2] = -1/t
            g[3] = -(x1 - x2)/t^2
        end 

        register(model, :d, 3, d, ∇d; autodiff = false)
        println("-------------------------------------Applying d Bound-------------------------------------")
        for n in 1:n_controls 
            for i in 1:n_timesteps-1
                @NLconstraint(model, -da_bound <= d(x[i + (n-1) * n_timesteps+1],
                                                    x[i + (n-1) * n_timesteps],
                                                    x[end]) <= da_bound)
                
            end 
        end
    end

    if(!isnothing(dda_bound))
        function dd(x1, x2, x3, t)
            return (x1 - 2 * x2 + x3)/t^2 
        end 

        function ∇dd(g, x1, x2, x3, t)
            g[1] = 1/t^2
            g[2] = -2/t^2
            g[3] = 1/t^2
            g[4] = -2 * (x1 - 2*x2 + x3) / t^3
        end

        register(model, :dd, 4, dd, ∇dd; autodiff = false)
        println("-------------------------------------Applying dda Bound-------------------------------------")
        for n in 1:n_controls 
            for i in 1:n_timesteps-2
                @NLconstraint(model, -dda_bound <= dd(x[i + (n-1) * n_timesteps+2],
                                                    x[i + (n-1) * n_timesteps+1],
                                                    x[i + (n-1) * n_timesteps],
                                                    x[end]) <= dda_bound)
                
            end 
        end
    end

    if(!isnothing(target_fidelity))
        infid = unitary_infidelity_objective(n_controls, sys, 1, U_goal)
        infig_grad = unitary_infidelity_objective(n_controls, sys, 1, U_goal)

        function fid(x...)
            return 1-infid(collect(x))
        end 

        function fid_grad(g::AbstractVector, x...)
            grad = -infig_grad(collect(x))
            copyto!(g, grad) 
            return nothing
        end
        

        register(model, :fid, n, fid, fid_grad; autodiff = false)
        println("-------------------------------------Applying Fidelity Constraint-------------------------------------")
        @NLconstraint(model, fid(x...) >= target_fidelity)
            
    end

    JuMP.optimize!(model)
    solution = value.(x)
    push!(hist, solution)
    a = vec_to_components(solution[1:end-1], n_controls)

    return obj_func, a, hist 
end 

function indirect_unitary_variational_opt(
    Δt::Real, 
    U_goal::AbstractMatrix, 
    varsys::VariationalQuantumSystem, 
    n_controls::Int,
    a_init::AbstractMatrix;
    iters::Int = 5000,
    a_bound::Union{Nothing, Real} = 1.0,
    da_bound::Union{Nothing, Real} = nothing, 
    dda_bound::Union{Nothing, Real} = nothing, 
    Δt_max::Real = 1.5 * Δt,
    Δt_min::Real = 0.5 * Δt,
    Q::Real = 100.0,
    Q_r::Real = 100.0,
    R_a::Real = 1e-2, 
    R_da::Real = 1e-2,
    R_dda::Real = 1e-2,
    target_fidelity::Union{Nothing, Real} = nothing
)
    ### Initial Datavector ###
    datavec = [vec(a_init'); Δt]
    n = length(datavec)

    ### Objective ###
    obj_func = unitary_variational_objective(n_controls, varsys, U_goal, R_a, R_da, R_dda, Q, Q_r)
    obj_grad = unitary_variational_grad(n_controls, varsys, U_goal, R_a, R_da, R_dda, Q, Q_r)

    ### model and callback
    model = Model(Ipopt.Optimizer)
    hist  = []

    function my_intermediate_cb(args...)
        # grab the current iterate for ALL variables
        xval = [callback_value(model, xi) for xi in x]

        # push a *copy* so later changes don't overwrite history
        push!(hist, copy(xval))
        return true  # return false to stop
    end

    function my_objective(x...)
        return obj_func(collect(x))
    end

    function my_gradient(g::AbstractVector, x...)
        grad = obj_grad(collect(x))
        copyto!(g, grad)  # More efficient than loop
        return nothing
    end

    model = Model(Ipopt.Optimizer)

    # Ipopt settings for better performance
    set_optimizer_attribute(model, "print_level", 3)          # Moderate verbosity
    set_optimizer_attribute(model, "max_iter", iters)          # More iterations
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    MOI.set(model, MOI.RawOptimizerAttribute("nlp_scaling_method"), "gradient-based")
    MOI.set(model, Ipopt.CallbackFunction(), my_intermediate_cb)


        
    # Variables with bounds and better starting points
    @variable(model, x[i=1:n])

    for i in 1:n
        set_start_value(x[i], datavec[i])
    end

    # Add bounds on controls (not timestep)
    for i in 1:(n-1)  # All except last (timestep)
        if (i % n_timesteps == 1 || i % n_timesteps == 0)
            set_lower_bound(x[i], 0)
            set_upper_bound(x[i], 0)
        else
            set_lower_bound(x[i], -a_bound)
            set_upper_bound(x[i], a_bound)
        end
    end

    # Bound timestep to reasonable values
    set_lower_bound(x[n], Δt_min)   # Minimum timestep
    set_upper_bound(x[n], Δt_max)    # Maximum timestep

    # Register functions with correct signatures
    register(model, :my_obj, n, my_objective, my_gradient; autodiff = false)


    # Objective
    @NLobjective(model, Min, my_obj(x...))


    if(!isnothing(da_bound))
        function d(x1, x2, t)
            return (x1 - x2)/t 
        end 

        function ∇d(g, x1, x2, t)
            g[1] = 1/t
            g[2] = -1/t
            g[3] = -(x1 - x2)/t^2
        end 

        register(model, :d, 3, d, ∇d; autodiff = false)
        println("-------------------------------------Applying d Bound-------------------------------------")
        for n in 1:n_controls 
            for i in 1:n_timesteps-1
                @NLconstraint(model, -da_bound <= d(x[i + (n-1) * n_timesteps+1],
                                                    x[i + (n-1) * n_timesteps],
                                                    x[end]) <= da_bound)
                
            end 
        end
    end

    if(!isnothing(dda_bound))
        function dd(x1, x2, x3, t)
            return (x1 - 2 * x2 + x3)/t^2 
        end 

        function ∇dd(g, x1, x2, x3, t)
            g[1] = 1/t^2
            g[2] = -2/t^2
            g[3] = 1/t^2
            g[4] = -2 * (x1 - 2*x2 + x3) / t^3
        end

        register(model, :dd, 4, dd, ∇dd; autodiff = false)
        println("-------------------------------------Applying dda Bound-------------------------------------")
        for n in 1:n_controls 
            for i in 1:n_timesteps-2
                @NLconstraint(model, -dda_bound <= dd(x[i + (n-1) * n_timesteps+2],
                                                    x[i + (n-1) * n_timesteps+1],
                                                    x[i + (n-1) * n_timesteps],
                                                    x[end]) <= dda_bound)
                
            end 
        end
    end

    if(!isnothing(target_fidelity))
        infid = unitary_infidelity_objective(n_controls, sys, 1, U_goal)
        infig_grad = unitary_infidelity_objective(n_controls, sys, 1, U_goal)

        function fid(x...)
            return 1-infid(collect(x))
        end 

        function fid_grad(g::AbstractVector, x...)
            grad = -infig_grad(collect(x))
            copyto!(g, grad) 
            return nothing
        end
        

        register(model, :fid, n, fid, fid_grad; autodiff = false)
        println("-------------------------------------Applying Fidelity Constraint-------------------------------------")
        @NLconstraint(model, fid(x...) >= target_fidelity)
            
    end

    JuMP.optimize!(model)
    solution = value.(x)
    push!(hist, solution)
    a = vec_to_components(solution[1:end-1], n_controls)

    return obj_func, a, hist 
end 

# ### Smooth Initial Controls ###
using FFTW 
function filter_sine_series(x, N)
    n = length(x)
    
    # Forward sine transform (DST-I)
    x_sine = FFTW.r2r(x[2:end-1], FFTW.RODFT00)
    
    # Apply frequency filter
    freqs = 1:(n-2)
    mask = freqs .<=  N
    x_sine_filtered = x_sine .* mask
    
    # Inverse sine transform
    x_filtered_interior = FFTW.r2r(x_sine_filtered, FFTW.RODFT00) / (2*(n-1))
    
    # Reconstruct with zero boundaries
    x_filtered = vcat([0.0], x_filtered_interior, [0.0])
    return x_filtered
end

function random_smooth_controls(n_controls, n_timesteps, a_bound, da_bound, dda_bound)
    a_init = (rand(n_controls, n_timesteps) .- 0.5) * 2 * a_bound
    a_init[:,1] *= 0 
    a_init[:, end] *= 0
    a_smooth = zeros(size(a_init))

    N = n_controls * n_timesteps
    for i in 1:N
        for c in 1:n_controls
            a_smooth[c, :] = filter_sine_series(a_init[c, :], N+1-i)
        end
        datavec = [vec(a_smooth'); Δt]

        if(maximum(abs.(second_derivative(n_controls, datavec))) <= dda_bound 
            && 
            maximum(abs.(first_derivative(n_controls, datavec))) <= da_bound)
            break 
        end
    end
    return a_smooth
end
