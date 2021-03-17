# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description
#
#   Functions to convert osculating elements to mean elements.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# References
#
#   [1] Vallado, D. A; Crawford, P (2008). SGP4 orbit determination. AIAA/AAS
#       Astrodynamics Specialist Conference, Honoulu, HI.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

export rv_to_mean_elements_j2

function rv_to_mean_elements_j2(vJD::AbstractVector{T},
                                vr::AbstractVector{Tv},
                                vv::AbstractVector{Tv},
                                W = I;
                                max_it::Int = 25,
                                j2_gc = j2_gc_egm08,
                                tol::T = 2e-4) where
    {T,Tv<:AbstractVector}

    # Number of measurements.
    num_meas = length(vr)

    # Dimension of the measurements.
    meas_dim = length(first(vr)) + length(first(vv))

    # The epoch of the mean elements will be the newest observation.
    epoch = vJD[end]

    # Initial guess of the mean elements, which is represented by a
    # state-vector.
    x₀ = SVector{6,T}(vr[end]..., vv[end]...)
    x₁ = x₀

    # Number of states that will be fitted.
    num_states = length(x₀)

    P  = SMatrix{num_states, num_states, T}(I)

    # Residuals of the iterations.
    σᵢ   = T(10000)
    σᵢ₋₁ = T(20000)
    σᵢ₋₂ = T(20000)

    # Loop until the maximum allowed iteration.
    @inbounds for it = 1:max_it
        x₀ = x₁

        # Variables to store the summations to compute the least square fitting
        # algorithm.
        ΣAᵀWA = @SMatrix zeros(num_states, num_states)
        ΣAᵀWb = @SVector zeros(num_states)
        vσ²   = @SVector zeros(T, meas_dim)

        # We will interpolate backwards in time, so that the mean elements
        # in x₁ are related to the newest measurement.
        for k = num_meas:-1:1
            # Obtain the measured ephemerides.
            y = vcat(vr[k], vv[k])

            # Obtain the computed ephemerides considering the current estimate
            # of the mean elements.
            Δt = (vJD[k] - epoch)*86400
            r̂, v̂ = _j2_sv(Δt, j2_gc, epoch, x₀)
            ŷ = vcat(r̂, v̂)

            # Compute the error.
            b = y - ŷ

            # Compute the Jacobian.
            A = _j2_jacobian(Δt, epoch, x₀, ŷ)

            # Accumulation.
            ΣAᵀWA += A'*W*A
            ΣAᵀWb += A'*W*b
            vσ²   += W*(b.^2)
        end

        # Update the estimate.
        P  = pinv(ΣAᵀWA)
        δx = P*ΣAᵀWb

        x₁ = x₀ + δx

        # Compute the residuals in this iteration.
        σᵢ₋₂ = σᵢ₋₁
        σᵢ₋₁ = σᵢ
        σᵢ   = sqrt.(sum(vσ²)/num_meas)

        # Compute the residual change between iterations.
        σp = abs((σᵢ-σᵢ₋₁)/σᵢ₋₁)

        println("PROGRESS: $it, σᵢ = $(round(σᵢ, digits = 4)), σp = $(round(σp, digits = 4))")

        # Check the convergence.
        if (σp < tol) || (σᵢ < tol)
            break
        end

        # Check if we are diverging.
        if (σᵢ > σᵢ₋₁ > σᵢ₋₂) && (σᵢ > 5e5)
            error("Iterations diverged!")
        end
    end

    # Return the mean elements for SGP4 and the covariance matrix.
    return vJD[end], x₁, P
end

################################################################################
#                              Private functions
################################################################################

# Compute the SGP4 algorithm considering all variables in SI.
function _j2_sv(Δt, j2_gc, epoch, sv::AbstractVector{T}) where T
    # Convert the state-vector to Keplerian elements.
    orb = rv_to_kepler(sv[1], sv[2], sv[3], sv[4], sv[5], sv[6])

    j2oscd = j2_init(j2_gc, epoch, orb.a, orb.e, orb.i, orb.Ω, orb.ω, orb.f, 0, 0)
    r, v = j2!(j2oscd, Δt)

    return r, v
end

function _j2_jacobian(Δt, JD₀, x₀::AbstractVector{T}, y₀::AbstractVector{T};
                      pert::T = 1e-5,
                      pertthreshold::T = 1e-7,
                      j2_gc = j2_gc_egm08) where T

    num_states = length(x₀)
    dim_obs    = length(y₀)
    M          = Matrix{T}(undef, dim_obs, num_states)

    # Auxiliary variables.
    x₁ = copy(x₀)

    @inbounds for i = 1:dim_obs, j = 1:num_states
        # This algorithm that perturbs each element is similar to Vallado's.
        α = x₁[j]
        ϵ = T(0)
        pert_i = pert

        for _ = 1:5
            ϵ  = α * pert_i
            abs(ϵ) > pertthreshold && break
            pert_i *= 1.4
        end

        # Avoid a division by 0 if α is very small.
        abs(ϵ) < pertthreshold && (ϵ = sign(ϵ) * pertthreshold)

        α += ϵ

        x₁ = setindex(x₁, α, j)
        r, v = _j2_sv(Δt, j2_gc, JD₀, x₁)
        y₁  = vcat(r, v)

        M[i,j] = (y₁[i] - y₀[i])/ϵ

        x₁ = setindex(x₁, x₀[j], j)
    end

    return M
end
