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

export rv_to_mean_elements_sgp4, rv_to_tle

"""
    rv_to_mean_elements_sgp4(vJD::AbstractVector{T}, vr::AbstractVector{Tv}, vv::AbstractVector{Tv}, W = I; max_it = 10000, sgp4_gc = sgp4_gc_wgs84) where {T,Tv<:AbstractVector}

Compute the mean elements for SGP4 based on the position `vr` and velocity
vectors `vr` represented in TEME reference frame. The epoch of those
measurements [Julian Day] must be in `vJD`.

The matrix `W` defined the weights for the least-square algorithm.

The variable `max_it` defines the maximum allowed number of iterations.

The variable `sgp4_gc` defines which constants should be used when running SGP4.

# Returns

* The epoch of the elements [Julian Day].
* The mean elements for SGP4 algorithm:
    - Mean motion [rad/s];
    - Eccentricity [];
    - Inclination [rad];
    - Right ascension of the ascending node [rad];
    - Argument of perigee [rad];
    - Mean anomaly [rad];
    - BSTAR.
* The covariance matrix of the mean elements estimation.

"""
function rv_to_mean_elements_sgp4(vJD::AbstractVector{T},
                                  vr::AbstractVector{Tv},
                                  vv::AbstractVector{Tv},
                                  W = I;
                                  max_it::Int = 25,
                                  sgp4_gc = sgp4_gc_wgs84,
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
    x₀ = SVector{7,T}(vr[end]..., vv[end]..., 0.0001)
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
            r̂, v̂, ~ = _sgp4_sv_si(Δt, sgp4_gc, epoch, x₀)
            ŷ = vcat(r̂, v̂)

            # Compute the error.
            b = y - ŷ

            # Compute the Jacobian.
            A = _sgp4_jacobian(Δt, epoch, x₀, ŷ)

            # Accumulation.
            ΣAᵀWA += A'*W*A
            ΣAᵀWb += A'*W*b
            vσ²   += W*(b.^2)
        end

        # Update the estimate.
        P  = pinv(ΣAᵀWA)
        δx = P*ΣAᵀWb

        x₁ = x₀ + δx

        # Limit B* change to avoid divergence.
        x₁ = setindex(x₁, (1 + 0.001sign(δx[7]))*x₀[7], 7)

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

"""
    rv_to_tle(args...; name::String = "UNDEFINED", sat_num::Int = 9999, classification::Char = 'U', int_designator = "999999", elem_set_number::Int = 0, rev_num, kwargs...)

Convert a set of position and velocity vectors represented in TEME reference
frame to a TLE. The arguments `args` and keywords `kwargs` are the same as those
described in the function `rv_to_mean_elements_sgp4`.

Additionally, the user can specify some parameters of the generated TLE.

This function prints the TLE to `stdout` using the function `print_tle` and also
returns the TLE string.

"""
function rv_to_tle(args...;
                   name::String = "UNDEFINED",
                   sat_num::Int = 9999,
                   classification::Char = 'U',
                   int_designator = "999999",
                   elem_set_number::Int = 0,
                   rev_num::Int = 0,
                   sgp4_gc = sgp4_gc_wgs84,
                   kwargs...)

    # Convert the position and velocity vectors to mean elements.
    JD, x, P = rv_to_mean_elements_sgp4(args...; sgp4_gc = sgp4_gc, kwargs...)

    # Compute the data as required by the TLE format.
    dt  = JDtoDate(DateTime, JD)
    dt₀ = JDtoDate(DateTime, DatetoJD(year(dt), 1, 1, 0, 0, 0))

    dt_year    = year(dt)
    epoch_year = dt_year < 1980 ? dt_year - 1900 : dt_year - 2000
    epoch_day  = (dt-dt₀).value/1000/86400 + 1

    # Compute the orbit elements from the state vector `x`.
    orb = rv_to_kepler(x[1], x[2], x[3], x[4], x[5], x[6])

    a_0 = orb.a/(1000sgp4_gc.R0)
    e_0 = orb.e
    i_0 = orb.i*180/pi
    Ω_0 = orb.Ω*180/pi
    ω_0 = orb.ω*180/pi
    M_0 = f_to_M(e_0, orb.f)*180/pi

    bstar = x[7]

    # Compute the mean motion.
    n_0 = (sgp4_gc.XKE / sqrt(a_0 * a_0 * a_0))*720/pi

    # Construct the TLE.
    tle = TLE(name,
              sat_num,
              classification,
              int_designator,
              epoch_year,
              epoch_day,
              JD,
              0.0,
              0.0,
              bstar,
              elem_set_number,
              0,
              i_0,
              Ω_0,
              e_0,
              ω_0,
              M_0,
              n_0,
              rev_num,
              0)

    tle_str = tle_to_str(tle)

    # Print the TLE to the `stdout`.
    print(tle_str)

    # Return the TLE string.
    return tle_str
end


################################################################################
#                              Private functions
################################################################################

# Compute the SGP4 algorithm considering all variables in SI.
function _sgp4_sv_si(Δt, sgp4_gc, epoch, sv::AbstractVector{T}) where T
    # Convert the state-vector to Keplerian elements.
    orb = rv_to_kepler(sv[1], sv[2], sv[3], sv[4], sv[5], sv[6])

    a_0 = orb.a/(1000sgp4_gc.R0)
    e_0 = orb.e
    i_0 = orb.i
    Ω_0 = orb.Ω
    ω_0 = orb.ω
    M_0 = f_to_M(e_0, orb.f)

    # Compute the mean motion using the provided constants.
    n_0 = sgp4_gc.XKE / sqrt(a_0 * a_0 * a_0)

    # Check if the state vector has the bstar parameter.
    bstar = length(sv) > 6 ? sv[7] : T(0)

    # Propagate the orbit.
    r, v, sgp4d = sgp4(Δt/60, sgp4_gc, epoch, n_0, e_0, i_0, Ω_0, ω_0, M_0, bstar)

    return 1000r, 1000v, sgp4d
end

function _sgp4_jacobian(Δt, JD₀, x₀::AbstractVector{T}, y₀::AbstractVector{T};
                        pert::T = 1e-3,
                        pertthreshold::T = 1e-7,
                        sgp4_gc = sgp4_gc_wgs84) where T

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
        r, v, ~ = _sgp4_sv_si(Δt, sgp4_gc, JD₀, x₁)
        y₁  = vcat(r, v)

        M[i,j] = (y₁[i] - y₀[i])/ϵ

        x₁ = setindex(x₁, x₀[j], j)
    end

    return M
end
