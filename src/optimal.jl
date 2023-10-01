#this is for problem set there
module BLP
using Distributions
import NLsolve
using LinearAlgebra

include("/Users/ella/Library/CloudStorage/Dropbox/ubc/econ622/src/integrate.jl")

@doc raw"""
    share(δ, Σ, dFν, x)

Computes shares in random coefficient logit with mean tastes `δ`, observed characteristics `x`, unobserved taste distribution `dFν`, and taste covariances `Σ`. 
Assumes there is an outside option with u=0. The outside option has share `1-sum(s)`

# Arguments

- `δ` vector of length `J`
- `Σ` `K` by `K` matrix
- `x` `J` by `K` array
- `∫` AbstractIntegrator for integrating over distribution of `ν`

# Returns

- vector of length `J` consisting of $s_1$, ..., $s_J$
"""
#this is the initial version of the code
function share(δ, Σ, x, ∫::Integrate.AbstractIntegrator)
  J,K = size(x)
  (length(δ) == J) || error("length(δ)=$(length(δ)) != size(x,1)=$J")
  (K,K) == size(Σ) || error("size(x,2)=$K != size(Σ)=$(size(Σ))")
  function shareν(ν)
    s = δ .+ x*Σ*ν
    smax=max(0,maximum(s))
    s .-= smax
    s .= exp.(s)
    s ./= (sum(s) + exp(0-smax))
    return(s)
  end
  return(∫(shareν))
end

using BenchmarkTools

# Create dummy data for the function.
J, K = 5, 5
δ = rand(J)
x = rand(J, K)

A = rand(K, K)
Σ = A*A' + K*I  # A*A' 
dist = MvNormal(Σ)  
integrator = Integrate.MonteCarloIntegrator(dist, 100)
@btime BLP.share($δ, $Σ, $x, $integrator)

#this is the optimized version of the code
function share_optimized(δ, Σ, x, ∫::Integrate.AbstractIntegrator)
    J, K = size(x)
    (length(δ) == J) || error("length(δ)=$(length(δ)) != size(x,1)=$J")
    (K, K) == size(Σ) || error("size(x,2)=$K != size(Σ)=$(size(Σ))")

    s = Vector{Float64}(undef, J)
    function shareν(ν)
        mul!(s, x, Σ*ν)  # In-place matrix-vector multiplication
        s .+= δ
        smax = max(0, maximum(s))
        s .-= smax
        s .= exp.(s)
        s ./= (sum(s) + exp(0 - smax))
        return s
    end
    return ∫(shareν)
end
end

b_old = @benchmark share($δ, $Σ, $x, $integrator)
b_new = @benchmark share_optimized($δ, $Σ, $x, $integrator)

println(b_old)
println(b_new)



