module Integrate

using Distributions
import Sobol: skip, SobolSeq
import Base.Iterators: take, Repeated
import HCubature: hcubature
import LinearAlgebra: cholesky
using QuadGK: quadgk          # for quadrature
using SparseGrids             # assuming there's a package for sparse grids

abstract type AbstractIntegrator end

(∫::AbstractIntegrator)(f::Function) = sum(w*f(x) for (w,x) in zip(∫.w, ∫.x))

struct FixedNodeIntegrator{Tx,Tw} <: AbstractIntegrator
    x::Tx
    w::Tw
end

MonteCarloIntegrator(distribution::Distribution, ndraw=100)=FixedNodeIntegrator([rand(distribution) for i=1:ndraw], Repeated(1/ndraw))

function QuasiMonteCarloIntegrator(distribution::UnivariateDistribution, ndraws=100)
    ss = skip(SobolSeq(1), ndraw)
    x = [quantile(distribution, x) for x in take(ss,ndraw)]
    w = Repeated(1/ndraw)
    FixedNodeIntegrator(x,w)
end 

function QuasiMonteCarloIntegrator(distribution::AbstractMvNormal, ndraw=100)
    ss = skip(SobolSeq(length(distribution)), ndraw)
    L = cholesky(distribution.Σ).L
    x = [L*quantile.(Normal(), x) for x in take(ss,ndraw)]
    w = Repeated(1/ndraw)
    FixedNodeIntegrator(x,w)
end 


struct AdaptiveIntegrator{FE,FT,FJ,A,L} <: AbstractIntegrator
    eval::FE
    transform::FT
    detJ::FJ
    args::A
    limits::L
end

(∫::AdaptiveIntegrator)(f::Function) = ∫.eval(t->f(∫.transform(t))*∫.detJ(t), ∫.limits...; ∫.args...)[1]

function AdaptiveIntegrator(dist::AbstractMvNormal; eval=hcubature, options=())
    D = length(dist)
    x(t) = t./(1 .- t.^2)
    Dx(t) = prod((1 .+ t.^2)./(1 .- t.^2).^2)*pdf(dist,x(t))
    args = options
    limits = (-ones(D), ones(D))
    AdaptiveIntegrator(hcubature,x,Dx,args, limits)
end

using FastGaussQuadrature, LinearAlgebra
import Base.Iterators: product, repeated
function ∫q(f, dx::MvNormal; ndraw=100)
  n = Int(ceil(ndraw^(1/length(dx))))
  x, w = gausshermite(n)
  L = cholesky(dx.Σ).L
  sum(f(√2*L*vcat(xs...) + dx.μ)*prod(ws)
      for (xs,ws) ∈ zip(product(repeated(x, length(dx))...),
                        product(repeated(w, length(dx))...))
        )/(π^(length(dx)/2))
end

using SparseGrids
function ∫sgq(f, dx::MvNormal; order=5)
  X, W = sparsegrid(length(dx), order, gausshermite, sym=true)
  L = cholesky(dx.Σ).L
  sum(f(√2*L*x + dx.μ)*w for (x,w) ∈ zip(X,W))/(π^(length(dx)/2))
end

# Testing
dist = Normal(0, 1)
f(x) = x^2

integrator = SparseGridQuadratureIntegrator(dist, 5)
result = integrator(f)

println("Sparse Grid Result: ", result)

known_result = 1.0  # E[X^2] for standard normal
println("Error for Sparse Grid: ", abs(result - known_result))

end