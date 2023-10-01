using Test
using Distributions
using LinearAlgebra

include("../src/integrate.jl")

@testset "Gauss-Legendre Quadrature Tests" begin
    
    f(x) = x^2
    a, b = 0, 2
    result = Integrate.gauss_legendre_integral(f, a, b, 10)
    expected = (b^3 - a^3) / 3  # Analytical integral of x^2 is x^3/3
    @test isapprox(result, expected, atol=1e-10)

    g(x) = x^4 + x^3 - 3x + 1
    a, b = -1, 1
    result = Integrate.gauss_legendre_integral(g, a, b, 10)
    expected = (b^5/5 + b^4/4 - 3b^2/2 + b) - (a^5/5 + a^4/4 - 3a^2/2 + a)
    @test isapprox(result, expected, atol=1e-10)
end

@testset "Sparse Grid Quadrature Tests" begin
    dist = MvNormal([0.0, 0.0], I) 

    f(vec) = vec[1]^2 + vec[2]^2
    
    result = Integrate.∫sgq(f, dist, order=5)
    
    @test isapprox(result, 2.0, atol=1e-5)
end
