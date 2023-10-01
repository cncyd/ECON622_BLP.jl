using Test

@testset "integrator" begin
    include("../src/integrate.jl") # for interactive execution
    using Distributions

    dimx = 3
    A = rand(dimx,dimx)
    Σ = A*A'
    dx = MvNormal(zeros(dimx), Σ)
    ∫a = Integrate.AdaptiveIntegrator(dx, options=(rtol=1e-6,initdiv=3))        
    V = ∫a(x->x*x')
    @test isapprox(V, Σ, rtol=1e-5)
    f(x) = exp(x[1])/sum(exp.(x)) 
    val = ∫a(f)
    for N ∈ [1_000, 10_000, 100_000]
        ∫mc = Integrate.MonteCarloIntegrator(dx, N)
        ∫qmc = Integrate.QuasiMonteCarloIntegrator(dx,N)        
        @test isapprox(∫mc(x->x*x'), Σ, rtol=10/sqrt(N))
        @test isapprox(∫qmc(x->x*x'), Σ, rtol=10/sqrt(N))

        f(x) = exp(x[1])/sum(exp.(x))        
        @test isapprox(∫mc(f),val,rtol=1/sqrt(N))
        @test isapprox(∫qmc(f),val,rtol=1/sqrt(N))
    end
end

@testset "share=δ⁻¹"
    include("../src/blp.jl") 
    using LinearAlgebra
    J = 4
    dimx=2
    dx = MvNormal(dimx, 1.0)
    Σ = [1 0.5; 0.5 1]
    N = 1_000
    ∫ = BLP.Integrate.QuasiMonteCarloIntegrator(dx, N)
    X = [(-1.).^(1:J) 1:J]
    δ = collect((1:J)./J)
    s = BLP.share(δ,Σ,X,∫) 
    d = BLP.delta(s, Σ, X, ∫)
    @test d ≈ δ

    J = 10
    dimx = 4
    X = rand(J, dimx)
    dx = MvNormal(dimx, 1.0)
    Σ = I + ones(dimx,dimx)
    ∫ = BLP.Integrate.QuasiMonteCarloIntegrator(dx, N)
    δ = 1*rand(J)
    s = BLP.share(δ,Σ,X,∫) 
    d = BLP.delta(s, Σ, X, ∫)
    @test isapprox(d, δ, rtol=1e-6)
    
end


@testset "1D Integrator Functions" begin
    # Define a test function and its integral under standard normal distribution
    f(x) = x^2
    analytic_integral = 1.0 # For standard normal distribution

    μ, σ = 0.0, 1.0
    @test isapprox(∫q_1d(f, μ, σ, ndraw=100), analytic_integral, rtol=1e-5)
    @test isapprox(∫sgq_1d(f, μ, σ, order=5), analytic_integral, rtol=1e-5)

    # You can add more tests as needed
end

