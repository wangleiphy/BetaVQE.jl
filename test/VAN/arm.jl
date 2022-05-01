using Test, Random, StatsBase
using Zygote
using BetaVQE.VAN
using BetaVQE.VAN:bitarray, get_energy, free_energy, network

@testset "normalize" begin
    Random.seed!(2)
    nbits = 6
    nhiddens = [10,10]
    model = AutoRegressiveModel(nbits, nhiddens)
    configs = bitarray(collect(0:(1<<nbits-1)), nbits)
    logp = get_logp(model, configs)
    norm = sum(exp.(logp))

    @test isapprox(norm, 1.0, rtol=1e-5)
end

@testset "sample" begin
    Random.seed!(3)
    nbits = 4
    K = randn(nbits, nbits)
    K = (K+K')/2
    nsamples = 5000
    nhiddens = [100]
    β = 1.0
    model = AutoRegressiveModel(nbits, nhiddens)

    configs = bitarray(collect(0:(1<<nbits-1)), nbits)
    logp = get_logp(model, configs)
    f = sum(exp.(logp) .* (get_energy(K, configs) + get_logp(model, configs)))

    samples = gen_samples(model, nsamples)
    f_sample = free_energy(K, model, samples)
    @test Zygote.gradient(free_energy, K, model, samples)[2] isa NamedTuple
    @test isapprox(f, f_sample, rtol=1e-2)
end

@testset "autoregressive" begin
    Random.seed!(2)
    nbits = 6
    nbatchs = 4
    nhiddens = [10, 20, 30]
    model = AutoRegressiveModel(nbits, nhiddens)
    samples = rand(0:1, nbits, nbatchs)

    f(model, samples, n, b) = network(model, samples)[n, b]

    for n in 1:nbits
        for b in 1:nbatchs
            g = gradient(f, model, samples, n, b)[2]
            dependency = (g .!= 0)
            correct = BitArray( x< n && y==b  for x = 1:nbits, y = 1:nbatchs)
            @test all(dependency .<= correct)
        end
    end
end

@testset "gradscorefunction" begin
    Random.seed!(2)
    nbits = 6
    nbatchs = 10000
    nhiddens = [10, 20]
    model = AutoRegressiveModel(nbits, nhiddens)
    samples = gen_samples(model, nbatchs)

    score(model, samples) = mean(get_logp(model, samples))
    grad = gradient(score, model, samples)[1]
    δ = collect(Iterators.flatten((grad.W..., grad.b...)))
    @test all(isapprox.(δ, 0.0, atol=1e-2))
end
