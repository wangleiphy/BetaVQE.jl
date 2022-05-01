using Test, Random, StatsBase
using Zygote
using BetaVQE.VAN
using BetaVQE.VAN:bitarray, get_energy, free_energy, network

@testset "normalize" begin
    Random.seed!(2)
    nbits = 6
    model = PSAModel(nbits)
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
    β = 1.0
    model = PSAModel(nbits)

    configs = bitarray(collect(0:(1<<nbits-1)), nbits)
    logp = get_logp(model, configs)
    f = sum(exp.(logp) .* (get_energy(K, configs) + get_logp(model, configs)))

    samples = gen_samples(model, nsamples)
    f_sample = free_energy(K, model, samples)
    @test isapprox(f, f_sample, rtol=1e-2)
end

@testset "gradscorefunction" begin
    Random.seed!(2)
    nbits = 6
    nbatchs = 10000
    model = PSAModel(nbits)
    samples = gen_samples(model, nbatchs)

    score(model, samples) = mean(get_logp(model, samples))
    grad = gradient(score, model, samples)[1]
    @show grad.w 
    @test all(isapprox.(grad.w, 0.0, atol=1e-1))
end
