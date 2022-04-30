using BetaVQE
using Yao, Yao.EasyBuild, Yao.BitBasis
using Test, Random, StatsBase
using Zygote
using VAN

@testset "sample" begin
    Random.seed!(3)
    nbits = 4
    nhiddens = [10, 20]
    nsamples = 1000
    β = 1.0

    h = heisenberg(nbits)
    c = dispatch!(variational_circuit(nbits), :random)
    model = AutoRegressiveModel(nbits, nhiddens)

    configs = bitarray(collect(0:(1<<nbits-1)), nbits)
    logp = get_logp(model, configs)
    @test isapprox(sum(exp.(logp)), 1.0, rtol=1e-2)

    f = sum(exp.(logp) .* free_energy_local(β, h, model, c, configs))

    samples = gen_samples(model, nsamples)
    f_sample = free_energy(β, h, model, c, samples)
    @test isapprox(f, f_sample, rtol=1e-1)
end

@testset "circuit diff" begin
    Random.seed!(3)
    nbits = 4
    nhiddens = [10]
    nsamples = 2000

    h = heisenberg(nbits)
    c = dispatch!(variational_circuit(nbits), :random)
    model = AutoRegressiveModel(nbits, nhiddens)
    samples = gen_samples(model, nsamples)

    # check the gradient of circuit parameters
    params = parameters(c)
    ϵ = 1e-5
    for k in 1:length(params)
        params[k] -= ϵ
        dispatch!(c, params)
        l1 = free_energy(2.0, h, model, c, samples)
        params[k] += 2ϵ
        dispatch!(c, params)
        l2 = free_energy(2.0, h, model, c, samples)
        params[k] -= ϵ
        dispatch!(c, params)
        g = (l2-l1)/2ϵ
        g2 = gradient(c->free_energy(2.0, h, model, c, samples), c)[1]
        @test isapprox(g, g2[k], rtol=1e-2)
    end
end

@testset "network diff" begin
    Random.seed!(7)
    nbits = 2
    nhiddens = [10]
    nsamples = 1000

    h = heisenberg(nbits)
    c = dispatch!(variational_circuit(nbits), :random)
    model = AutoRegressiveModel(nbits, nhiddens)

    ϵ = 1e-5
    # check the gradient of model
    configs = bitarray(collect(0:(1<<nbits-1)), nbits)
    function f(model)
        logp = get_logp(model, configs)
        sum(exp.(logp) .* free_energy_local(2.0, h, model, c, configs))
    end

    m, n = 1, 1
    params = model_parameters(model)
    params[m][n] -= ϵ
    model_dispatch!(model, params)
    l1 = f(model)
    params[m][n] += 2ϵ
    model_dispatch!(model, params)
    l2 = f(model)
    g = (l2-l1)/2ϵ

    #tune it back
    params[m][n] -= ϵ
    model_dispatch!(model, params)
    samples = gen_samples(model, nsamples)
    g2 = gradient(model->free_energy(2.0, h, model, c, samples), model)[1]
    @test isapprox(g, g2.W[1][n], rtol=1e-1)
    @show g, g2.W[1][n]
end

@testset "tns circuit diff" begin
    Random.seed!(3)
    nx = 2
    ny = 2
    nbits = nx*ny
    depth = 3
    nhiddens = [10]
    nsamples = 2000

    h = heisenberg(nbits)
    c = tns_circuit(nbits, depth, EasyBuild.pair_square(nx, ny; periodic=false); entangler=(n,i,j)->put(n,(i,j)=>general_U4(rand(15)*2π)))
    model = AutoRegressiveModel(nbits, nhiddens)
    samples = gen_samples(model, nsamples)

    # check the gradient of circuit parameters
    params = parameters(c)
    ϵ = 1e-5
    for k in 1:length(params)
        params[k] -= ϵ
        dispatch!(c, params)
        l1 = free_energy(2.0, h, model, c, samples)
        params[k] += 2ϵ
        dispatch!(c, params)
        l2 = free_energy(2.0, h, model, c, samples)
        params[k] -= ϵ
        dispatch!(c, params)
        g = (l2-l1)/2ϵ
        g2 = gradient(c->free_energy(2.0, h, model, c, samples), c)[1]
        @test isapprox(g, g2[k], atol=1e-2)
    end
end


@testset "train" begin
    nbits = 4
    h = hamiltonian(TFIM(nbits, 1; Γ=0.0, periodic=false))
    network = PSAModel(nbits)
    circuit = tns_circuit(nbits, 2, EasyBuild.pair_square(nbits, 1; periodic=false); entangler=(n,i,j)->put(n,(i,j)=>general_U4()))
    network_params, circuit_params = train(1.0, h, network, circuit; nbatch=1000, niter=100)
    samples = gen_samples(network, 1000)
    @test free_energy(1.0, h, network, circuit, samples) <= -3.2
end