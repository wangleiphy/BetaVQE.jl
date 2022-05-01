using Comonicon
using JLD2
using LinearAlgebra
using BetaVQE.VAN
using Yao
using Yao.EasyBuild
using BetaVQE
using Optimisers: ADAM
using StatsBase
using Random

# file storage
function build_key(nx, ny, Γ, β, depth, nsamples, nhiddens, lr; folder="data")
    mkpath(folder)
    key = "tfim"
    key *= "_nx$nx"
    key *= "_ny$ny"
    key *= "_Gamma$Γ"
    key *= "_beta$β"
    key *= "_d$depth"
    key *= "_batch$nsamples"
    key *= "_lr$lr"
    key *= "_nhiddens"

    for h in nhiddens
        key = key * "_$h"
    end
    return joinpath(folder, key)
end

@cast function exact_spectra(nx::Int, ny::Int, Γ::Float64; folder=joinpath("data", "exact"))
    mkpath(folder)
    h = hamiltonian(TFIM(nx, ny; Γ=Γ, periodic=false))
    H = mat(h)
    w, _ = eigen(Matrix(H))
    save(joinpath(folder, "nx$nx"*"ny$ny"*"Gamma$Γ"*".jld2"), "spectra", w)
end

@cast function scan_beta(nx::Int=2, ny::Int=2, Γ::Float64=1.0;
                         depth::Int=5, nsamples::Int=1000, nhiddens::Vector{Int}=[500], lr::Float64=0.01, niter::Int=500, cont::Bool=false)

    for β in collect(0.1:0.1:1.0)
        learn(nx, ny, Γ, β; depth=depth, nsamples=nsamples, nhiddens=nhiddens, lr=lr, niter=niter, cont=cont)
    end

end

@cast function scan_gamma(nx::Int=2, ny::Int=2, β::Float64=1.0;
                         depth::Int=5, nsamples::Int=1000, nhiddens::Vector{Int}=[500], lr::Float64=0.01, niter::Int=500, cont::Bool=false)

    for Γ in collect(0.0:1.0:4.0)
        learn(nx, ny, Γ, β; depth=depth, nsamples=nsamples, nhiddens=nhiddens, lr=lr, niter=niter, cont=cont)
    end

end

@cast function learn(nx::Int=2, ny::Int=2, Γ::Float64=1.0, β::Float64=1.0;
                  depth::Int=5, nsamples::Int=1000, nhiddens::Vector{Int}=[500], lr::Float64=0.01, niter::Int=500, cont::Bool=false)

    Random.seed!(42)
    key = build_key(nx, ny, Γ, β, depth, nsamples, nhiddens, lr; folder=joinpath("data", "tns3"))
    println(key)

    nbits = nx*ny

    if nhiddens[1] == 0
        network = PSAModel(nbits)
    else
        network = AutoRegressiveModel(nbits, nhiddens)
    end

    circuit = tns_circuit(nbits, depth, EasyBuild.pair_square(nx, ny; periodic=false); entangler=(n,i,j)->put(n,(i,j)=>general_U4()))

    h = hamiltonian(TFIM(nx, ny; Γ=Γ, periodic=false))
    F_exact, E_exact, S_exact, Cv_exact, γ_exact = BetaVQE.exact(β, h)

    chkp_file = key*".jld2"
    if cont && isfile(chkp_file)
        chkp = load(chkp_file)
        model_dispatch!(network, chkp["cparams"])
        dispatch!(circuit, chkp["qparams"])
        println("load chkp from $chkp_file")
    else
        message = "# $F_exact"
        println(message)

        logfile = open(key*".log", "w")
        write(logfile, message*"\n")
        close(logfile)
    end

    logfile = open(key*".log", "a")
    cparams, qparams = BetaVQE.train(β, h, network, circuit, logfile; optimizer=ADAM(lr), nbatch=nsamples, niter=niter)
    close(logfile)

    #compute observables
    samples = gen_samples(network, nsamples)
    E = BetaVQE.energy(h, network, circuit, samples)
    E2 = BetaVQE.energy2(h, network, circuit, samples)
    S = BetaVQE.entropy(h, network, circuit, samples)
    γ = BetaVQE.purity(h, network, circuit, samples)
    F = E - S/β

    Cv = β^2*(E2 - E^2)

    save(key*".jld2", "cparams", cparams, "qparams", qparams, "exact", (F_exact, E_exact, S_exact, Cv_exact, γ_exact), "result", (F, E, S, Cv, γ))
end

@cast function inference(nx::Int=2, ny::Int=2, Γ::Float64=1.0, β::Float64=1.0;
                  depth::Int=5, nsamples::Int=1000, nhiddens::Vector{Int}=[500], lr::Float64=0.01, niter::Int=500)

    Random.seed!(42)
    key = build_key(nx, ny, Γ, β, depth, nsamples, nhiddens, lr; folder=joinpath("data", "tns3"))
    println(key)

    nbits = nx*ny

    if nhiddens[1] == 0
        network = PSAModel(nbits)
    else
        network = AutoRegressiveModel(nbits, nhiddens)
    end

    circuit = tns_circuit(nbits, depth, EasyBuild.pair_square(nx, ny; periodic=false); entangler=(n,i,j)->put(n,(i,j)=>general_U4()))

    h = hamiltonian(TFIM(nx, ny; Γ=Γ, periodic=false))

    chkp_file = key*".jld2"
    if isfile(chkp_file)
        chkp = load(chkp_file)
        model_dispatch!(network, chkp["cparams"])
        dispatch!(circuit, chkp["qparams"])
        println("load chkp from $chkp_file")
    else
        throw(ArgumentError("file not exists: $chkp_file"))
    end

    samples = gen_samples(network, nsamples)
    samples = unique(samples, dims=2)
    s = BetaVQE.spectra(h, network, circuit, samples)
    jldopen(chkp_file, "a+") do file
        file["spectra"] = s
    end
end

@main