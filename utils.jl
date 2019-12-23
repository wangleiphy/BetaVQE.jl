using FileIO
using Yao
using YaoExtensions
using ThermalVQE
using LinearAlgebra

function build_key(nx, ny, Γ, β, depth, nsamples, nhiddens, lr; folder="data/")
    mkpath(folder)
    key = folder*"/tfim"
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
    key
end

function exact_spectra(nx::Int, ny::Int, Γ::Real; folder="data/exact/")
    mkpath(folder)
    h = hamiltonian(TFIM(nx, ny; Γ=Γ, periodic=false))
    H = mat(h)
    w, _ = eigen(Matrix(H))
    save(folder*"/nx$nx"*"ny$ny"*"Gamma$Γ"*".jld2", "spectra", w)
end
