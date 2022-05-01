module BetaVQE
using LinearAlgebra
using Yao, Yao.EasyBuild, Yao.BitBasis

using Zygote
import Optimisers
import ChainRulesCore: @non_differentiable, NoTangent, Tangent, rrule
using StatsBase
include("VAN/VAN.jl")
using .VAN

export qaoa_circuit, tns_circuit 
export free_energy, free_energy_local, energy, entropy
export TFIM, hamiltonian
export train, exact

include("hamiltonian.jl")
include("circuit.jl")
include("free_energy.jl")
include("zygote_patch.jl")

# exact value
function exact(β, h)
    H = mat(h)
    w, _ = eigen(Matrix(H))
    Z = sum(e->exp(-β*e), w)
    F = -log(Z)/β
    E = sum(e->exp(-β*e)*e, w)/Z
    E2 = sum(e->exp(-β*e)*e*e, w)/Z
    γ =  sum(e->exp(-2*β*e), w)/Z^2
    S = (E-F)*β
    Cv = (E2-E^2)*β^2
    F, E, S, Cv, γ
end

function loss(β, H, sampler, circuit, nbatch::Int)
    samples = gen_samples(sampler, nbatch)
    free_energy(β, H, sampler, circuit, samples)
end

function train(β::Real, H::AbstractBlock, sampler::AbstractSampler, circuit::AbstractBlock, logfile=nothing;
                optimizer=Optimisers.ADAM(0.1), nbatch::Int=1000, niter::Int=100)
    ϕ = model_parameters(sampler)
    θ = parameters(circuit)
    opt = Optimisers.setup(optimizer, (ϕ=ϕ,θ=θ))
    for i = 1:niter
        _, _, gϕ, gθ, _ = Zygote.gradient(loss, β, H, sampler, circuit, nbatch)
        Optimisers.update!(opt, (ϕ=ϕ, θ=θ), (ϕ=collect_gradients(sampler, gϕ), θ=gθ))
        model_dispatch!(sampler, ϕ)
        dispatch!(circuit, θ)
        message = "$i $(loss(β, H, sampler, circuit, nbatch))"
        println(message)
        flush(stdout)
        if !(logfile === nothing)
            write(logfile, message*"\n")
            flush(logfile)
        end
    end
    return ϕ, θ
end

# collect parameters into a tuple
collect_gradients(model::AutoRegressiveModel, g) = (g.W..., g.b...)
collect_gradients(model::PSAModel, g) = (g.w, )

end # module
