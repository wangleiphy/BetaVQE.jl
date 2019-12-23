module ThermalVQE
using Yao, LinearAlgebra
using Flux.Optimise: ADAM, update!
using StatsBase
using VAN

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

function train(β::Real, H::AbstractBlock{N}, sampler::AbstractSampler, circuit::AbstractBlock{N}, logfile=nothing;
                optimizer=ADAM(0.1), nbatch::Int=1000, niter::Int=100) where N
    ϕ = model_parameters(sampler)
    θ = parameters(circuit)
    for i = 1:niter
        _, _, gϕ, gθ, _ = gradient(loss, β, H, sampler, circuit, nbatch)
        update!.(Ref(optimizer), ϕ, gϕ)
        update!(optimizer, θ, gθ)
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
    ϕ, θ
end

end # module
