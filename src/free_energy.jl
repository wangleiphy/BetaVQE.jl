using BitBasis
using VAN:get_logp, unpack_gradient

export free_energy, free_energy_local, energy, entropy

function get_input_reg(nbits::Int, samples::AbstractArray)
    get_input_reg(nbits, Vector(packbits(samples)))
end

function get_input_reg(nbits::Int, samples::Vector{<:Integer})
    nbatch = length(samples)
    config = zeros(ComplexF64, nbatch, 1<<nbits)
    for i = 1:nbatch
        config[i,Int(samples[i])+1] = 1
    end
    # transpose the register for better performance
    return ArrayReg(transpose(config))
end

function free_energy(β::Real, H::AbstractBlock{N}, sampler::AbstractSampler, circuit::AbstractBlock{N}, samples) where N
    mean(free_energy_local(β, H, sampler, circuit, samples))
end

function energy(H::AbstractBlock{N}, sampler, circuit::AbstractBlock{N}, samples) where N
    reg = get_input_reg(N, samples)
    mean(real.(expect(H, reg|>circuit)))
end

function energy2(H::AbstractBlock{N}, sampler::AbstractSampler, circuit::AbstractBlock{N}, samples) where N
    reg = get_input_reg(N, samples)
    mean(real.(expect(H*H, reg|>circuit)))
end

function entropy(H::AbstractBlock{N}, sampler::AbstractSampler, circuit::AbstractBlock{N}, samples) where N
    -mean(get_logp(sampler, samples))
end

function purity(H::AbstractBlock{N}, sampler::AbstractSampler, circuit::AbstractBlock{N}, samples) where N
    mean(exp.(get_logp(sampler, samples)))
end

function spectra(H::AbstractBlock{N}, sampler::AbstractSampler, circuit::AbstractBlock{N}, samples) where N
    reg = get_input_reg(N, samples)
    real.(expect(H, reg|>circuit))
end

function free_energy_local(β::Real, H::AbstractBlock{N}, sampler, circuit::AbstractBlock{N}, samples) where N
   reg = get_input_reg(N, samples)
   return transpose(real.(expect(H, reg|>circuit))) + get_logp(sampler, samples)/β
end

# E(x) = <x|U'HU|x>
# d([E(x)]{x∼p} - TS)/dθ = [d(E(x))/dθ]{x∼p}
function grad_θ(H, circuit::AbstractBlock{N}, samples) where N
    reg = get_input_reg(N, samples)
    _, paramsδ = expect'(H, reg=>circuit)
    return paramsδ /size(samples)[end] # divide by batch size since there is a mean in loss
end

function loss_reinforce(β, H, sampler, circuit::AbstractBlock{N}, samples) where N
    f = free_energy_local(β, H, sampler, circuit, samples)
    logp = get_logp(sampler, samples)
    b = mean(f)
    return mean(logp .* (f .- b))
end

function grad_sampler(β, H, sampler, circuit::AbstractBlock{N}, samples) where N
    g = gradient(loss_reinforce, β, H, sampler, circuit, samples)[3]
    unpack_gradient(sampler, g)
end
