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
    return arrayreg(transpose(config); nbatch=nbatch)
end

function free_energy(β::Real, H::AbstractBlock, sampler::AbstractSampler, circuit::AbstractBlock, samples)
    mean(free_energy_local(β, H, sampler, circuit, samples))
end

function energy(H::AbstractBlock, sampler, circuit::AbstractBlock, samples)
    reg = get_input_reg(nqubits(H), samples)
    mean(real.(expect(H, reg|>circuit)))
end

function energy2(H::AbstractBlock, sampler::AbstractSampler, circuit::AbstractBlock, samples)
    reg = get_input_reg(nqubits(H), samples)
    mean(real.(expect(H*H, reg|>circuit)))
end

function entropy(H::AbstractBlock, sampler::AbstractSampler, circuit::AbstractBlock, samples)
    -mean(get_logp(sampler, samples))
end

function purity(H::AbstractBlock, sampler::AbstractSampler, circuit::AbstractBlock, samples)
    mean(exp.(get_logp(sampler, samples)))
end

function spectra(H::AbstractBlock, sampler::AbstractSampler, circuit::AbstractBlock, samples)
    reg = get_input_reg(nqubits(H), samples)
    real.(expect(H, reg|>circuit))
end

function free_energy_local(β::Real, H::AbstractBlock, sampler, circuit::AbstractBlock, samples)
   reg = get_input_reg(nqubits(H), samples)
   return transpose(real.(expect(H, reg|>circuit))) + get_logp(sampler, samples)/β
end

# E(x) = <x|U'HU|x>
# d([E(x)]{x∼p} - TS)/dθ = [d(E(x))/dθ]{x∼p}
function grad_θ(H, circuit::AbstractBlock, samples)
    reg = get_input_reg(nqubits(H), samples)
    _, paramsδ = expect'(H, reg=>circuit)
    return paramsδ /size(samples)[end] # divide by batch size since there is a mean in loss
end

function loss_reinforce(β, H, sampler, circuit::AbstractBlock, samples)
    f = free_energy_local(β, H, sampler, circuit, samples)
    logp = get_logp(sampler, samples)
    b = mean(f)
    return mean(logp .* (f .- b))
end

function grad_sampler(β, H, sampler, circuit::AbstractBlock, samples)
    return Zygote.gradient(loss_reinforce, β, H, sampler, circuit, samples)[3]
end
