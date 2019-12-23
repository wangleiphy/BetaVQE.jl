using Zygote
using Zygote: @adjoint, @nograd

@adjoint function free_energy(β::Real, H::AbstractBlock{N}, sampler, circuit::AbstractBlock{N}, samples) where N
    free_energy(β, H, sampler, circuit, samples), function (adjy)
        adjθ = grad_θ(H, circuit, samples) * adjy
        adjsampler = grad_sampler(β, H, sampler, circuit, samples) .*adjy
        return (nothing, nothing, adjsampler, adjθ, nothing)
    end
end

@nograd update!, dispatch!, get_input_reg, free_energy_local
