function ChainRulesCore.rrule(::typeof(free_energy), β::Real, H::AbstractBlock, sampler, circuit::AbstractBlock, samples)
    free_energy(β, H, sampler, circuit, samples), function (adjy)
        adjθ = grad_θ(H, circuit, samples) * adjy
        nt = map(x->x === nothing ? NoTangent() : x .* adjy, grad_sampler(β, H, sampler, circuit, samples))
        adjsampler = Tangent{typeof(sampler)}(; nt...)
        return (NoTangent(), NoTangent(), NoTangent(), adjsampler, adjθ, NoTangent())
    end
end

@non_differentiable Optimisers.update!(x, x̄)
@non_differentiable dispatch!(x, collection)
@non_differentiable get_input_reg(nbits, samples)
@non_differentiable free_energy_local(β, H, sampler, circuit, samples)
