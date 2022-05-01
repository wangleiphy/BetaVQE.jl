function rrule(::typeof(free_energy), K::Matrix{T}, model::AbstractSampler, samples) where T <: Real
    free_energy(K, model, samples), function (adjy)
        nt = map(x->x === nothing ? NoTangent() : x .* adjy, grad_model(K, model, samples))
        adjmodel = Tangent{typeof(model)}(; nt...)
        return (NoTangent(), NoTangent(), adjmodel, NoTangent())
    end
end

@non_differentiable gen_samples(model, nbatch)
@non_differentiable createmasks(order, hs)
@non_differentiable AutoRegressiveModel(nbits, nhidden)
@non_differentiable PSAModel(nbits)
