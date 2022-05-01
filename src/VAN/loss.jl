function get_energy(K::Matrix{T}, samples) where T <: Real
    energy = sum(samples .* (K*samples), dims=1)
end

function free_energy(K::Matrix{T}, model::AbstractSampler, samples) where T <: Real
    return mean(get_energy(K, samples) .+ get_logp(model, samples))
end

function loss(K::Matrix{T}, model::AbstractSampler, nbatch::Int) where T <: Real
    samples = gen_samples(model, nbatch)
    free_energy(K, model, samples)
end

function loss_reinforce(K::Matrix{T}, model::AbstractSampler, samples) where T <: Real
    e = get_energy(K, samples)
    logp = get_logp(model, samples)
    f = e .+ logp
    b = mean(f)
    return mean(logp.* (f .- b))
end

function grad_model(K::Matrix{T}, model::AbstractSampler, samples) where T <: Real
    return Zygote.gradient(loss_reinforce, K, model, samples)[2]
end

function train(K::Matrix{T}, model::AbstractSampler; optimizer=ADAM(0.1), nbatch::Int=100, niter::Int=100) where T <: Real
    θ = model_parameters(model)
    for i = 1:niter
        _, gθ, _ = gradient(loss, K, model, nbatch)
        update!.(Ref(optimizer), θ, gθ)
        model_dispatch!(model, θ)
        println("$i, Free Energy = ", loss(K, model, nbatch))
    end
end
