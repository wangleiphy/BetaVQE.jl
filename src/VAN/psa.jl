struct PSAModel{T} <: AbstractSampler
    nbits::Int
    w::Vector{T}
end

function PSAModel(nbits::Int)
    w = rand(nbits) .- 0.5
    PSAModel(nbits, w)
end

function get_logp(model::PSAModel, x::AbstractArray)
    xhat = sigmoid.(model.w)
    ϵ = 1E-7
  sum(x.*log.(xhat.+ϵ) + (1 .-x).*log.(1.0 .- xhat .+ ϵ), dims=1)
end

function gen_samples(model::PSAModel, nbatch::Int)
    x = zeros(Int, model.nbits, nbatch)
    for i = 1:model.nbits
        xhat = sigmoid.(model.w)
        x[i, :] .= xhat[i,:] .> rand(nbatch)
    end
    x
end

model_parameters(model::PSAModel) = (model.w, )

function model_dispatch!(model::PSAModel, θ)
    model.w .= θ[1]
end
