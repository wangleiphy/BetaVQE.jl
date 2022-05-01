struct AutoRegressiveModel{N, T} <: AbstractSampler
    nbits::Int
    masks::Vector{BitMatrix}
    W::NTuple{N, Matrix{T}}
    b::NTuple{N, Vector{T}}
end

function AutoRegressiveModel(nbits::Int, nhiddens)
    masks = createmasks(collect(1:nbits), nhiddens)

    nhiddens = [nbits, nhiddens..., nbits]
    W = ntuple(i -> glorot_uniform(nhiddens[i+1], nhiddens[i]), length(nhiddens)-1)
    b = ntuple(i -> zeros(nhiddens[i+1]), length(nhiddens)-1)

    AutoRegressiveModel(nbits, masks, W, b)
end

#follows https://github.com/karpathy/pytorch-made/blob/master/made.py#L68
function createmasks(order::Vector{Int}, hs::Vector{Int})
    D = length(order)
    L = length(hs)
    ## assign each unit in the hidden layer an integer
    hs_index = Dict()
    hs_index[0] = collect(1:D)
    for l in 1:L
        hs_index[l] = rand(minimum(hs_index[l-1]):D-1, hs[l])
    end
    ## construct the mask matrices
    masks = [hs_index[i] .>= hs_index[i-1]' for i in 1:L]
    push!(masks, hs_index[0] .> hs_index[L]')
    masks
end

function network(model::AutoRegressiveModel{N}, x::AbstractArray) where N
    for n in 1:N
        x = model.b[n] .+ (model.W[n] .* model.masks[n]) * x
        if n < N
            x = relu.(x)
        else
            x = sigmoid.(x)
        end
    end
    x
end

function get_logp(model::AutoRegressiveModel, x::AbstractArray)
    xhat = network(model, x)
    ϵ = 1E-7
    sum(x.*log.(xhat.+ϵ) + (1 .-x).*log.(1.0 .- xhat .+ ϵ), dims=1)
end

function gen_samples(model::AutoRegressiveModel, nbatch::Int)
    x = zeros(Int, model.nbits, nbatch)
    for i = 1:model.nbits
        xhat = network(model, x)
        x[i, :] .= xhat[i,:] .> rand(nbatch)
    end
    x
end

model_parameters(model::AutoRegressiveModel) = (model.W..., model.b...)

function model_dispatch!(model::AutoRegressiveModel{N}, θ) where N
    for n in 1:N
        model.W[n] .= θ[n]
        model.b[n] .= θ[N+n]
    end
end
