#some useful functions
glorot_uniform(dims...) = (rand(Float64, dims...) .- 0.5) .* sqrt(24.0/sum(dims))

relu(x::Real) = max(zero(x), x)

sigmoid(x::Real) = one(x) / (one(x) + exp(-x))

softplus(x::Real) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

