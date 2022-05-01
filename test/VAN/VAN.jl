using Test

@testset "test product model" begin
    include("psa.jl")
end

@testset "test autoregressive model" begin
    include("arm.jl")
end
