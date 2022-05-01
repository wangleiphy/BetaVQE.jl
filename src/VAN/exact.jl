function bitarray(v::Vector{T}, num_bit::Int)::BitArray{2} where T<:Number
    #ba = BitArray{2}(0, 0)
    ba = BitArray(undef, 0, 0)
    ba.len = 64*length(v)
    ba.chunks = UInt64.(v)
    ba.dims = (64, length(v))
    view(ba, 1:num_bit, :)
end

function exact_free_energy(K)
    nbits = size(K, 1)
    configs = bitarray(collect(0:(1<<nbits-1)), nbits)
    energy = sum(configs .* (K*configs), dims=1)
    -log( sum( exp.(-energy)))
end
