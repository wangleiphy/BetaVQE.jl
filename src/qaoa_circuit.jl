export qaoa_circuit
using Yao
using Yao.EasyBuild

get_layout_pairs(nbit::Int; periodic::Bool) = map(i->(i=>i%nbit+1), 1:(periodic ? nbit : nbit-1))
get_layout_pairs(nx::Int, ny::Int; periodic::Bool) = pair_square(nx, ny; periodic=periodic)

@const_gate ZZ = mat(kron(Z, Z))

rzz_layer(nbit::Int, layout) = chain(nbit, [put(nbit, (i,j)=>rot(ZZ, 0.0)) for (i,j) in layout])
rx_layer(nbit::Int) = chain(nbit, [put(nbit, i=>Rx(0.0)) for i=1:nbit])

function qaoa_circuit(size, nlayer::Int; periodic=false)
    nbit = prod(size)
    layout = get_layout_pairs(size...; periodic=periodic)
    c = chain(nbit)
    for i=1:nlayer
        push!(c, rx_layer(nbit))
        push!(c, rzz_layer(nbit, layout))
    end
    return c
end
