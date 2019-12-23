export TFIM, hamiltonian
abstract type AbstractModel{D} end

nspin(model::AbstractModel) = prod(model.size)

struct TFIM{D} <:AbstractModel{D}
    size::NTuple{D, Int}
    Γ::Real
    periodic::Bool
    TFIM(size::Int...; Γ::Real, periodic::Bool) = new{length(size)}(size, Γ, periodic)
end

function get_bonds(model::AbstractModel{1})
    nbit, = model.size
    [(i, i%nbit+1) for i in 1:(model.periodic ? nbit : nbit-1)]
end

function get_bonds(model::AbstractModel{2})
    m, n = model.size
    cis = LinearIndices(model.size)
    bonds = Tuple{Int, Int}[]
    for i=1:m, j=1:n
        (i!=m || model.periodic) && push!(bonds, (cis[i,j], cis[i%m+1,j]))
        (j!=n || model.periodic) && push!(bonds, (cis[i,j], cis[i,j%n+1]))
    end
    bonds
end

function hamiltonian(model::TFIM)
    nbit = nspin(model)
    -sum([repeat(nbit, Z, (i,j)) for (i,j) in get_bonds(model)]) +
    -sum([put(nbit, i=>X) for i=1:nbit])*model.Γ
end

#https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.3339
struct SY <:AbstractModel{0}
    size::Int
    J::Real
end

function hamiltonian(model::SY)
    nbit = nspin(model)
    h = Add{nbit}()
    coupling = randn(nbit*(nbit-1)÷2)*model.J/sqrt(2*nbit)
    t = 1
    for i in 1:nbit
        for j in i+1:nbit
            for σ in (X, Y, Z)
                push!(h, repeat(nbit, σ, (i,j)) * coupling[t] )
            end
            t += 1
        end
    end
    h
end
