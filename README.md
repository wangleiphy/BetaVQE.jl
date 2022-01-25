

<div align="center">
<img align="middle" src="_assets/logo.png" width="300" alt="logo"/>
</div>

## Solving Quantum Statistical Mechanics with Variational Autoregressive Networks and Quantum Circuits 

[![Build Status](https://travis-ci.com/wangleiphy/BetaVQE.jl.svg?branch=master)](https://travis-ci.com/wangleiphy/BetaVQE.jl)

### Setup
Typing `]` in a Julia REPL, and then
```julia
pkg> dev https://github.com/wangleiphy/BetaVQE.jl.git
```

To make sure BetaVQE is installed properly, type
```bash
pkg> test BetaVQE
```

### Run

Run this to train the transverse field Ising model, open a terminal and type
```bash 
$ cd ~/.julia/dev/BetaVQE
$ julia --project runner.jl learn 2 2 2.0 2.0
```

For windows user, the Julia develop folder might be different.

This utility accepts the following arguments

* nx::Int=2, lattice size in x direction,
* ny::Int=2, lattice size in y direction,
* Γ::Real=1.0, the strength of transverse field,
* β::Real=1.0, inverse temperature,

and keyword arguments

* depth::Int=5, circuit depth,
* nsamples::Int=1000, the batch size used in training,
* nhiddens::Vector{Int}=[500], dimension of the VAN's hidden layer,
* lr::Real=0.01, the learning rate of the ADAM optimizer,
* niter::Int=500, number of iteration,
* cont::Bool=false, continue from checkpoint if true.

### Paper
[arXiv:1912.11381](https://arxiv.org/abs/1912.11381)

