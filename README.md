

<div align="center">
<img align="middle" src="_assets/logo.png" width="300" alt="logo"/>
</div>

## Solving Quantum Statistical Mechanics with Variational Autoregressive Networks and Quantum Circuits 

[![Build Status](https://travis-ci.com/wangleiphy/BetaVQE.jl.svg?branch=master)](https://travis-ci.com/wangleiphy/BetaVQE.jl)
[![Codecov](https://codecov.io/gh/wangleiphy/BetaVQE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/BetaVQE/betaVQEVQE.jl)

### Setup
Clone this repo, add dependancies by typing `]` in a Julia REPL, and then
```julia
pkg> add Yao YaoExtensions
pkg> add StatsBase Zygote Flux JLD2 FileIO Fire
pkg> dev https://github.com/wangleiphy/VAN.jl.git 
pkg> dev .
```

To make sure it works, type
```bash
julia test/runtests.jl
```
in a terminal to run tests.

### Run

Run this to train the model
```bash 
julia runner.jl learn 2 2 2.0 2.0
```
