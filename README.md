

<div align="center">
<img align="middle" src="_assets/logo.png" width="500" alt="logo"/>
</div>

# Solving Quantum Statistical Mechanics with Variational Autoregressive Networks and Quantum Circuits 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GiggleLiu.github.io/ThermalVQE.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GiggleLiu.github.io/ThermalVQE.jl/dev)
[![Build Status](https://travis-ci.com/GiggleLiu/ThermalVQE.jl.svg?branch=master)](https://travis-ci.com/GiggleLiu/ThermalVQE.jl)
[![Codecov](https://codecov.io/gh/GiggleLiu/ThermalVQE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/ThermalVQE.jl)

## Setup
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

## Run

Run this to train the model
```bash 
julia runner.jl learn 2 2 2.0 2.0
```