using Test
using BenchmarkTools

using LinearAlgebra
using Random
using FiniteDiff

using NablaNet

@testset "Layer"   verbose=true begin include("layer.jl") end
@testset "Net"     verbose=true begin include("net.jl") end
