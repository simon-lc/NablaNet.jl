module DiffNet

using LinearAlgebra
using Random
using Symbolics


include("layer.jl")
include("net.jl")

export
    Layer,
    Net,
    evaluation!,
    jacobian_input!,
    jacobian_parameters!,
    get_output

end # module
