# DiffNet.jl

[![CI](https://github.com/simon-lc/DiffNet.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/simon-lc/DiffNet.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/simon-lc/DiffNet.jl/branch/main/graph/badge.svg?token=CHJNI2LRNZ)](https://codecov.io/gh/simon-lc/DiffNet.jl)

With DiffNet.jl you can quickly build fully connected neural networks. 
DiffNet.jl provides methods for fast and allocation-free evaluation of the neural network for a given input and given parameters. 
It also provides methods for fast and allocation-free computation of the Jacobians of the output wih respect to the input and the parameters. 

```
# create a fully connected neural network with nonlinear activations

## Dimensions
# input size = 3
ni = 3
# final output size = 4
no = 4

## Neural Network
# generate a random input and some random parameters
xi = rand(ni)
# create neural network with 4 fully connected layers of sizes 6, 2, 3, no
# the activation functions are tanh, tanh, tanh, and identity for the last layer
net = Net(ni, no, dim_layers=[6,2,3], activations=[x->tanh.(x), x->tanh.(x), x->tanh.(x), x->x])
# generate a random input and some random parameters
θ = rand(DiffNet.parameter_dimension(net))

## Computations and Gradients
# evaluate the output of the neural network for given input and parameters
evaluation!(net, xi, θ)
get_output(net)

# Jacobian of the output with respect ot the input
jacobian_input!(net, xi, θ)
net.jacobian_input

# Jacobian of the output with respect ot the input
jacobian_parameters!(net, xi, θ)
net.jacobian_parameters

```


See [this example](../main/examples/net.jl) for more details.
