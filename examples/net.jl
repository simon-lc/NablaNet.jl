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
