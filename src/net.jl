mutable struct Net{N,NI,Nθ,NO,T}
    dim_layers::Vector{Int}
    dim_parameters::Vector{Int}
    layers::Vector{<:Layer{T}}
    θ::Vector{T}
    jacobian_input_stages::Vector{Matrix{T}}
    jacobian_parameters_stages::Vector{Matrix{T}}
    jacobian_input::Matrix{T}
    jacobian_parameters::Matrix{T}
end

function Net(NI, NO; dim_layers=zeros(Int,0), activations=[x->x for i=1:length(dim_layers)+1], T=Float64)
    @assert length(dim_layers) == length(activations) - 1
    N = 1 + length(dim_layers)
    dim_layers = [dim_layers; NO]

    layers = Vector()
    jacobian_input_stages = Vector{Matrix{T}}()
    jacobian_parameters_stages = Vector{Matrix{T}}()
    ni = NI
    for i = 1:N
        no = dim_layers[i]
        layer = Layer(ni, no, activations[i])
        push!(layers, layer)
        push!(jacobian_input_stages, zeros(NO, ni))
        push!(jacobian_parameters_stages, zeros(NO, layer.nθ))
        ni = no
    end
    layers = [layers...]
    dim_parameters = [layer.nθ for layer in layers]
    Nθ = sum(dim_parameters)

    θ = zeros(Nθ)
    jacobian_input = zeros(NO, NI)
    jacobian_parameters = zeros(NO, Nθ)

    T = eltype(layers[1].xi)
    return Net{N,NI,Nθ,NO,T}(dim_layers, dim_parameters, layers,
        θ,
        jacobian_input_stages, jacobian_parameters_stages, jacobian_input, jacobian_parameters)
end

parameter_dimension(net::Net{N,NI,Nθ}) where {N,NI,Nθ} = Nθ
get_input(net::Net) = net.layers[1].xi
get_output(net::Net{N}) where N = net.layers[N].xo

function set_input!(net::Net, xi::Vector{T}) where T
    net.layers[1].xi = xi
    return nothing
end

function set_parameters!(net::Net{N}, θ::Vector{T}) where {N,T}
    net.θ = θ
    layers = net.layers
    off = 0
    for i = 1:N
        nθ = layers[i].nθ
        for j = 1:nθ
            layers[i].θ[j] = θ[off + j]
        end
        off += nθ
    end
    return nothing
end

function evaluation!(net::Net, xi::Vector{T}, θ::Vector{T}) where T
    set_input!(net, xi)
    set_parameters!(net, θ)
    evaluation!(net)
    return nothing
end

function evaluation!(net::Net{N}) where N
    for i = 1:N
        evaluation!(net.layers[i])
        (i < N) && (net.layers[i+1].xi = net.layers[i].xo)
    end
    return nothing
end

function jacobian_input!(net::Net{N}) where N
    # evaluation
    for i = 1:N
        layer = net.layers[i]
        evaluation!(layer)
        jacobian_input!(layer)
        (i < N) && (net.layers[i+1].xi = layer.xo)
    end

    # jacobian
    layer = net.layers[N]
    net.jacobian_input_stages[N] = layer.jacobian_input
    for i = N-1:-1:1
        layer = net.layers[i]
        mul!(net.jacobian_input_stages[i], net.jacobian_input_stages[i+1], layer.jacobian_input)
    end
    net.jacobian_input = net.jacobian_input_stages[1]
    return nothing
end

function jacobian_input!(net::Net, xi::Vector{T}, θ::Vector{T}) where T
    set_input!(net, xi)
    set_parameters!(net, θ)
    jacobian_input!(net)
    return nothing
end

function jacobian_parameters!(net::Net{N}) where N
    # evaluation & jacobian input
    jacobian_input!(net)
    # jacobian parameters
    for i = 1:N
        layer = net.layers[i]
        jacobian_parameters!(layer)
    end
    for i = 1:N-1
        layer = net.layers[i]
        mul!(net.jacobian_parameters_stages[i], net.jacobian_input_stages[i+1], layer.jacobian_parameters)
    end
    net.jacobian_parameters_stages[N] = net.layers[N].jacobian_parameters

    off = 0
    for i = 1:N
        nθ = net.layers[i].nθ
        net.jacobian_parameters[:, off .+ (1:nθ)] = net.jacobian_parameters_stages[i]
        off += nθ
    end
    return nothing
end

function jacobian_parameters!(net::Net, xi::Vector{T}, θ::Vector{T}) where T
    set_input!(net, xi)
    set_parameters!(net, θ)
    jacobian_parameters!(net)
    return nothing
end
