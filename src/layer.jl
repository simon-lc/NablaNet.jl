mutable struct Layer{T,F<:Function,L,NL,JLX,JNL}
    ni::Int
    no::Int
    nθ::Int
    activation::F
    xi::Vector{T}
    xm::Vector{T}
    xo::Vector{T}
    θ::Vector{T}
    ∂xm∂xi::Matrix{T}
    ∂xm∂θ::Matrix{T}
    ∂xo∂xm::Matrix{T}
    jacobian_input::Matrix{T}
    jacobian_parameters::Matrix{T}
    linearity!::L
    nonlinearity!::NL
    jacobian_input!::JLX
    jacobian_nonlinearity!::JNL
    # jacobian_parameters!::JLθ
end

function Layer(ni, no, activation)
    nθ = no * (ni + 1)

    xi_ = Symbolics.variables(:xi_, 1:ni)
    xm_ = Symbolics.variables(:xm_, 1:no)
    θ_ = Symbolics.variables(:θ_, 1:nθ)
    xm = evaluation(ni, no, xi_, θ_)
    xo = activation(xm_)
    ∂xm∂xi = Symbolics.jacobian(xm, xi_)
    ∂xm∂θ = Symbolics.jacobian(xm, θ_)
    ∂xo∂xm = Symbolics.jacobian(xo, xm_)

    linearity! = Symbolics.build_function(xm, xi_, θ_,
        checkbounds=true,
        expression=Val{false})[2]

    nonlinearity! = Symbolics.build_function(xo, xm_,
        checkbounds=true,
        expression=Val{false})[2]

    jacobian_input! = Symbolics.build_function(∂xm∂xi, xi_, θ_,
        checkbounds=true,
        expression=Val{false})[2]

    jacobian_nonlinearity! = Symbolics.build_function(∂xo∂xm, xm_,
        checkbounds=true,
        expression=Val{false})[2]

    # jacobian_parameters! = Symbolics.build_function(∂xm∂θ, xi_, θ_,
    #     checkbounds=true,
    #     expression=Val{false})[2]

    xi = zeros(ni)
    xm = zeros(no)
    xo = zeros(no)
    θ = zeros(nθ)
    ∂xm∂xi = zeros(no, ni)
    ∂xm∂θ = zeros(no, nθ)
    ∂xo∂xm = zeros(no, no)
    jacobian_input = zeros(no, ni)
    jacobian_parameters = zeros(no, nθ)

    return Layer(
        ni, no, nθ,
        activation,
        xi, xm, xo, θ,
        ∂xm∂xi,
        ∂xm∂θ,
        ∂xo∂xm,
        jacobian_input,
        jacobian_parameters,
        linearity!,
        nonlinearity!,
        jacobian_input!,
        jacobian_nonlinearity!,
        # jacobian_parameters!,
        )
end

function evaluation(ni::Int, no::Int, xi::Vector, θ::Vector, activation::Function)
    xm = evaluation(ni, no, xi, θ)
    xo = activation(xm)
    return xo
end

function evaluation(ni::Int, no::Int, xi::Vector, θ::Vector)
    xm = reshape(θ, no, ni+1) * [1; xi]
    return xm
end

function slow_evaluation(xi::Vector, layer::Layer)
    evaluation(layer.ni, layer.no, xi, layer.θ, layer.activation)
end

function evaluation!(layer::Layer)
    layer.linearity!(layer.xm, layer.xi, layer.θ)
    layer.nonlinearity!(layer.xo, layer.xm)
    return nothing
end

function jacobian_input!(layer::Layer)
    # Compute xm
    layer.linearity!(layer.xm, layer.xi, layer.θ)
    # compute gradients
    layer.jacobian_input!(layer.∂xm∂xi, layer.xi, layer.θ)
    layer.jacobian_nonlinearity!(layer.∂xo∂xm, layer.xm)
    # chain rule
    mul!(layer.jacobian_input, layer.∂xo∂xm, layer.∂xm∂xi)
    return nothing
end

function jacobian_parameters!(layer::Layer)
    ni = layer.ni
    no = layer.no
    # Compute xm
    layer.linearity!(layer.xm, layer.xi, layer.θ)
    # compute gradients
    # layer.jacobian_parameters!(layer.∂xm∂θ, layer.xi, layer.θ)
    layer.jacobian_nonlinearity!(layer.∂xo∂xm, layer.xm)
    # chain rule
    # mul!(layer.jacobian_parameters, layer.∂xo∂xm, layer.∂xm∂θ)
    for i = 1:no
        layer.jacobian_parameters[i, i+(1-1)*no] = layer.∂xo∂xm[i,i] * 1.0
        for j = 2:ni+1
            layer.jacobian_parameters[i, i+(j-1)*no] = layer.∂xo∂xm[i,i] * layer.xi[j-1]
        end
    end
    return nothing
end
