@testset "Layer" begin
    ni = 3
    no = 6
    nθ = no * (ni + 1)
    activation0(x) = tanh.(x)

    xi0 = rand(ni)
    θ0 = rand(nθ)

    layer0 = NablaNet.Layer(ni, no, activation0)
    layer0.xi = xi0
    layer0.θ = θ0

    xo0 = NablaNet.evaluation(ni, no, xi0, θ0, activation0)
    evaluation!(layer0)
    allocation = @ballocated $NablaNet.evaluation!($layer0)
    @test allocation == 0
    @test norm(xo0 - layer0.xo, Inf) < 1e-5

    J0 = FiniteDiff.finite_difference_jacobian(xi0 -> NablaNet.evaluation(ni, no, xi0, θ0, activation0), xi0)
    NablaNet.jacobian_input!(layer0)
    allocation = @ballocated $NablaNet.jacobian_input!($layer0)
    @test allocation == 0
    @test norm(J0 - layer0.jacobian_input, Inf) < 1e-5

    J0 = FiniteDiff.finite_difference_jacobian(θ0 -> NablaNet.evaluation(ni, no, xi0, θ0, activation0), θ0)
    NablaNet.jacobian_parameters!(layer0)
    allocation = @ballocated $NablaNet.jacobian_parameters!($layer0)
    @test allocation == 0
    @test norm(J0 - layer0.jacobian_parameters, Inf) < 1e-5
end
