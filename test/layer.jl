@testset "Layer" begin
    ni = 3
    no = 6
    nθ = no * (ni + 1)
    activation0(x) = tanh.(x)

    xi0 = rand(ni)
    θ0 = rand(nθ)

    layer0 = DiffNet.Layer(ni, no, activation0)
    layer0.xi = xi0
    layer0.θ = θ0

    xo0 = DiffNet.evaluation(ni, no, xi0, θ0, activation0)
    evaluation!(layer0)
    allocation = @ballocated $DiffNet.evaluation!($layer0)
    @test allocation == 0
    @test norm(xo0 - layer0.xo, Inf) < 1e-5

    J0 = FiniteDiff.finite_difference_jacobian(xi0 -> DiffNet.evaluation(ni, no, xi0, θ0, activation0), xi0)
    DiffNet.jacobian_input!(layer0)
    allocation = @ballocated $DiffNet.jacobian_input!($layer0)
    @test allocation == 0
    @test norm(J0 - layer0.jacobian_input, Inf) < 1e-5

    J0 = FiniteDiff.finite_difference_jacobian(θ0 -> DiffNet.evaluation(ni, no, xi0, θ0, activation0), θ0)
    DiffNet.jacobian_parameters!(layer0)
    allocation = @ballocated $DiffNet.jacobian_parameters!($layer0)
    @test allocation == 0
    @test norm(J0 - layer0.jacobian_parameters, Inf) < 1e-5
end
