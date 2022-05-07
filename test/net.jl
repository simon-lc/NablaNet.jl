@testset "Net" begin
    function local_evaluation(net, xi, θ)
        DiffNet.evaluation!(net, xi, θ)
        return copy(DiffNet.get_output(net))
    end

    ni = 3
    no = 4
    xi0 = rand(ni)
    net0 = DiffNet.Net(ni, no, dim_layers=[6,2,3], activations=[x->x, x->x, x->x, x->x])
    θ0 = rand(DiffNet.parameter_dimension(net0))

    DiffNet.evaluation!(net0, xi0, θ0)
    allocation = @ballocated $DiffNet.evaluation!($net0, $xi0, $θ0)
    @test allocation == 0

    DiffNet.jacobian_input!(net0, xi0, θ0)
    allocation = @ballocated $DiffNet.jacobian_input!($net0, $xi0, $θ0)
    @test allocation == 0
    J0 = FiniteDiff.finite_difference_jacobian(xi0 -> local_evaluation(net0, xi0, θ0), xi0)
    @test norm(net0.jacobian_input - J0, Inf) < 1e-5


    DiffNet.jacobian_parameters!(net0, xi0, θ0)
    allocation = @ballocated $DiffNet.jacobian_parameters!($net0, $xi0, $θ0)
    @test allocation == 0
    J0 = FiniteDiff.finite_difference_jacobian(θ0 -> local_evaluation(net0, xi0, θ0), θ0)
    @test norm(net0.jacobian_parameters - J0, Inf) < 1e-5
end
