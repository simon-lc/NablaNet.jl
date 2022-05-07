@testset "Net" begin
    function local_evaluation(net, xi, θ)
        NablaNet.evaluation!(net, xi, θ)
        return copy(NablaNet.get_output(net))
    end

    ni = 3
    no = 4
    xi0 = rand(ni)
    net0 = NablaNet.Net(ni, no, dim_layers=[6,2,3], activations=[x->x, x->x, x->x, x->x])
    θ0 = rand(NablaNet.parameter_dimension(net0))

    NablaNet.evaluation!(net0, xi0, θ0)
    allocation = @ballocated $NablaNet.evaluation!($net0, $xi0, $θ0)
    @test allocation == 0

    NablaNet.jacobian_input!(net0, xi0, θ0)
    allocation = @ballocated $NablaNet.jacobian_input!($net0, $xi0, $θ0)
    @test allocation == 0
    J0 = FiniteDiff.finite_difference_jacobian(xi0 -> local_evaluation(net0, xi0, θ0), xi0)
    @test norm(net0.jacobian_input - J0, Inf) < 1e-5


    NablaNet.jacobian_parameters!(net0, xi0, θ0)
    allocation = @ballocated $NablaNet.jacobian_parameters!($net0, $xi0, $θ0)
    @test allocation == 0
    J0 = FiniteDiff.finite_difference_jacobian(θ0 -> local_evaluation(net0, xi0, θ0), θ0)
    @test norm(net0.jacobian_parameters - J0, Inf) < 1e-5
end
