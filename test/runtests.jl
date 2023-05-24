using ADNLPModels, TunerTestSolver, Test

@testset "Test TunerTestSolver" begin
    f(x) = sum(x)
    x = ones(3)
    nlp = ADNLPModel(f, x)
    TunerTestSolver.lbfgs(nlp)
end
