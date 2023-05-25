using BBModels
using ADNLPModels, NLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems
using TunerTestSolver
using BenchmarkTools

# select problems
# 1st basic strategy: select 5 small unconstrained problems
n = 5
meta = OptimizationProblems.meta
list =
    meta[meta.minimize.&(meta.ncon.==0).&.!meta.has_bounds.&(20 .≤ meta.nvar .≤ 100), :name]
problems = [eval(p)() for p ∈ Symbol.(list[1:n])]

param_set = TunerTestSolver.LBFGSParameterSet()

function fun(vec_metrics::Vector{ProblemMetrics})
    penalty = 1e2
    global fx = 0
    for p in vec_metrics
        failed = is_failure(BBModels.get_status(p))
        fx += failed * penalty
        if !failed
            id = get_pb_id(p)
            nlp = problems[id]
            nvar = nlp isa AbstractNLPModel ? nlp.meta.nvar : nlp().meta.nvar # better way to access number of variables?
            nobj = get_counters(p).neval_obj
            ngrad = get_counters(p).neval_grad * nvar
            nhprod = get_counters(p).neval_hprod * nvar
            nhess = get_counters(p).neval_hess * nvar^2
            fx += nobj + ngrad + nhprod + nhess
        end
    end
    return fx
end

model = BBModel(
    param_set, # AbstractParameterSet
    problems, # vector of AbstractNLPModel
    TunerTestSolver.lbfgs, # (::AbstractNLPModel, ::AbstractParameterSet) -> GenericExecutionStats
    fun, # time_only, memory_only, sumfc OR a hand-made function
)

vals = BBModels.random_search(model, verbose = 0)

open("parameters.jl", "w") do io
    for (name, val) in zip(names(param_set), vals)
        println(io, "$name = $val")
    end
end
