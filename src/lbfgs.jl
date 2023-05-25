# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools

import SolverCore.solve!
import Krylov.solve!

function get_status(
    nlp;
    elapsed_time = 0.0,
    optimal = false,
    unbounded = false,
    max_eval = Inf,
    max_time = Inf,
)
    if optimal
        :first_order
    elseif unbounded
        :unbounded
    elseif neval_obj(nlp) > max_eval ≥ 0
        :max_eval
    elseif elapsed_time > max_time
        :max_time
    else
        :unknown
    end
end

mutable struct LBFGSSolver{T,V,Op<:AbstractLinearOperator{T},M<:AbstractNLPModel{T,V}} <:
               AbstractOptimizationSolver
    x::V
    xt::V
    gx::V
    gt::V
    d::V
    H::Op
    h::LineModel{T,V,M}
end

function LBFGSSolver(nlp::M; mem::Int = 5) where {T,V,M<:AbstractNLPModel{T,V}}
    nvar = nlp.meta.nvar
    x = V(undef, nvar)
    d = V(undef, nvar)
    xt = V(undef, nvar)
    gx = V(undef, nvar)
    gt = V(undef, nvar)
    H = InverseLBFGSOperator(T, nvar, mem = mem, scaling = true)
    h = LineModel(nlp, x, d)
    Op = typeof(H)
    return LBFGSSolver{T,V,Op,M}(x, xt, gx, gt, d, H, h)
end

function SolverCore.reset!(solver::LBFGSSolver)
    reset!(solver.H)
end

function SolverCore.reset!(solver::LBFGSSolver, nlp::AbstractNLPModel)
    reset!(solver.H)
    solver.h = LineModel(nlp, solver.x, solver.d)
    solver
end

function SolverCore.solve!(
    solver::AbstractOptimizationSolver,
    param::AbstractParameterSet,
    model::AbstractNLPModel;
    kwargs...,
)
    stats = GenericExecutionStats(model)
    solve!(solver, param, model, stats; kwargs...)
end

function lbfgs(nlp::AbstractNLPModel; mem::Int = 5, kwargs...)
    param = LBFGSParameterSet(mem = mem)
    return lbfgs(nlp, param; kwargs...)
end

function lbfgs(
    nlp::AbstractNLPModel,
    param::LBFGSParameterSet;
    x::V = nlp.meta.x0,
    kwargs...,
) where {V}
    mem = value(param.mem)
    solver = LBFGSSolver(nlp; mem = mem)
    return solve!(solver, param, nlp; x = x, kwargs...)
end

function SolverCore.solve!(
    solver::LBFGSSolver{T,V},
    param::LBFGSParameterSet,
    nlp::AbstractNLPModel{T,V},
    stats::GenericExecutionStats{T,V};
    callback = (args...) -> nothing,
    x::V = nlp.meta.x0,
    atol::T = √eps(T),
    rtol::T = √eps(T),
    max_eval::Int = -1,
    max_time::Float64 = 30.0,
    τ₁::T = T(0.9999), # parameter
    bk_max::Int = 25, # parameter
    verbose::Int = 0,
    verbose_subsolver::Int = 0,
) where {T,V}
    if !(nlp.meta.minimize)
        error("lbfgs only works for minimization problem")
    end
    if !unconstrained(nlp)
        error("lbfgs should only be called for unconstrained problems. Try tron instead")
    end

    reset!(stats)
    start_time = time()
    set_time!(stats, 0.0)

    n = nlp.meta.nvar

    solver.x .= x
    x = solver.x
    xt = solver.xt
    ∇f = solver.gx
    ∇ft = solver.gt
    d = solver.d
    h = solver.h
    H = solver.H
    reset!(H)

    f, ∇f = objgrad!(nlp, x, ∇f)

    ∇fNorm = nrm2(n, ∇f)
    ϵ = atol + rtol * ∇fNorm

    set_iter!(stats, 0)
    set_objective!(stats, f)
    set_dual_residual!(stats, ∇fNorm)

    verbose > 0 && @info log_header(
        [:iter, :f, :dual, :slope, :bk],
        [Int, T, T, T, Int],
        hdr_override = Dict(:f => "f(x)", :dual => "‖∇f‖", :slope => "∇fᵀd"),
    )

    optimal = ∇fNorm ≤ ϵ

    set_status!(
        stats,
        get_status(
            nlp,
            elapsed_time = stats.elapsed_time,
            optimal = optimal,
            max_eval = max_eval,
            max_time = max_time,
        ),
    )

    callback(nlp, solver, stats)

    done =
        (stats.status == :first_order) ||
        (stats.status == :max_eval) ||
        (stats.status == :max_time) ||
        (stats.status == :user) ||
        (stats.status == :not_desc)

    while !done
        mul!(d, H, ∇f, -one(T), zero(T))
        slope = dot(n, d, ∇f)
        if slope ≥ 0
            @error "not a descent direction" slope
            set_status!(stats, :not_desc)
            done = true
            continue
        end

        # Perform improved Armijo linesearch.
        t, good_grad, ft, nbk, nbW = armijo_wolfe(
            h,
            f,
            slope,
            ∇ft,
            τ₁ = τ₁,
            bk_max = bk_max,
            verbose = Bool(verbose_subsolver),
        )

        verbose > 0 &&
            mod(stats.iter, verbose) == 0 &&
            @info log_row(Any[stats.iter, f, ∇fNorm, slope, nbk])

        copyaxpy!(n, t, d, x, xt)
        good_grad || grad!(nlp, xt, ∇ft)

        # Update L-BFGS approximation.
        d .*= t
        @. ∇f = ∇ft - ∇f
        push!(H, d, ∇f)

        # Move on.
        x .= xt
        f = ft
        ∇f .= ∇ft

        ∇fNorm = nrm2(n, ∇f)

        set_objective!(stats, f)
        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)
        set_dual_residual!(stats, ∇fNorm)
        optimal = ∇fNorm ≤ ϵ

        set_status!(
            stats,
            get_status(
                nlp,
                elapsed_time = stats.elapsed_time,
                optimal = optimal,
                max_eval = max_eval,
                max_time = max_time,
            ),
        )

        callback(nlp, solver, stats)

        done =
            (stats.status == :first_order) ||
            (stats.status == :max_eval) ||
            (stats.status == :max_time) ||
            (stats.status == :user) ||
            (stats.status == :not_desc)
    end
    verbose > 0 && @info log_row(Any[stats.iter, f, ∇fNorm])

    set_solution!(stats, x)
    stats
end
