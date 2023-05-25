module TunerTestSolver

using SolverParameters

include("parameters.jl")

struct LBFGSParameterSet{T<:Real} <: AbstractParameterSet
    mem::Parameter{Int,IntegerRange{Int}}
    τ₁::Parameter{T,RealInterval{T}}
    bk_max::Parameter{Int,IntegerRange{Int}}
    # add scaling

    function LBFGSParameterSet{T}(;
        mem::Int = mem,
        τ₁::T = T(τ₁),
        bk_max::Int = bk_max,
    ) where {T}
        p_set = new(
            Parameter(mem, IntegerRange(Int(1), Int(20)), "mem"),
            Parameter(τ₁, RealInterval(T(0), T(1)), "τ₁"),
            Parameter(bk_max, IntegerRange(Int(10), Int(50)), "bk_max"),
        )
        return p_set
    end

    function LBFGSParameterSet(; kwargs...)
        return LBFGSParameterSet{Float64}(; kwargs...)
    end
end

include("lbfgs.jl")

end # end of module
