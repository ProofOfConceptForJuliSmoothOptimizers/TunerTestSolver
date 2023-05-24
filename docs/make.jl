using TunerTestSolver
using Documenter

DocMeta.setdocmeta!(TunerTestSolver, :DocTestSetup, :(using TunerTestSolver); recursive = true)

makedocs(;
  modules = [TunerTestSolver],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Tangi Migot and Monssaf Toukal",
  repo = "https://github.com/ProofOfConceptForJuliSmoothOptimizers/TunerTestSolver.jl/blob/{commit}{path}#{line}",
  sitename = "TunerTestSolver.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://ProofOfConceptForJuliSmoothOptimizers.github.io/TunerTestSolver.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/ProofOfConceptForJuliSmoothOptimizers/TunerTestSolver.jl",
  push_preview = true,
  devbranch = "main",
)
