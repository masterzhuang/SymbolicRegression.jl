#######################################################################
# SeedPopulation.jl — Seed-injected initial populations for ASOUL-SR
#
# This module lives in the `asoul-qd-v1` fork of
# `MilesCranmer/SymbolicRegression.jl`. Its job is to parse a bundle of
# seed expressions supplied via the `ASOUL_SR_INITIAL_SEEDS_JSON`
# environment variable and splice the parsed trees into the first
# population of each output at the start of `_main_search_loop!`, so a
# neural proposer can seed the search without any PySR kwarg-whitelist
# round-trip (see docs/notes/neurosymbolic_seeded_pysr_design_v1.md).
#
# Rollback invariant: with the env var unset, `seeds_from_env()` returns
# `nothing`, `Options.initial_seed_strings === nothing`, and the
# `_initialize_search!` splice branch is skipped. The hot path is
# byte-identical to upstream SymbolicRegression.jl v1.11.3 plus the
# existing QD env-var bridge.
#
# See the ASOUL-SR design document:
#   docs/notes/neurosymbolic_seeded_pysr_design_v1.md  (project-local)
#######################################################################

module SeedPopulationModule

using ..CoreModule: AbstractOptions, DATA_TYPE, LOSS_TYPE, Dataset
using ..PopMemberModule: PopMember
using ..PopulationModule: Population
using ..LossFunctionsModule: eval_cost
using DynamicExpressions:
    AbstractExpression, AbstractExpressionNode, Node, parse_expression
import JSON3

export SEEDS_ENV_VAR,
       seeds_from_env,
       parse_seed_tree,
       splice_seeds_into_population!,
       seed_initial_population!,
       last_splice_stats,
       reset_splice_stats!

const SEEDS_ENV_VAR = "ASOUL_SR_INITIAL_SEEDS_JSON"

# ---------------------------------------------------------------------
# 0. Splice-stats state channel (Python-readable via juliacall)
# ---------------------------------------------------------------------
#
# Codex branch review P2 (2026-04-18): `seed_initial_population!`
# returned the spliced count but `_initialize_search!` discarded it, so
# the Python driver couldn't tell whether the seeded call actually
# injected anything. A run where every seed failed `parse_expression`
# or `PopMember` construction would fall through to random-init while
# the Python side still reported "seeded".
#
# Fix: a module-level Ref holds the stats of the most recent
# `seed_initial_population!` call. The Python driver queries it via
# juliacall after each `PySRRegressor.fit()` (see
# scripts/80a_neurosymbolic_smoke.py::_probe_seed_splice_stats) and
# records the counts on the dataset's seeded-provenance block. A
# `spliced == 0` reading with `n_seeds > 0` flags the zero-splice
# failure mode explicitly rather than silently passing.
#
# Scope note: the Ref is a single last-call snapshot, not a per-dataset
# accumulator. For the current smoke workflow (sequential per-dataset
# fit), querying right after each fit() captures the correct stats.
# Multi-process :multiprocessing parallelism would each maintain its
# own Ref in its own process; the driver queries only the main process
# for chunk-h so that is fine.

const LAST_SPLICE_STATS = Ref((
    n_seeds      = 0,
    parsed_ok    = 0,
    parse_failed = 0,
    build_failed = 0,
    spliced      = 0,
))

"""
    last_splice_stats() -> NamedTuple

Return the stats of the most recent `seed_initial_population!` call
as a `(n_seeds, parsed_ok, parse_failed, build_failed, spliced)`
NamedTuple. All fields are `0` before any splice runs (default Ref
value) and after `reset_splice_stats!()`.

Exposed so the Python driver can query via juliacall after each
`PySRRegressor.fit()` and record the counts on the per-dataset
seeded-provenance block (Codex branch review P2, 2026-04-18).
"""
last_splice_stats() = LAST_SPLICE_STATS[]

"""
    reset_splice_stats!() -> NamedTuple

Reset the splice-stats Ref to all zeros. Intended for the Python
driver to call before each dataset's `fit()` so the subsequent
`last_splice_stats()` read reflects that dataset only, not a
left-over snapshot from a previous one.

Returns the reset value (all zeros).
"""
function reset_splice_stats!()
    LAST_SPLICE_STATS[] = (
        n_seeds      = 0,
        parsed_ok    = 0,
        parse_failed = 0,
        build_failed = 0,
        spliced      = 0,
    )
    return LAST_SPLICE_STATS[]
end

# ---------------------------------------------------------------------
# 1. Env-variable bridge
# ---------------------------------------------------------------------

"""
    seeds_from_env(; env_var = SEEDS_ENV_VAR) -> Union{Nothing, Vector{String}}

Read the path held in `ENV[env_var]`, load the JSON document at that
path, validate its shape, and return the `seed_expressions` array as a
`Vector{String}`.

Behaviour contract (designed so the upstream test suite cannot observe
any effect unless the env var is deliberately set):

  - Env var unset / empty          -> `nothing`, no warning
  - Path missing on disk           -> `nothing`, one `@warn`
  - JSON parse failure             -> `nothing`, one `@warn`
  - Wrong schema (`version != 1`
    or `seed_expressions` not an
    array)                         -> `nothing`, one `@warn`
  - All entries blank / non-string -> `nothing`, no warning
  - At least one parseable string  -> `Vector{String}` in file order,
                                      with empty / non-string entries
                                      dropped silently

Called lazily from `Options.jl::_seed_strings_from_env` so the rollback
invariant is preserved: scripts that never set `ASOUL_SR_INITIAL_SEEDS_JSON`
incur zero extra work at `Options(...)` construction.

Expected JSON shape:

    {"version": 1, "seed_expressions": ["X3*log(X1/X2)", "X1*log(X2/X3)"]}
"""
function seeds_from_env(
    ; env_var::AbstractString = SEEDS_ENV_VAR,
)::Union{Nothing, Vector{String}}
    raw = get(ENV, env_var, nothing)
    (raw === nothing || isempty(strip(raw))) && return nothing
    path = strip(raw)
    if !isfile(path)
        @warn "SeedPopulation: $env_var points at missing file; ignoring" path
        return nothing
    end
    doc = try
        JSON3.read(read(path, String))
    catch err
        @warn "SeedPopulation: failed to parse JSON; ignoring" path exception=(err, catch_backtrace())
        return nothing
    end
    version = get(doc, :version, nothing)
    exprs   = get(doc, :seed_expressions, nothing)
    if version != 1 || !(exprs isa AbstractVector)
        @warn "SeedPopulation: unexpected JSON shape (want version=1 + seed_expressions array); ignoring" path version
        return nothing
    end
    strings = String[]
    for s in exprs
        (s isa AbstractString) || continue
        trimmed = strip(String(s))
        isempty(trimmed) && continue
        push!(strings, String(trimmed))
    end
    isempty(strings) && return nothing
    return strings
end

# ---------------------------------------------------------------------
# 2. String -> tree parser
# ---------------------------------------------------------------------

"""
    parse_seed_tree(seed_str, options, dataset) -> Union{Nothing, AbstractExpression}

Parse `seed_str` (a Python / sympy-style expression string such as
`"X3 * log(X1 / X2)"`) into an `AbstractExpression` using
`DynamicExpressions.parse_expression`. The operator enum and variable
names come from the fully-constructed `options.operators` and the
dataset respectively, which is why this function runs **inside**
`_initialize_search!` rather than at `Options(...)` body time —
`datasets[j].variable_names` only exists after `equation_search` has
built the dataset bundle.

Returns `nothing` on any parse failure; a single aggregated summary
warning is emitted by `seed_initial_population!` rather than one
per-seed warning, to avoid flooding stderr when a batch of 16 seeds
has a grammar mismatch.

Notes
-----
- `parse_expression` expects an `Expr`, so the string is first run
  through `Meta.parse`. A `Meta.ParseError` from malformed Julia syntax
  surfaces as a caught exception and triggers the skip path.
- `variable_names` is pulled from `dataset.variable_names` when the
  dataset carries names (v1.11.3 `Dataset` sets this from PySR's
  `variable_names` kwarg); otherwise we fall back to `X1, X2, ...`
  which matches the naming the seed JSON is generated against by
  `scripts/80a_neurosymbolic_smoke.py`.
- Operator resolution uses `options.operators` (the fully-built
  `OperatorEnum`) rather than re-deriving it from
  `options.operators.binops / unaops`, so PySR-registered custom ops
  like `inv(x) = 1/x` resolve correctly.
- `expression_type` + `node_type` are forwarded from the `Options`
  so the produced tree matches the type the population / `PopMember`
  expect (critical for users running `ParametricExpression`,
  `TemplateExpression`, etc. — if the seed grammar is incompatible
  with that expression type, `parse_expression` throws and we skip
  the seed).
"""
function parse_seed_tree(
    seed_str::AbstractString,
    options::AbstractOptions,
    dataset::Dataset{T, L},
) where {T, L}
    varnames = _resolve_variable_names(dataset)
    # SymPy → Julia syntax normalization. The Python-side driver writes
    # `sp.sstr(c.canonical_expr)` which uses sympy's StrPrinter and
    # emits `**` for `Pow` nodes (e.g. `X3**(-1)` from `inv(X3)`).
    # `Meta.parse` rejects `**` outright because Julia's power operator
    # is `^`. Without this normalization any seed with a power would be
    # silently dropped and the "seeded" call would fall through to
    # random-init — exactly the failure mode Codex branch review P1
    # flagged on 2026-04-18.
    normalized = replace(String(seed_str), "**" => "^")
    ast_expr = try
        Meta.parse(normalized)
    catch
        return nothing
    end
    try
        return parse_expression(
            ast_expr;
            operators       = options.operators,
            variable_names  = varnames,
            expression_type = options.expression_type,
            node_type       = options.node_type,
        )
    catch
        return nothing
    end
end

function _resolve_variable_names(dataset::Dataset{T, L}) where {T, L}
    if hasproperty(dataset, :variable_names)
        names = getproperty(dataset, :variable_names)
        if names isa AbstractVector && !isempty(names)
            return [String(n) for n in names]
        end
    end
    nfeat = hasproperty(dataset, :nfeatures) ? getproperty(dataset, :nfeatures) : 0
    nfeat = nfeat isa Integer ? Int(nfeat) : 0
    return ["X$(i)" for i in 1:max(nfeat, 1)]
end

# ---------------------------------------------------------------------
# 3. Splice into population
# ---------------------------------------------------------------------

"""
    splice_seeds_into_population!(pop, seed_trees, dataset, options) -> (spliced, build_failed)

Overwrite the first `min(length(seed_trees), length(pop.members))`
members of `pop` with fresh `PopMember`s built around each parsed
seed tree. `cost` and `loss` are populated at construction via the
dataset-aware `PopMember(dataset, tree, options)` convenience
constructor (v1.11.3 upstream), which invokes `eval_cost` internally.

Returns a `(spliced::Int, build_failed::Int)` tuple so the convenience
wrapper `seed_initial_population!` can aggregate parsing and building
failures into a single summary warning.

A no-op (returns `(0, 0)`) when `seed_trees` is empty. The remainder of
the population remains random-init so migration and mutation have room
to propagate good seeds rather than being anchored to a homogeneous
initial pool.

Any individual PopMember construction that throws (e.g. a parsed tree
fails `check_constraints` or `eval_cost` hits a domain error) is
caught; that slot is left as-is and counted in `build_failed`.
"""
function splice_seeds_into_population!(
    pop::Population{T, L, N},
    seed_trees::Vector,
    dataset::Dataset{T, L},
    options::AbstractOptions,
)::Tuple{Int, Int} where {T, L, N}
    isempty(seed_trees) && return (0, 0)
    n = min(length(seed_trees), length(pop.members))
    spliced = 0
    build_failed = 0
    for k in 1:n
        tree = seed_trees[k]
        try
            # v1.11.3 dataset-aware convenience constructor computes
            # cost + loss + birth internally via eval_cost. Does NOT
            # accept a `deterministic` kwarg; determinism / ref bump is
            # handled via the `parent=-1` sentinel (no parent lineage).
            member = PopMember(dataset, tree, options; parent = -1)
            pop.members[k] = member
            spliced += 1
        catch
            build_failed += 1
        end
    end
    return (spliced, build_failed)
end

"""
    seed_initial_population!(pop, dataset, options) -> Int

Convenience wrapper that parses `options.initial_seed_strings` into
expression trees against `dataset + options`, then splices the resulting
trees into `pop`. Intended as the single call site from
`src/SymbolicRegression.jl::_initialize_search!`. Returns the number of
members actually spliced.

Aggregates failures into a single summary warning rather than per-seed
logs: on any non-zero parse or build failure, emits one `@warn` listing
the counts (`n_seeds / parsed_ok / build_failed / spliced`), so a batch
of 16 seeds with a grammar mismatch produces one log line instead of 16.

A no-op when `options.initial_seed_strings === nothing` or empty —
callers should still gate on that before invoking, but this helper is
defensive for safety.
"""
function seed_initial_population!(
    pop::Population{T, L, N},
    dataset::Dataset{T, L},
    options::AbstractOptions,
)::Int where {T, L, N}
    strings = options.initial_seed_strings
    (strings === nothing || isempty(strings)) && return 0
    n_seeds = length(strings)
    trees = []
    parsed_ok = 0
    for s in strings
        t = parse_seed_tree(s, options, dataset)
        if t === nothing
            continue
        end
        push!(trees, t)
        parsed_ok += 1
    end
    parse_failed = n_seeds - parsed_ok
    if isempty(trees)
        if parse_failed > 0
            @warn "SeedPopulation: all seeds failed to parse; falling back to random-init" n_seeds parse_failed
        end
        LAST_SPLICE_STATS[] = (
            n_seeds      = n_seeds,
            parsed_ok    = parsed_ok,
            parse_failed = parse_failed,
            build_failed = 0,
            spliced      = 0,
        )
        return 0
    end
    spliced, build_failed = splice_seeds_into_population!(
        pop, trees, dataset, options,
    )
    if parse_failed > 0 || build_failed > 0
        @warn "SeedPopulation: seed splice summary" n_seeds parsed_ok parse_failed build_failed spliced
    end
    LAST_SPLICE_STATS[] = (
        n_seeds      = n_seeds,
        parsed_ok    = parsed_ok,
        parse_failed = parse_failed,
        build_failed = build_failed,
        spliced      = spliced,
    )
    return spliced
end

end  # module SeedPopulationModule
