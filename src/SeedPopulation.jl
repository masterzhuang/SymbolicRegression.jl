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
using ..CoreModule.OperatorsModule: safe_log, safe_sqrt
using ..PopMemberModule: PopMember
using ..PopulationModule: Population
using ..LossFunctionsModule: eval_cost
using DynamicExpressions:
    AbstractExpression, AbstractExpressionNode, Node, parse_expression, get_tree
import JSON3

# Rationale for the `using ..CoreModule.OperatorsModule: safe_log,
# safe_sqrt` import above + the `_alias_pysr_safe_ops` rewrite below
# (Codex Lane D0.R task-mo6kbwnf-9081c9 research, 2026-04-20):
#
# PySR's `Options(unary_operators=["log","sqrt","inv(x)=1/x"])` wraps
# `log` / `sqrt` with domain-safe variants at enum construction time,
# so the OperatorEnum's actual callable symbols are `safe_log` /
# `safe_sqrt` (and a custom `inv`). Python-emitted sympy seed strings
# carry bare `log(...)` / `sqrt(...)` calls.
#
# The root cause, read from `DynamicExpressions/src/Parse.jl:265-269`:
# `parse_expression` resolves an `Expr(:call, callee, args...)`
# callee via `Core.eval(EmptyModule, callee)`. `EmptyModule` is a
# module with no top-level bindings, so bare `:log` / `:safe_log`
# Symbol lookups ALL fail before `evaluate_on` is consulted — no
# caller-module import, no `evaluate_on` vector, no `using`
# statement rescues this path.
#
# Lane D0.R verified on live PySR `opts + dataset`: rewriting the
# callee position of each `Expr(:call, ...)` to a `GlobalRef(Module,
# :name)` skips the EmptyModule evaluation entirely, because
# `GlobalRef` is a direct binding reference. The successful Julia
# snippet (A10) was:
#     ex = Meta.parse("X3*log(X1/X2)")
#     ex.args[3].args[1] = GlobalRef(SymbolicRegression, :safe_log)
#     parse_expression(ex; operators=opts.operators, ...)
# → `max_abs_err = 5.96e-8` on 30-sample evaluation vs the Python-
# computed target (machine epsilon for Float32).
#
# `_alias_pysr_safe_ops` below implements this: walk the AST and
# replace each `:log` / `:sqrt` call head with a `GlobalRef` to the
# safe-op binding in the parent SymbolicRegression module. `:inv`
# is left as-is for now because PySR's `inv(x) = 1/x` is often an
# anonymous binding outside SymbolicRegression's exports, so the
# right GlobalRef target is environment-dependent; the instrumented
# @warn surfaces any residual failure if a seed using `:inv` shows
# up in practice.

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
    # Match node_type's T to Dataset's T for the default-stack case
    # (PySR's `options.node_type === Node`, unparameterized) so the
    # returned Expression can be spliced into a PopMember without a
    # MethodError. For non-default expression stacks
    # (`ParametricExpression`, `TemplateExpression`, `GraphNode`, or
    # any user-registered `AbstractExpressionNode` subtype),
    # `options.node_type` is forwarded through unchanged — the
    # original v1.11.3 upstream contract is preserved for those
    # callers (see module docstring about upstream-compatibility).
    #
    # Why this matters: PySR's default stack sets
    # `options.node_type = Node` (the base unparameterized type).
    # `parse_expression` then fills in `Node{Float32}` by default,
    # producing `Expression{Float32, Node{Float32}, ...}`. But
    # PySR's Dataset is Float64 from numpy inputs, so
    # `PopMember(::Dataset{T, L}, ::AbstractExpression{T}, ::Options)
    # where {T, L}` has no method for `Dataset{Float64}` +
    # `Expression{Float32}` — MethodError → caught silently as
    # build_failed. Codex Lane D0.7 `task-mo6ryvel-dc0zb8`
    # (2026-04-20) captured this exact MethodError via the @warn in
    # splice_seeds_into_population!.
    #
    # For non-default stacks the user has opted into a specific
    # node_type (e.g. `ParametricNode`) whose T is already decided
    # by the user; overriding it to `Node{T}` would regress
    # expression-type compatibility (flagged by
    # PR #221 stop-review). Only override when the default base
    # `Node` is detected.
    #
    # The D0.R probe missed the T-mismatch altogether because it
    # reused an exemplar tree from `state[2].members[1].tree` whose
    # T happened to match the probe's ad-hoc dataset — i.e., D0.R
    # verified the parse path itself, not the Dataset↔Expression
    # T-matching requirement that PopMember enforces downstream.
    matched_node_type = options.node_type === Node ? Node{T} : options.node_type

    # First pass: try the raw AST so non-PySR callers whose
    # OperatorEnum carries plain `log` / `sqrt` (whose bare
    # Symbol callees happen to resolve in EmptyModule because
    # they map to Base functions) continue to parse bit-
    # identically with upstream.
    #
    # We UNWRAP the resulting `Expression{T, Node{T}, NT_...}` back
    # to its inner `Node{T}` via `get_tree(...)` and return that
    # bare tree. Rationale: PopMember's signature accepts
    # `Union{AbstractExpressionNode{T}, AbstractExpression{T}}`, but
    # PySR's Population stores members as
    # `PopMember{T, L, Expression{T, Node{T}, NamedTuple{(ops, vn),
    # Tuple{Nothing, Nothing}}}}` — Population sets metadata to
    # (nothing, nothing) because operators + variable_names live on
    # `options`, not per-expression. `parse_expression` builds the
    # Expression with metadata populated from its kwargs, producing
    # `NamedTuple{..., Tuple{OperatorEnum{...}, Vector{String}}}`
    # metadata. That parameterisation doesn't convert to the
    # Population's member type → MethodError at the
    # `pop.members[k] = member` assignment. Codex Lane D0.9b
    # task-mo6vb0wu-812ugp (2026-04-20) captured this exact
    # MethodError via the 304205a2 splice @warn. Returning the bare
    # Node lets PopMember wrap it with default (Nothing, Nothing)
    # metadata that matches the Population's member type.
    raw_err = nothing
    try
        expr = parse_expression(
            ast_expr;
            operators       = options.operators,
            variable_names  = varnames,
            expression_type = options.expression_type,
            node_type       = matched_node_type,
        )
        return get_tree(expr)
    catch err_raw
        raw_err = err_raw
    end
    # Second pass: rewrite safe-op callees to GlobalRef before
    # parse_expression sees them. See the big comment near the top
    # of this module + Codex Lane D0.R task-mo6kbwnf-9081c9
    # verification (max_abs_err 5.96e-8 on live PySR opts).
    aliased = _alias_pysr_safe_ops(ast_expr)
    try
        expr = parse_expression(
            aliased;
            operators       = options.operators,
            variable_names  = varnames,
            expression_type = options.expression_type,
            node_type       = matched_node_type,
        )
        return get_tree(expr)
    catch aliased_err
        # Diagnostic: emit both errors so any residual failure
        # (e.g. a new Symbol that needs GlobalRef aliasing)
        # surfaces the exception instead of disappearing into
        # silent parse_failed counters. The five-round repair
        # journey (f5af37dd → c81b6f59 → f8fce565 → 803b8ebd →
        # THIS patch) would have been impossible without this
        # instrumentation.
        @warn (
            "SeedPopulation: parse_seed_tree both raw and aliased " *
            "parses failed; falling back (seed dropped)"
        ) seed_str raw_err aliased_err
        return nothing
    end
end

"""
    _alias_pysr_safe_ops(x)

Return a new AST with every `:log(...)` / `:sqrt(...)` call head
rewritten to a `GlobalRef` pointing at the corresponding safe-op
function. Pure / non-mutating; walks the Julia AST recursively.

Why `GlobalRef` and not bare Symbol: `DynamicExpressions.Parse`
resolves an `Expr(:call, callee, args...)` callee by
`Core.eval(EmptyModule, callee)` (see
`DynamicExpressions/src/Parse.jl:265-269`). `EmptyModule` is a
blank module with no top-level bindings, so a Symbol like `:log`
or `:safe_log` CANNOT be resolved there regardless of what the
calling module has imported, exported, or passed via
`evaluate_on`. A `GlobalRef(Mod, :name)`, by contrast, is a
direct binding reference — `Core.eval` returns the bound value
without needing any scope lookup. Codex Lane D0.R
task-mo6kbwnf-9081c9 (2026-04-20) verified this empirically on
live PySR Options + Dataset: `max_abs_err = 5.96e-8` between the
parsed-tree evaluation and the Python-computed target (machine
epsilon for Float32).

The target module is `parentmodule(@__MODULE__)` (i.e.
`SymbolicRegression` when this module is loaded inside the fork).
Both `safe_log` and `safe_sqrt` are exported at the
SymbolicRegression top level (see `src/SymbolicRegression.jl`
export list), so `GlobalRef(SymbolicRegression, :safe_log)`
resolves to the same function `Options(unary_operators=["log"])`
wraps into the OperatorEnum.

`:inv` is NOT aliased: PySR's `inv(x) = 1/x` is a user-defined
binding that lives in Python-driven Main scope, not in
SymbolicRegression exports. If a seed carrying `inv(...)` shows
up in practice, the instrumented `@warn` surfaces the exact
failure so we can add the right GlobalRef target then.

No-op on anything other than `:call` heads for `:log` / `:sqrt`
— symbols like `:X1`, numeric literals, and nested
`log(log(X1))` forms are all handled correctly.
"""
function _alias_pysr_safe_ops(x)
    if x isa Expr
        new_args = Any[_alias_pysr_safe_ops(a) for a in x.args]
        if x.head === :call && !isempty(new_args)
            sr = parentmodule(@__MODULE__)
            if new_args[1] === :log
                new_args[1] = GlobalRef(sr, :safe_log)
            elseif new_args[1] === :sqrt
                new_args[1] = GlobalRef(sr, :safe_sqrt)
            end
        end
        return Expr(x.head, new_args...)
    end
    return x
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
    first_build_err = nothing
    first_build_err_tree_type = nothing
    for k in 1:n
        tree = seed_trees[k]
        try
            # v1.11.3 dataset-aware convenience constructor computes
            # cost + loss + birth internally via eval_cost. It
            # REQUIRES explicit `deterministic` and `parent` kwargs —
            # the docstring previously said `parent=-1` sentinel was
            # enough, but Codex Lane D0.8 task-mo6t6o39-69atf2
            # (2026-04-20) captured the PopMember MethodError:
            #   ArgumentError: You must declare `deterministic` as
            #   `true` or `false`, it cannot be left undefined.
            # Forward options.deterministic verbatim so the seeded
            # slots match the same reproducibility contract as the
            # rest of the population that PySR's standard
            # initialization builds.
            member = PopMember(
                dataset, tree, options;
                parent        = -1,
                deterministic = options.deterministic,
            )
            pop.members[k] = member
            spliced += 1
        catch build_err
            build_failed += 1
            if first_build_err === nothing
                first_build_err = build_err
                first_build_err_tree_type = typeof(tree)
            end
        end
    end
    # Codex Lane D0.6b (ASOUL-SR task-mo6pe2xe-ej59eh, 2026-04-20):
    # the GlobalRef parse fix landed (parsed_ok=2 per dataset) but
    # PopMember construction failed silently with build_failed=2 on
    # every dataset. Logging the first caught exception surfaces the
    # concrete cause (likely check_constraints / eval_cost domain /
    # node_type mismatch) so the next repair round is one-shot again
    # instead of blind.
    if build_failed > 0 && first_build_err !== nothing
        @warn (
            "SeedPopulation: PopMember construction failed for at " *
            "least one parsed seed — falling back to random for those slots"
        ) n_attempts=n spliced build_failed tree_type=first_build_err_tree_type err=first_build_err
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
