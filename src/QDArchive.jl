#######################################################################
# QDArchive.jl — Search-time MAP-Elites archive for symbolic regression
#
# This module lives in the `asoul-qd-v1` fork of
# `MilesCranmer/SymbolicRegression.jl`. Its job is to maintain a
# multi-dimensional archive of elite expressions indexed by
# `(complexity, variable_signature, operator_motif)` so that migration
# back into live populations is diversity-biased rather than
# parsimony-biased.
#
# See the ASOUL-SR design document:
#   docs/notes/qd_search_time_design_v1.md  (project-local path)
#
# Rollback invariant: with `options.use_qd_archive == false`, none of
# this module's functions are called. Every allocation, update, and
# migration path is guarded at the call site in
# `src/SymbolicRegression.jl::_main_search_loop!`, so setting the master
# switch to `false` reproduces upstream SymbolicRegression.jl behaviour
# byte-for-byte.
#
# This file is Part 1 of 2 (rule 9 in AGENTS.md). Part 2 appends the
# admission, migration sampling, compression, and env-var helpers.
#######################################################################

module QDArchiveModule

using Random
using ..CoreModule: AbstractOptions, DATA_TYPE, LOSS_TYPE
using ..PopMemberModule: PopMember
using ..ComplexityModule: compute_complexity
using ..HallOfFameModule: HallOfFame
using DynamicExpressions: AbstractExpression, AbstractExpressionNode, tree_mapreduce

export QDArchive,
       QDKey,
       variable_signature,
       operator_motif,
       update_qd_archive!,
       sample_qd_archive,
       compress_to_hall_of_fame!,
       qd_options_from_env,
       qd_archive_stats

# ---------------------------------------------------------------------
# 1. Archive key and struct
# ---------------------------------------------------------------------

"""
    QDKey

Cell key for the MAP-Elites archive. The three coordinates are:

  1. `complexity::Int` — tree node count; same definition used by the
     upstream HallOfFame indexing.
  2. `variable_signature::Tuple{Vararg{Int}}` — sorted, deduplicated
     tuple of feature indices that appear as leaves in the expression.
  3. `operator_motif::UInt64` — compact hash of the arity-separated
     operator multiset. Two expressions with the same multiset of
     operators (regardless of argument order or feature identity)
     collide into one cell, which is exactly the niching equivalence
     quality-diversity wants.
"""
const QDKey = Tuple{Int, Tuple{Vararg{Int}}, UInt64}

"""
    QDArchive{T,L,N,PM}

MAP-Elites archive holding at most one elite `PopMember` per `QDKey`
cell, plus capacity bookkeeping.

Fields
------
- `cells::Dict{QDKey, PM}` — one elite per niche.
- `max_cells::Int` — hard upper bound; eviction triggers above this.
- `ncells_per_complexity::Dict{Int,Int}` — running per-complexity cell
  counts, used by the eviction heuristic.
- `hits::Int` — total admissions / improvements (monitoring).
- `rejects::Int` — rejected admissions (monitoring).

The archive is **additive**: it runs alongside the upstream
`HallOfFame` rather than replacing it. The existing `HallOfFame`
remains the canonical 1-D reporting surface; the archive only changes
what gets fed into `hof_migration`.
"""
mutable struct QDArchive{
    T <: DATA_TYPE,
    L <: LOSS_TYPE,
    N <: AbstractExpression{T},
    PM <: PopMember{T,L,N},
}
    cells::Dict{QDKey, PM}
    max_cells::Int
    ncells_per_complexity::Dict{Int, Int}
    hits::Int
    rejects::Int
end

function QDArchive(
    ::Type{PM}; max_cells::Int = 4096,
) where {T, L, N, PM <: PopMember{T, L, N}}
    return QDArchive{T, L, N, PM}(
        Dict{QDKey, PM}(),
        max_cells,
        Dict{Int, Int}(),
        0,
        0,
    )
end

Base.length(a::QDArchive) = length(a.cells)
Base.isempty(a::QDArchive) = isempty(a.cells)

# ---------------------------------------------------------------------
# 2. Feature extractors (variable signature + operator motif)
# ---------------------------------------------------------------------

"""
    variable_signature(tree::AbstractExpression) -> Tuple{Vararg{Int}}

Return the sorted, deduplicated tuple of feature indices that appear as
leaves in `tree`. Constants contribute nothing. An expression whose
only leaves are constants yields the empty tuple `()`.

Equivalent in spirit to the Python-side
`frozenset(r.variable_names)` used in
`src/asoul_sr/baselines/pysr_runner.py::_structure_bucket_key`,
lifted into Julia so the archive can be updated during search instead
of after it.
"""
function variable_signature(tree::AbstractExpression)
    # DynamicExpressions.tree_mapreduce passes the branch_fn result PLUS
    # one result per child into the reducer, so a binary node produces a
    # 3-argument call `(branch_val, left_val, right_val)`. The upstream
    # Complexity.jl works around this by using `+` (variadic) as the
    # reducer; we use `vcat` which is also variadic and concatenates any
    # number of vectors. A 2-arg closure would MethodError on binary
    # nodes — that was the bug in the v1.11.3 rebase.
    features = tree_mapreduce(
        leaf -> leaf.constant ? Int[] : Int[leaf.feature],
        _branch -> Int[],
        vcat,
        tree,
        Vector{Int};
        break_sharing = Val(true),
    )
    unique!(sort!(features))
    return Tuple(features)
end

"""
    operator_motif(tree::AbstractExpression) -> UInt64

Compact hash of the arity-separated operator multiset in `tree`.

Why separated by arity: two unary ops are never algebraically equivalent
to one binary op, so collapsing them into the same motif would create
spurious niching.

Why a hash rather than a sorted multiset: a tuple key would fragment the
archive every time a single operator differs under random mutation
(which is the common case). The hash compacts near-identical motifs
into the same cell and keeps the archive size bounded. The upper bound
of 32 operator slots per arity is more than enough for any realistic
`Options` binary/unary operator count.
"""
function operator_motif(tree::AbstractExpression)
    unary_counts  = zeros(Int, 32)
    binary_counts = zeros(Int, 32)
    # The branch_fn closure records counts via side effects into
    # unary_counts / binary_counts; the reducer return value is unused
    # (we return the hash of the side-effect arrays below). A variadic
    # `(args...) -> nothing` reducer handles both unary (2-arg) and
    # binary (3-arg) calls from DynamicExpressions.tree_mapreduce —
    # same fix as variable_signature above.
    tree_mapreduce(
        _leaf -> nothing,
        function (br)
            if br.degree == 1
                @inbounds unary_counts[br.op]  += 1
            elseif br.degree == 2
                @inbounds binary_counts[br.op] += 1
            end
            return nothing
        end,
        (args...) -> nothing,
        tree,
        Nothing;
        break_sharing = Val(true),
    )
    u = UInt64(hash(unary_counts)  & 0x00000000ffffffff)
    b = UInt64(hash(binary_counts) & 0x00000000ffffffff)
    return (u << 32) | b
end

# ---------------------------------------------------------------------
# 3. Admission policy
# ---------------------------------------------------------------------

"""
    update_qd_archive!(archive, members, options)

Fold a vector of candidate `PopMember`s into the archive. For each
candidate we compute its `QDKey` and either:

  (a) admit it into an empty cell (first occupant of that niche), or
  (b) replace the existing cell occupant if the new candidate has
      strictly lower `cost`, or
  (c) reject it.

`cost` here is the same cost used by the upstream HallOfFame update
(`loss + parsimony`), so the within-cell selection is **not** more
permissive than upstream — it is only the cross-cell preservation that
differs. This is important for the rollback invariant: if every
candidate happens to fall into the same cell, the archive collapses to
the existing 1-D HOF behaviour automatically.

Invalid members (size out of range, constraint violation) are silently
skipped, matching `update_hall_of_fame!`'s contract.

Eviction is triggered lazily when `length(archive.cells) > max_cells`.
The eviction heuristic evicts the worst-cost cell inside the most
populated complexity band, which drops stale elites from crowded
niches before touching rare structural discoveries. If the caller
exceeds `max_cells` by more than a factor of two between cycles, this
single eviction pass may leave the archive slightly over-capacity — a
subsequent `update_qd_archive!` call will trim further. We accept this
rather than run unbounded eviction loops inside the hot path.
"""
function update_qd_archive!(
    archive::QDArchive{T,L,N,PM},
    members::AbstractVector{PM},
    options::AbstractOptions,
) where {T, L, N, PM <: PopMember{T,L,N}}
    @inbounds for m in members
        size_i = compute_complexity(m, options)
        (0 < size_i <= options.maxsize) || continue
        # Note: v1.11.3's update_hall_of_fame! does not call check_constraints
        # (added in 2.0-alpha). We mirror that to keep update_qd_archive!
        # symmetric with v1.11.3's HOF update behaviour.

        key = (size_i, variable_signature(m.tree), operator_motif(m.tree))
        existing = get(archive.cells, key, nothing)
        if existing === nothing
            archive.cells[key] = copy(m)
            archive.ncells_per_complexity[size_i] =
                get(archive.ncells_per_complexity, size_i, 0) + 1
            archive.hits += 1
        elseif m.cost < existing.cost
            archive.cells[key] = copy(m)
            archive.hits += 1
        else
            archive.rejects += 1
        end
    end

    if length(archive.cells) > archive.max_cells
        _evict_one_worst!(archive)
    end
    return archive
end

"""
    _evict_one_worst!(archive)

Internal eviction step. Finds the most populated complexity band, then
within that band finds the cell whose occupant has the worst (highest)
cost, and deletes it. Exported only for tests.

Rationale: evicting by cost alone would preferentially destroy
high-complexity structural discoveries (which naturally have larger
cost under parsimony). Evicting within the most populated band ensures
that rare niches survive the capacity pressure.
"""
function _evict_one_worst!(archive::QDArchive{T,L,N,PM}) where {T,L,N,PM}
    isempty(archive.cells) && return archive

    # 1. Pick the most populated complexity band.
    target_complexity = 0
    target_count      = -1
    for (c, n) in archive.ncells_per_complexity
        if n > target_count
            target_complexity = c
            target_count      = n
        end
    end
    target_count <= 0 && return archive

    # 2. Within that band, pick the cell with the worst cost.
    worst_key::Union{Nothing, QDKey} = nothing
    worst_cost = nothing
    for (k, v) in archive.cells
        k[1] == target_complexity || continue
        if worst_cost === nothing || v.cost > worst_cost
            worst_cost = v.cost
            worst_key  = k
        end
    end

    if worst_key !== nothing
        delete!(archive.cells, worst_key)
        archive.ncells_per_complexity[target_complexity] =
            max(0, archive.ncells_per_complexity[target_complexity] - 1)
    end
    return archive
end

# ---------------------------------------------------------------------
# 4. Migration sampling
# ---------------------------------------------------------------------

"""
    sample_qd_archive(archive, k; rng=Random.default_rng()) -> Vector{PM}

Sample up to `k` elites from the archive, **weighted uniformly across
cells, not across cost**. This is the diversity-biased migration source
that breaks the parsimony feedback loop identified in
`docs/notes/qd_search_time_design_v1.md` §2.4.

Returned members are deep copies — the caller owns them and is free to
mutate without corrupting the archive.

If the archive is empty, returns an empty vector, letting the call
site fall back to the upstream dominating-set migration path.
"""
function sample_qd_archive(
    archive::QDArchive{T,L,N,PM},
    k::Int;
    rng::AbstractRNG = Random.default_rng(),
) where {T, L, N, PM <: PopMember{T,L,N}}
    n = length(archive.cells)
    n == 0 && return PM[]
    k <= 0 && return PM[]
    keys_arr = collect(keys(archive.cells))
    m = min(k, n)
    # Sample without replacement so one rare niche cannot dominate the
    # migrants vector even at tiny archive sizes.
    idxs = randperm(rng, n)[1:m]
    return PM[copy(archive.cells[keys_arr[i]]) for i in idxs]
end

# ---------------------------------------------------------------------
# 5. Compression to 1-D HallOfFame (reporting compatibility)
# ---------------------------------------------------------------------

"""
    compress_to_hall_of_fame!(hof, archive, options)

Fold the archive into a 1-D `HallOfFame`: for each complexity slot,
keep the lowest-cost elite whose `QDKey[1] == complexity`. This is
called at end-of-search (and optionally between cycles) so downstream
consumers that read the HOF continue to see the usual 1-D shape —
except now the per-complexity winners were drawn from a structurally
diverse pool rather than from a parsimony-collapsed one.

This function only **augments** the HOF: it never deletes or downgrades
an existing HOF entry. If the current HOF occupant at complexity `c`
already beats everything in the archive at that complexity, the
existing HOF entry is preserved unchanged.
"""
function compress_to_hall_of_fame!(
    hof::HallOfFame{T,L,N},
    archive::QDArchive{T,L,N,PM},
    options::AbstractOptions,
) where {T, L, N, PM <: PopMember{T,L,N}}
    @inbounds for (key, m) in archive.cells
        size_i = key[1]
        (0 < size_i <= options.maxsize) || continue
        if !hof.exists[size_i] || m.cost < hof.members[size_i].cost
            hof.members[size_i] = copy(m)
            hof.exists[size_i]  = true
        end
    end
    return hof
end

# ---------------------------------------------------------------------
# 6. Environment-variable bridge
# ---------------------------------------------------------------------

"""
    qd_options_from_env() -> NamedTuple

Read the four QD environment variables and return a NamedTuple with
the resolved values. Called by the `Options` constructor to populate
its new fields when the caller does not pass them explicitly.

The env-var bridge exists because PySR's `PySRRegressor.__init__`
validates kwargs against a whitelist in our pinned version range, so
the Python side cannot reliably forward new Julia `Options` fields.
See `fork_staging/symbolicregression_jl_asoul_qd_v1/README.md` for the
rationale and the Python-side helper that sets these vars.

| Env var                | Field              | Default |
|------------------------|--------------------|---------|
| `ASOUL_USE_QD_ARCHIVE` | `use_qd_archive`   | `false` |
| `ASOUL_QD_MIGRATION`   | `qd_migration`     | `true`  |
| `ASOUL_QD_MIGRATION_K` | `qd_migration_k`   | `16`    |
| `ASOUL_QD_MAX_CELLS`   | `qd_max_cells`     | `4096`  |

Bool vars accept `"1" / "true" / "yes" / "on"` (case-insensitive).
Int vars accept any `parse(Int, ...)`-compatible string; malformed
values fall back to the default and emit a `@warn`.
"""
function qd_options_from_env()
    return (
        use_qd_archive = _env_bool("ASOUL_USE_QD_ARCHIVE", false),
        qd_migration   = _env_bool("ASOUL_QD_MIGRATION",   true),
        qd_migration_k = _env_int("ASOUL_QD_MIGRATION_K",  16),
        qd_max_cells   = _env_int("ASOUL_QD_MAX_CELLS",    4096),
    )
end

function _env_bool(name::AbstractString, default::Bool)::Bool
    raw = get(ENV, name, nothing)
    raw === nothing && return default
    s = lowercase(strip(raw))
    return s in ("1", "true", "yes", "on", "t", "y")
end

function _env_int(name::AbstractString, default::Int)::Int
    raw = get(ENV, name, nothing)
    raw === nothing && return default
    try
        return parse(Int, strip(raw))
    catch
        @warn "QDArchive: ignoring malformed $name=$raw; falling back to $default"
        return default
    end
end

# ---------------------------------------------------------------------
# 7. Monitoring helper
# ---------------------------------------------------------------------

"""
    qd_archive_stats(archive) -> NamedTuple

Return a small NamedTuple with `(hits, rejects, n_cells,
per_complexity)` suitable for JSON-serialising to the 14d report
directory. The `per_complexity` field is a copy so callers can store
it without risking later archive mutation.
"""
function qd_archive_stats(archive::QDArchive)
    return (
        hits           = archive.hits,
        rejects        = archive.rejects,
        n_cells        = length(archive.cells),
        per_complexity = copy(archive.ncells_per_complexity),
    )
end

end  # module QDArchiveModule
