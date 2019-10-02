include("mdp.jl")
using JSON
using SplitApplyCombine
using Memoize
using StatsFuns
using Optim
using DataFrames

# %% ====================  ====================
version = "1.0"
data = open(JSON.parse, "../data/processed/$version/trials.json")
data = filter(data) do x
    length(x) == 8
end

# %% ====================  ====================
REWARD = DiscreteNonParametric(Float64[-100, -50, -35, -25], ones(4)/4)

function MetaMDP(edges, cost)
    n_node = maximum(flatten(edges))
    graph = [Int[] for _ in 1:n_node]
    for (a, b) in edges
        push!(graph[a], b)
    end
    MetaMDP(graph=graph,
            reward_dist=REWARD,
            cost=cost,
            min_reward=-400);
end

function parse_mdp(t::Dict)
   MetaMDP(map(x->Int.(x) .+ 1, t["edges"]), 5)
end

@memoize Dict function solve(m)
    V = ValueFunction(m)
    V(initial_belief(m))
    V
end

# %% ====================  ====================

struct Trial
    m::MetaMDP
    bs::Vector{Belief}
    cs::Vector{Int}
    path::Vector{Int}
end

function Trial(t::Dict{String,Any})
    m = parse_mdp(t)

    bs = Belief[]
    cs = Int[]

    b = initial_belief(m)

    for (c, value) in t["reveals"]
        c += 1  # 0->1 indexing
        push!(bs, copy(b))
        push!(cs, c)
        b[c] = -value
    end
    push!(bs, b)
    push!(cs, TERM)
    path = Int.(t["route"] .+ 1)[2:end-1]
    # @assert path in paths(m)
    Trial(m, bs, cs, path)
end

function Base.show(io::IO, t::Trial)
    print(io, "T")
end

function labeler()
    idx = 0
    @memoize Dict function f(x)
        idx += 1
        idx
    end
end

trials = combinedims([[Trial(d) for d in sd] for sd in data])
trials = trials[2:end, :]
label = labeler()
M = map(trials) do t
    label(t.m)
end

# %% ====================  ====================

function logp(t::Trial, α::Float64)
    V = solve(t.m)
    mapreduce(+, eachindex(t.bs)) do i
        q = Q(V, t.bs[i])
        log(softmax(α .* q)[t.cs[i]+1])
    end
end

rand_logp(t::Trial) = logp(t, 1e-10)

fits = map(eachindex(data)) do i
    res = optimize(1e-10, 10) do α
        -sum(logp.(trials[:, i], α))
    end
    (α = res.minimizer,
     like = -res.minimum,
     vs_radn = -res.minimum - sum(rand_logp.(trials[:, i])))
end |> DataFrame


# %% ====================  ====================

function logp(value::Function, t::Trial, α::Float64)
    mapreduce(+, eachindex(t.bs)) do i
        q = value(t.m, t.bs[i])
        log(softmax(α .* q)[t.cs[i]+1])
    end
end

function fit_softmax(value, trials)
    res = optimize(1e-10, 10) do α
        -sum(logp.(value, trials, α))
    end
    (α=res.minimizer, logp=-res.minimum)
end

function fit_participants(value)
    map(eachindex(data)) do i
        fit_softmax(value, trials[:, i])
    end
end

@time greedy_fits = fit_participants(voc1) |> DataFrame



fits = map(eachindex(data)) do i
    res = optimize(1e-10, 10) do α
        -sum(logp.(value, trials[:, i], α))
    end
    (α = res.minimizer,
     like = -res.minimum,
     vs_radn = -res.minimum - sum(rand_logp.(trials[:, i])))
end |> DataFrame

logp.(voc1, trials[:, 1], 1.)
# %% ====================  ====================

println(fits)
