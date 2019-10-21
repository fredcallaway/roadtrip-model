include("mdp.jl")
include("box.jl")
include("results.jl")

using JSON
using SplitApplyCombine
using Memoize
using StatsFuns
using Optim
using DataFrames


version = "1.0"
data = open(JSON.parse, "../data/processed/$version/trials.json")
data = filter(data) do x
    length(x) == 8
end



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

function parse_mdp(t::Dict, cost)
   MetaMDP(map(x->Int.(x) .+ 1, t["edges"]), cost)
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

function Trial(t::Dict{String,Any}, cost)
    m = parse_mdp(t, cost)

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

function logp(t::Trial, α::Float64)
    V = solve(t.m)
    mapreduce(+, eachindex(t.bs)) do i
        q = Q(V, t.bs[i])
        log(softmax(α .* q)[t.cs[i]+1])
    end
end


# %% ====================  ====================

function logp(value::Function, t::Trial, α::Float64)
    mapreduce(+, eachindex(t.bs)) do i
        q = value(t.m, t.bs[i])
        log(softmax(α .* q)[t.cs[i]+1])
    end
end

rand_logp(t::Trial) = logp(voc1, t, 1e-10)


function logp(qs::Vector{Vector{Float64}}, cs::Vector{Int}, α::Float64)
    mapreduce(+, eachindex(cs)) do i
        log(softmax(α .* qs[i])[cs[i]+1])
    end
end

function fit_softmax(qs::Vector{Vector{Vector{Float64}}}, trials)
    opt = optimize(1e-10, 10) do α
        - mapreduce(+, eachindex(trials)) do i
            logp(qs[i], trials[i].cs, α)
        end
    end
    (α=opt.minimizer, logp=-opt.minimum)
end

function fit_softmax(Q::Function, trials)
    qs = map(trials) do t
        map(b->Q(t.m, b), t.bs)
    end
    fit_softmax(qs, trials)
end

function fit_participants(value)
    map(eachindex(data)) do i
        fit_softmax(value, trials[:, i])
    end
end

Q(m::MetaMDP, b::Belief) = Q(solve(m), b)


# %% ====================  ====================


function voc1(m::MetaMDP, b::Belief, c::Int)
    c == TERM && return 0.
    q = mapreduce(+, results(m, b, c)) do (p, b1, r)
        p * (term_reward(m, b1) + r)
    end
    q - term_reward(m, b)
end

voc1(m, b) = [voc1(m, b, c) for c in 0:length(b)]


function fit_cost_greedy(trials)
    res = optimize(0, 10) do cost
        -fit_softmax(voc1_cost(cost), trials).logp
    end

    (cost=res.minimizer, logp=-res.minimum,
     α=fit_softmax(voc1_cost(res.minimizer), trials).α)
end

voc1_cost(cost) = (m, b) -> voc1(change_cost(m,cost), b)
change_cost(m::MetaMDP, cost) = MetaMDP(m.graph, m.reward_dist, cost, m.min_reward)

