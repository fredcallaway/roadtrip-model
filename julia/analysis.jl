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

function Trial(t::Dict{String,Any})
    m = parse_mdp(t, 0)

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

map(trials) do t
    map(b->Q(solve(t.m), b), t.bs)
end



# %% ====================  ====================
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
@time solve(trials[1].m)
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

function logp(qs::Vector{Vector{Float64}}, cs::Vector{Int}, α::Float64)
    mapreduce(+, eachindex(cs)) do i
        log(softmax(α .* qs[i])[cs[i]+1])
    end
end

function fit_softmax(value, trials)
    qs = map(trials) do t
        map(b->value(t.m, b), t.bs)
    end
    res = optimize(1e-10, 10) do α
        - mapreduce(+, eachindex(trials)) do i
            logp(qs[i], trials[i].cs, α)
        end
    end
    (α=res.minimizer, logp=-res.minimum)
end

# function fit_softmax(value, trials)
#     res = optimize(1e-10, 10) do α
#         -sum(logp.(value, trials, α))
#     end
#     (α=res.minimizer, logp=-res.minimum)
# end

function fit_participants(value)
    map(eachindex(data)) do i
        fit_softmax(value, trials[:, i])
    end
end


# %% ====================  ====================
Q(m::MetaMDP, b::Belief) = Q(solve(m), b)
@time opt_fits = fit_participants(Q) |> DataFrame
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
    res = optimize(0, 100) do cost
        -fit_softmax(voc1_cost(cost), trials).logp
    end

    (cost=res.minimizer, logp=-res.minimum,
     α=fit_softmax(voc1_cost(res.minimizer), trials).α)
end

voc1_cost(cost) = (m, b) -> voc1(change_cost(m,cost), b)
change_cost(m::MetaMDP, cost) = MetaMDP(m.graph, m.reward_dist, cost, m.min_reward)


greedy_cost_fits = map(eachindex(data)) do i
    fit_cost_greedy(all_trials[:, i])
end |> DataFrame

# %% ====================  ====================

ress = filter(get_results("qs/1.0")) do res
    exists(res, :opt_fits)
end

fit_costs = map(ress) do res
    (fit=load(res, :opt_fits), cost=load(res, :cost))
end

sort!(fit_costs; by=f->f.cost)

fits, costs = invert(fit_costs)
F = combinedims(fits)
L = getfield.(F, :logp)



# %% ====================  ====================


│ Row │ α         │ logp     │
│     │ Float64   │ Float64  │
├─────┼───────────┼──────────┤
│ 1   │ 0.112655  │ -109.579 │
│ 2   │ 1.0e-10   │ -21.2851 │
│ 3   │ 1.0e-10   │ -16.7853 │
│ 4   │ 0.204151  │ -74.8366 │
│ 5   │ 0.141129  │ -66.25   │
│ 6   │ 1.0e-10   │ -21.3904 │
│ 7   │ 0.054212  │ -109.627 │
│ 8   │ 0.0951699 │ -71.9572 │
│ 9   │ 0.0706721 │ -95.6935 │

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
