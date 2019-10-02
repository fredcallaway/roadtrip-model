using Distributed

@everywhere begin
    include("mdp.jl")

    using SplitApplyCombine
    using JSON
    using Memoize

    REWARD = DiscreteNonParametric(Float64[-100, -50, -35, -25], ones(4)/4)
    COST = 1

    function MetaMDP(edges, cost=COST)
        n_node = maximum(flatten(edges))
        graph = [Int[] for _ in 1:n_node]
        for (a, b) in edges
            push!(graph[a], b)
        end
        MetaMDP(graph=graph,
                reward_dist=REWARD,
                cost=cost,
                min_reward=-300);
    end

    @memoize function solve(m)
        V = ValueFunction(m)
        V(initial_belief(m))
        V
    end

    no_compute_value(m) = term_reward(m, initial_belief(m))
end

# %% ====================  ====================
graphs = open(JSON.parse, "graphs.json");
mdps = map(graphs) do g
    edges = map(e-> Int.(e) .+ 1, g["graph"]["edges"])
    MetaMDP(edges)
end;

none_vs = map(no_compute_value, mdps)

opt_vs = pmap(WorkerPool(workers()), mdps) do m
    V = solve(m)
    V(initial_belief(m))
end

# %% ====================  ====================

function labeler()
    idx = 0
    DefaultDict(passkey=true) do x
        idx += 1
        idx
    end
end

# %% ====================  ====================

function remote_solve(m)

end



# %% ====================  ====================
opt_vs = [solve(m)(initial_belief(m)) for m in mdps]

opt_vs - none_vs


function rand_value(m; N=1000)
    pol = RandomPolicy(m)
    mapreduce(+, 1:N) do i
        rollout(pol)
    end
end





vs = pmap(value, mdps)


# %% ==================== Solve MDPs ====================
@memoize function solve(m)
    V = ValueFunction(m)
    V(initial_belief(m))
    V
end
for m in mdps
    @time solve(m)
end

