include("mdp.jl")

using SplitApplyCombine
using JSON
using Memoize

# %% ==================== Construct MDPs ====================
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
            min_reward=-400);
end

graphs = open(JSON.parse, "graphs.json");

mdps = map(graphs) do g
    edges = map(e-> Int.(e) .+ 1, g["graph"]["edges"])
    MetaMDP(edges)
end

# %% ==================== Solve MDPs ====================
@memoize function solve(m)
    V = ValueFunction(m)
    V(initial_belief(m))
    V
end
for m in mdps
    @time solve(m)
end



# %% ==================== Path distributions ====================

function path_distribution(m; N=1000)
    pths = paths(m)
    prob = zeros(length(pths))
    pol = OptimalPolicy(solve(m))
    for i in 1:N
        b = rollout(pol).belief
        prob .+= softmax(1e10.*path_values(m, b))
    end
    prob ./= N
end

for m in mdps
    println(round.(path_distribution(m); digits=2))
end

# %% ==================== Policy JSON ====================


encode_b(b::Belief) = join(map(b) do x
    isnan(x) ? "X" : -Int(x)
end, "-")

encode_qs(q) = join(map(q) do x
    isfinite(x) ? round(-x; digits=3) : "null"
end, ",")

possible(q) = findall(q .== maximum(q)) .- 1

function Q_dict(m::MetaMDP)
    d = Dict{String,String}()
    V = solve(m)
    seen = Set{String}()
    stack = [initial_belief(m)]
    function process(b)
        eb = encode_b(b)
        eb in seen && return
        qs = [Q(V, b, c) for c in 0:length(b)]
        push!(seen, eb)
        d[eb] = encode_qs(qs)
        for c in possible(qs)
            c == TERM && continue
            for (p,b1,r) in results(m, b, c)
                push!(stack, b1)
            end
        end
    end
    while !isempty(stack)
        process(pop!(stack))
    end
    d
end

open("Q.js", "w") do f
    write(f, "var optimal_solutions = ")
    write(f, json(map(Q_dict, mdps)))
end

# %% ====================  ====================



# %% ====================  ====================
Q_dict = Dict{String,Vector{Float64}}()

encode(b::Belief) = join(map(b) do x
    isnan(x) ? "X" : -Int(x)
end, "-")

possible = [NaN; support(REWARD)]

x = [possible for i in 1:length(b)]
x[1] = [0.]

for b in Iterators.product(x...)
    b = collect(b)
    qs = [Q(pol.V, b, c) for c in 0:length(b)]
    Q_dict[encode(b)] = qs
end

name = join(map(x->-Int(x), support(REWARD)), "-") * "_$COST"

# open("policies/$name.json", "w+") do f
#     write(f, json(pol_dict))
# end

open("Q11-$name.js", "w+") do f
    write(f, "var optimal_solution = ")
    write(f, json(Q_dict))
end

# %% ====================  ====================


# %% ====================  ====================
function grid_graph(depth, width)
    n = (depth * width) + 2
    graph = [Int[] for i in 1:n]
    push!(graph[1], 2:width+1...)
    for d in 0:(depth-1)
        println(d)
        for w in 1:width
            idx = 1 + d * width + w
            if d == depth-1
                push!(graph[idx], n)
            else
                push!(graph[idx], idx + width)
                push!(graph[idx], idx + (width + 1) % width)
            end
            # push!(graph[idx], n)
        end
    end
    graph
end
g = grid_graph(3, 3)

using JSON
open("../python/example.json", "w+") do f
    j = JSON.json([x .- 1 for x in g])
    println(j)
    write(f, j);
end


# %% ==================== Cogsci Graph ====================

let
    m = MetaMDP(
        graph=[[2, 6, 10], [3], [4, 5], [], [], [7], [8, 9], [], [], [11], [12, 13], [], []],
        cost=1,
        reward_dist=DiscreteNonParametric([-10, -5, 5, 10], ones(4)/4)
    );
    V = ValueFunction(m);
    b = initial_belief(m);
    @time V(b)
    println(V(b), "  ", length(V.cache))
end

# %% ==================== Chain ====================
let
    m = MetaMDP(graph=[[2,3], [4], [4], [5,6,7], [8], [8], [8], [] ])
    V = ValueFunction(m)
    b = initial_belief(m)
    b[1] = 0
    @time V(b)
    println(V(b), "  ", length(V.cache))
end

# %% ==================== Test symmetry_breaking_hash ====================
let
    m = MetaMDP(graph=[[2,3], [4], [4], [5,6,7], [8], [8], [8], [] ])
    b = initial_belief(m)
    b1 = copy(b)
    b1[2] = 1
    b2 = copy(b)
    b2[3] = 1
    b3 = copy(b)
    b3[4] = 1

    @assert symmetry_breaking_hash(m, b1) == symmetry_breaking_hash(m, b2)
    @assert symmetry_breaking_hash(m, b1) != symmetry_breaking_hash(m, b3)
end
