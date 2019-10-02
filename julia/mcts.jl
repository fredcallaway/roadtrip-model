using POMDPModels # for the GridWorld problem
using MCTS
mdp = GridWorld()
solver = MCTSSolver(n_iterations=50, depth=20, exploration_constant=5.0)
policy = solve(solver, mdp)
a = action(policy, s)


mdp
# %% ====================  ====================


using POMDPs
using Random # for AbstractRNG
using POMDPModelTools # for Deterministic


@with_kw struct NewMDP <: MDP{Belief,Int}
    n_arm::Int = 3
    obs_sigma::Float64 = 1
    sample_cost::Float64 = 0.001
    switch_cost::Float64 = 1
end

m = MetaMDP()
s = State(m)
b = Belief(s)
p = NewMDP()
rng = MersenneTwister()

POMDPs.initialstate_distribution(m::NewMDP) = Deterministic(Belief(s))

function POMDPs.generate_s(p::NewMDP, b::Belief, c::Int, rng::AbstractRNG)
    b = copy(b)
    transition!(b, s, c)
    b
end
function POMDPs.reward(p::NewMDP, b::Belief, c::Int)
    c == TERM ? term_reward(b) : cost(m, b, c)
    s*p.r_hungry + a*p.r_feed
end


generate_s(p, b, 2, rng)


