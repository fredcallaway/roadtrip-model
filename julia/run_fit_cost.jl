include("fit_cost.jl")

res = Results("fits/$version")
job = parse(Int, ARGS[1])
cost = logscale.(0:0.01:1, .1, 100)[job]
save(res, :cost, cost)
trials = combinedims([[Trial(d, cost) for d in sd] for sd in data])
@time qs = map(trials) do t
    map(b->Q(solve(t.m), b), t.bs)
end
save(res, :qs, qs)
@time opt_fits = fit_participants(Q)
save(res, :opt_fits, opt_fits)