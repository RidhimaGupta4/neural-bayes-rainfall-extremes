# stan/

`gev_cell.stan`: the reference model. A cell's 39 winter maxima are independent GEV(μ, σ, ξ),
with priors μ ~ Uniform(10, 150) mm, σ ~ Uniform(2, 50) mm and ξ + 0.5 ~ Beta(9, 6). Parameter
values whose support excludes an observation get zero likelihood. Generated quantities: log σ
and the 10-, 100- and 1,000-winter return levels.

This is exactly the prior and likelihood the neural Bayes estimator was trained on, so the
comparison judges the estimator against the posterior it is meant to approximate.
