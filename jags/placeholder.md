# jags/

`gev_cell.jags`: the single-cell GEV model in JAGS, with the same likelihood and prior as
`stan/gev_cell.stan`. JAGS has no GEV distribution, so the log-likelihood is added through the
"zeros trick" (a zero count from a Poisson whose mean is a large constant minus the log-likelihood).

Used by `R/11_jags.R` to cross-check the Stan reference on 16 cells.
