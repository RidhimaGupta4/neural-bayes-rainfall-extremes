// GEV for the n winter maxima of one cell, with the prior in PREREGISTRATION.md:
//   mu ~ Uniform(10, 150), sigma ~ Uniform(2, 50), xi + 0.5 ~ Beta(9, 6).
// xi > 0 is a heavy tail. Parameter values whose support excludes an observation have
// likelihood zero (target = -inf), as the GEV requires.
functions {
  real gev_lpdf(vector y, real mu, real sigma, real xi) {
    int n = num_elements(y);
    vector[n] z = (y - mu) / sigma;
    if (abs(xi) < 1e-6) {
      return -n * log(sigma) - sum(z) - sum(exp(-z));
    }
    vector[n] t = 1 + xi * z;
    if (min(t) <= 0) return negative_infinity();
    return -n * log(sigma) - (1 + 1 / xi) * sum(log(t)) - sum(exp(-log(t) / xi));
  }
  real gev_return_level(real T, real mu, real sigma, real xi) {
    real yp = -log1m(1 / T);
    if (abs(xi) < 1e-6) return mu - sigma * log(yp);
    return mu + sigma / xi * (pow(yp, -xi) - 1);
  }
}
data {
  int<lower=1> n;
  vector[n] y;
}
parameters {
  real<lower=10, upper=150> mu;
  real<lower=2, upper=50> sigma;
  real<lower=-0.5, upper=0.5> xi;
}
model {
  // uniform priors on mu and sigma are implied by the bounds
  (xi + 0.5) ~ beta(9, 6);
  y ~ gev(mu, sigma, xi);
}
generated quantities {
  real log_sigma = log(sigma);
  real z10 = gev_return_level(10, mu, sigma, xi);
  real z100 = gev_return_level(100, mu, sigma, xi);
  real z1000 = gev_return_level(1000, mu, sigma, xi);
}
