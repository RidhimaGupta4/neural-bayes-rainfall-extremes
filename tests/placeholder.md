# tests/

`test_nbe.py`: unit tests for the estimator's building blocks. Run with `python tests/test_nbe.py`.

* GEV quantiles match SciPy's `genextreme` (including the Gumbel limit ξ → 0)
* The return-level definition: F(z₁₀₀) = 0.99
* Prior draws stay within their bounds, and ξ has the stated mean
* The input features do not depend on the order of the sample, and the target transformations invert exactly
* Predicted quantiles never cross
* The pinball loss is minimised at the target quantile
