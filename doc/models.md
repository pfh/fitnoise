
Noise models
===

The most typical model to use will be `Model_t()` for microarrays or RNA-Seq (in combination with Limma's `voom`). However we will first introduce a couple of simpler models.


Model_normal()
---

Errors `epsilon[i,j]` are assumed normally distributed, with equal varariance for all genes.

If a weights matrix is supplied with the dataset, the variance of each `epsilon[i,j]` is `variance / weight[i,j]`, where variance is the globally estimated `variance` parameter.

In Python, a weights matrix is given by creating a `fitnoise.Dataset` object with 

```
fitnoise.Dataset(y, dict(weights=weights))
```

In R, weights are passed as a parameter to `fitnoise.fit` or the `y` parameter of `fitnoise.fit` is given as a Limma `EList` object which includes a weight matrix.



Model_independent()
---

As with `Model_normal()`, but genes each have their own variance, independently.

Internally this is implemented using a multivariate t distribution with `df` (very close to) zero.



Model_t()
---

As with `Model_normal`, but each `epsilon[i,:]` is assumed to be a multivariate t distribution.

This closely mimics the behaviour of Limma.



Model_normal_per_sample() / Model_t_per_sample()
---

As with `Model_normal()` and `Model_t()`, however each sample receives its own variance estimate.

(This is similar to `arrayWeights` in Limma, but with the array weighting folded into the `eBayes` step and with a consistent noise model throughout.)

**Note:** Do not use this with two replicates per experimental group, per-sample variances are only identifiable with 3x replication or higher.



Model_normal_factors(n) / Model_t_factors(n)
---

Estimate a noise model containing `n` random effects. The direction and variance of each random effect is estimated.

This should behave similarly to RUV-4, however has not been thoroughly tested.

**Note:** This requires control genes to be specified.



Model_normal_patseq() / Model_t_patseq()
---

`y` are average tail lengths. Additionally a `counts` matrix also needs to be given. Where the count is zero, y should be NaN (Python) or NA (R).

