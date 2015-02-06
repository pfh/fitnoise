
Noise models
===

Model_normal()
---

Errors `epsilon[i,j]` are assumed normally distributed, with equal varariance for all genes.

If a weights matrix is supplied with the dataset, the variance of each `epsilon[i,j]` of 
is `variance / weight[i,j]`, where variance is the globally estimated `variance` parameter.

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




Model_normal_patseq() / Model_t_patseq()
---

`y` are average tail lengths. Additionally a `counts` matrix also needs to be given. Where the count is zero, y should be NaN (Python) or NA (R).

