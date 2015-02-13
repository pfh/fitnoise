
Noise models
===

The most typical model to use will be `Model_t()` for microarrays. `Model_t()` can also be used for RNA-Seq in combination with Limma's `voom` function. However we will introduce a couple of simpler models first.

Contextual information for noise models
---

Some noise models require contextual information, such as a weights or counts matrix.

In Python, instead of passing a matrix to `model.fit`, pass an instance of `fitnoise.Dataset`, eg `fitnoise.Dataset(y, {"weights":weights})`.

In R, you can use a Limma `EList` object containing `elist$weights` or `elist$other$counts`, or you can pass `weights=` or `counts=` as extra paramters to `fitnoise.fit`.



Model_normal()
---

Errors `epsilon[i,j]` are assumed normally distributed, with equal varariance for all genes.

If a weights matrix is supplied with the dataset, the variance of each `epsilon[i,j]` is `variance / weight[i,j]`, where variance is the globally estimated `variance` parameter.

In Python, a weights matrix is given by creating a `fitnoise.Dataset` object with

```
fitnoise.Dataset(y, dict(weights=weights))
```

In R, weights are passed as a parameter to `fitnoise.fit` or the `y` parameter of `fitnoise.fit` is given as a Limma `EList` object which includes a weight matrix. Note that this means you can directly use the output of Limma's `voom` function as input to `fitnoise.fit`.



Model_independent()
---

As with `Model_normal()`, but genes each have their own variance, independently. This is like performing a t-test or F-test on each gene independently.

Internally this is implemented using a multivariate t distribution with `df` (very close to) zero.



Model_t()
---

As with `Model_normal`, but each `epsilon[i,:]` is assumed to be a multivariate t distribution.

This closely mimics the behaviour of Limma.



Model_normal_per_sample() / Model_t_per_sample()
---

As with `Model_normal()` and `Model_t()`, however each sample receives its own variance estimate.

(This is similar to `arrayWeights` in Limma, but with the array weight estimation performed simultaneously with estimation of other noise parameters.)

**Note:** Unless you specify control genes, do not use this with two replicates per experimental group. Per-sample variances are only identifiable with 3x replication or higher, or with control genes specified.



Model_normal_factors(n) / Model_t_factors(n)
---

Estimate a noise model containing `n` random effects. The direction and variance of each random effect is estimated.

This should behave similarly to RUV-4, however has not been thoroughly tested.

**Note:** This requires control genes to be specified.



Model_normal_patseq() / Model_t_patseq()
---

`y` are average tail lengths. Additionally a `counts` matrix needs to be given. Where the count is zero, y should be NaN (Python) or NA (R).

The accuracy of each tail length is a function of per-read variance and per-sample variance. The more reads that contributed to the average, the more accurate, but because there is also sample-level variance the accuracy doesn't go to zero with more and more reads.

`Model_normal_patseq_per_sample()` and `Model_t_patseq_per_sample()` are also available.

*Note:* Model_t_patseq_v1() was used for the Lorne Genome Conference 2015 poster. This cheats slightly by basing the noise on individual tail lengths. I don't think this makes a practical difference, but for the sake of punctilious correctness this current version uses the tail length averaged over all samples (weighted by count).


