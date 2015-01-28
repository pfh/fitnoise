
What is Fitnoise?
===

Context:

* We have a set of genes (or possibly other items, however we will talk about genes in this documentation).

* We have a set of biological samples.

* We have an expression level for each pair of gene and sample (or possibly some other quantity, but this documentation will generally talk about expression levels).

We model the expression level for each gene with a linear model then test whether certain coefficients (or contrasts) of this model are significantly non-zero. This is a very general formulation of hypothesis testing that encompasses many standard statistical tests. The simplest is testing for differential expression levels between two groups of samples.

If you have used Limma, this should all seem strangely familiar.

Formally, for each gene we model the expression levels y with:

    y = X beta + e
    
    e sampled from E
    
where

    y = a vector giving the expression level for each sample (for this gene)
    X = a design matrix (same for all genes, unless there are control genes)
    beta = coefficients to be estimated for this gene
    e = random noise (for this gene)
    E = the multivariate distribution from which e is sampled (for this gene)

The distribution E is a function of a set of hyper-parameters (to be estimated using the entire data-set) and possibly contextual information such as precision weights. This function is the *noise model*. Fitnoise provides a variety of noise models, and it is easy to add new models.

We assume:

* There are many more genes than samples.
* There are many more genes than hyper-parameters.

Fitnoise:

1. Estimates the hyper-parameters using REML.

2. Fits a linear model for each gene.

3. Performs a hypothesis test on each gene. (eg is it differentially expressed?)


Control genes
---

Some hyper-parameters to do with covariance may only be identifiable if there are nominated control genes. These genes are given a simplified design matrix (X) when estimating hyper-parameters.

This idea is taken from RUV.


Multivariate distributions
---

There are some contstraints on the multivariate distribution used for E: it must be possible to linearly transform it, compute marginal and conditional distributions, and compute densities and p-values.

Ok, this is the cool bit. One of the cool bits.

The most obvious form of distribution to use for E is a multivariate normal distribution. This yields a noise model where the variance is fixed globally (or a function of contextual information). Fitnoise supports this distribution, however it is generally better to allow the variance to vary from gene to gene somewhat. This can be achieved by using a multivariate t distribution.

There is a "degrees of freedom" (df) parameter in this distribution that determines to what extent the variance is based on a global estimate versus an estimate from the data from that gene. When the df is small, the variance is mostly based on the gene. When the df is large, the variance is mostly based on the global estimate.

This replicates Limma's moderated t-test and F-test capability.








