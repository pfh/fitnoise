
How to use Fitnoise
===

This is a basic run-through of how to use Fitnoise.

You can use Fitnoise either from R+ or Python.



Loading
---

|Python|R|
|---|---|
|`import fitnoise`|`library(fitnoise)`|


Choose a noise model
---

Fitnoise provides a variety of "noise models". fitnoise.Model_t behaves very similarly to limma.

|Python|R|
|---|---|
|```model = fitnoise.Model_t()```| We will pass an appropriate model selection in the next step. |


Fitting noise and estimating coefficients
---

<table>
<tr><th>Python</th><th>R</th></tr>
<tr><td><pre>
fitted = model.fit_noise(
    data, design).fit_coef()
</pre></td><td><pre>
fitted &lt;- fitnoise.fit(
    data, design, model="Model_t()")
</pre></td></tr></table>


* data is our matrix of expression levels. Columns are samples and rows are genes.

* design is our design matrix. Columns are coefficients and rows are samples.

For example if we had two samples in two groups, an appropriate design matrix might be:

|Python|R|
|---|---|
|`design = [[1,0], [1,0], [1,1], [1,1]]`|`design <- rbind(c(1,0),c(1,0),c(1,1),c(1,1))`|

The result contains:

|Python|R|Description|
|---|---|---|
|`fitted.coef`|`fitted$coef`|Matrix of fitted coefficients.|
|`fitted.coef_dists`||List of posterior multivariate distributions of coefficients.|
|`fitted.noise_p_values`|`fitted$noise.p.values`|Vector of "noise p-values". A small value for a particular gene may indicate that the noise model was a poor fit for that gene.|
|`fitted.noise_combined_p_value`|fitted$noise.combined.p.value`|Bonferroni corrected p-value of the noise p-values. A small value may indicate an overall poor fit for the noise model.|
|`repr(fitted)`|`fitted$description`|A summary of various important quantities from the noise fit.|


Testing a hypothesis
---

Say we want to test if the second coefficient is non-zero.

|Python|R|
|---|---|
|`tested = fitted.test(coef=[1])`|`tested <- fitted.test(coef=c(1))`|

It's also possible to test multiple coefficients at once, or a contrast of coefficients, or multiple contrasts.

The result contains:

|Python|R|Description|
|---|---|---|
|`tested.p_values`|`tested$p.values`|Vector of p values.|
|`tested.q_values`|`tested$q.values`|Vector of FDR values, calculated using the Benjamini & Hochberg method.|
|`tested.contrasts`|`tested$contrasts`|Matrix of the values of the contrasts or coefficients that were tested.|
|`tested.contrast_dists`||List of multivariate distributions of the contrasts that were tested.|





