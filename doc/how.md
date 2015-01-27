
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
<tr><th>Python</th><th>R</th>
<tr><td>
```
fitted = model.fit_noise(
    data, design
    ).fit_coef()
```
</td><td>
```
fitted <- fitnoise.fit(
    data, design, model="Model_t()")
```
</td></tr></table>

* data is our matrix of expression levels. Columns are samples and rows are genes.

* design is our design matrix. Columns are coefficients and rows are samples.

For example if we had two samples in two groups, an appropriate design matrix might be:

|Python|R|
|---|---|
|`design = [[1,0], [1,0], [1,1], [1,1]]`|`design <- rbind(c(1,0),c(1,0),c(1,1),c(1,1))`|


Testing a hypothesis
---

Say we want to test if the second coefficient is non-zero.

Python:

```python
tested = fitted.test(coef=[1])
```

R:

```R
tested <- fitted.test(coef=c(1))
```

It's also possible to test multiple coefficients at once, or a contrast of coefficients, or multiple contrasts.

The result contains:





