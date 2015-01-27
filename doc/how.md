
How to use Fitnoise
===

This is a basic run-through of how to use Fitnoise.

You can use Fitnoise either from R+ or Python.


Loading
---

R:

```R
library(fitnoise)
```

Python:

```python
import fitnoise
```

Choose a noise model
---

Fitnoise provides a variety of "noise models". fitnoise.Model_t behaves very similarly to limma.

Python:

```python
model = fitnoise.Model_t()
```

R: We will pass an appropriate model selection in the next step.


Fitting noise and estimating coefficients
---

Python:

```python
fit = model.fit_noise(data, design).fit_coef()
```

R:
```R
fit <- fitnoise.fit(data, design, model="Model_t()")
```

* data is our matrix of expression levels. Columns are samples and rows are genes.

* design is our design matrix. Columns are coefficients and rows are samples.

For example if we had two samples in two groups, an appropriate design matrix might be:

Python:

```python
design = [[1,0], [1,0], [1,1], [1,1]]
```

R:

```R
design <- rbind(c(1,0),c(1,0),c(1,1),c(1,1))
```


Testing if a coefficient is non-zero
---


Quality of the result
---


