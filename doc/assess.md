
Assessing the quality of a fit
===

Fitnoise provides various statistics that will help you decide whether your noise model was a good fit to your data.

These are reported in the `description` of the fit.

The "noise combined p value", if all is going well, is a random number between zero and one. As the Bonferroni correction is quite conservative, it will often be 1, however smaller values are also fine. A very small value is cause for concern, it indicates a poor fit to the data.

If using a model based on the normal distribution and you obtain a small "noise combined p value", this is good reason to switch to the corresponding model based on the t distribution.

Generally models based on the t distribution will be able to avoid a small noise combined p value. When something goes wrong with these models, the "df" (degrees of freedom) becomes small. Smaller than around 5 is cause for concern. When the "df" is small, you will also obtain few genes declared significantly differentially expressed. In this sense, the models based on the t distribution are safer, their failure mode is to not find significant differential expression.

If you obtain a small "df", you may have samples of variable quality, or there may be unwanted variation such as batch effects. Possible solutions are to exclude low quality samples, exclude lowly expressed genes, include a coefficient for a known batch effect, or to use a more advanced [noise model](models.md). For example you could use a noise model with per-sample variances, or a noise model that includes random effects identified using control genes.

Choosing between noise models
---

When trying to choose between noise models, or decide on the exact form a noise model should take for some new kind of data, the model with the smallest "noise fit score" is preferable. The smaller the score, the less surprising the data is given that model.

You could assess whether this score differs significantly between models by bootstrapping.