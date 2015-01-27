
Assessing the quality of a fit
===

Fitnoise provides various statistics that will help you decide whether your noise model was a good fit to your data.

These are reported in the `description` of the fit.

The "noise combined p value", if all is going well, is a random number between zero and one. A very small value is cause for concern, it indicates a poor fit to the data.

If using a model based on the normal distribution and you obtain a small "noise combined p value", this is good reason to switch to the corresponding model based on the t distribution.

Generally models based on the t distribution will be able to avoid a small noise combined p-value. When something goes wrong with these models, the "prior df" becomes small. Smaller than around 5 is cause for concern. When the "prior df" is small, you will also obtain few genes declared significantly differentially expressed. In this sense, the models based on the t distribution are safe, their failure mode is to not find significant differential expression.

If you obtain a small "prior df", you may have samples of variable quality, or there may be unwanted variation such as batch effects. Possible solutions are to exclude low quality samples, include a coefficient for a known batch effect, or to use a more advanced noise model.