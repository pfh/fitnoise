
Control genes
===

Control genes may be specified. These are assumed to have the same expression level in each experimental group, and to have typical variation. Specifying control genes allows the noise model to be estimated more accurately. Control genes are absolutely needed for "per sample" noise model where the group size is two, and for the "factors" noise model.

Note that spike-in controls are *not* appropriate to use as controls, since we require control genes to have typical variation.

Control genes are specified as a boolean vector.

<table>
<tr><th>Python</th><th>R</th></tr>
<tr><td><pre>
fitted = model.fit(
    data, design, 
    controls=controls)
</pre></td><td><pre>
fitted &lt;- fitnoise.fit(
    data, design, model=" ... ", 
    controls=controls)
</pre></td></tr></table>


Automatic selection of control genes
===

You may not know which genes can be used as control genes, but be fairly certain that most genes could be used as such. In this case, you might do something like:

* run the differential test without control genes
* pick 50% of genes with largest p-value as control genes
* run the differential test again with these control genes

I don't have any experience with how well this works in practice.


What to do if you don't have replicates
===

Don't do this.

Ok, you've been handed some data without replicates and have to make the best of it. Here's what I suggest:

* Set *every* gene as a control gene.
* Use a "normal" [noise model](models.md).
* Use the resultant list of differentially expressed genes only as a guide for further lab work.
