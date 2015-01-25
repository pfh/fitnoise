Fitnoise
===

Fitnoise is a Python 2 library for statistical analysis of RNA-Seq, PAT-Seq, and microarray data using linear models.

An R wrapper is provided to allow access from R.

Fitnoise uses the Theano deep-learning library for speed.

Fitnoise is developed by Dr. Paul Harrison for the [RNA Systems Biology Laboratory](rnasystems.erc.monash.edu), Monash University.

This iteration of Fitnoise is not yet complete.

Download:

* [Download latest or older versions](https://github.com/pfh/fitnoise/releases)

Links:

* [Fitnoise on Github](https://github.com/pfh/fitnoise)
* [Fitnoise on PyPI](https://pypi.python.org/pypi/fitnoise/)


Installing into a virtualenv from source
---

This is the easiest way to try out Fitnoise.

The following creates a virtualenv in directory venv for both Python and R:

    ./make_virtualenv.sh venv

To freshen the virtualenv after pulling a new version of Fitnoise from github or hacking on the code:

    ./freshen.sh venv


Installing globally
---

Installing dependencies:

    apt-get install python-pip python-numpy python-scipy r-base
    # (or whichever package manager is appropriate to your Linux distribution)
    # (MacOS users can use brew and Anaconda Python)

    pip install --upgrade git+git://github.com/Theano/Theano.git

    R
      install.packages(c("rPython", "jsonlite"))
      source("http://bioconductor.org/biocLite.R")
      biocLite("limma")

To install Fitnoise with pip:

    pip install fitnoise
    
    python -m fitnoise
    # This prints out instructions to install the R component

Alternatively, to install Fitnoise from source:
    
    python setup.py install     
    R CMD INSTALL fitnoise




References
---

Fitnoise copies many features of Limma:

http://bioinf.wehi.edu.au/limma/

The design of Fitnoise has been influenced by RUV-4, although conceived in different terms. See Berkley statistical department technical report 820:

http://statistics.berkeley.edu/tech-reports/820






