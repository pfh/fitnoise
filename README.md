Fitnoise
===

Fitnoise is a Python 2 library for statistical analysis of RNA-Seq data.

A lightweight R wrapper is provided to allow access from R.

Fitnoise uses the Theano deep-learning library for speed.


This iteration of Fitnoise is not yet complete.


Installing in a virtualenv
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
    # (MacOS users perhaps use brew and Anaconda Python)

    pip install --upgrade git+git://github.com/Theano/Theano.git

    R
    install.packages("rPython")
    source("http://bioconductor.org/biocLite.R")
    biocLite("limma")

Installing Fitnoise from source:
    
    python setup.py install     
    R CMD INSTALL fitnoise


