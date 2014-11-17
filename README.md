Fitnoise
===

Fitnoise is a Python 2 library for statistical analysis of RNA-Seq data.

A lightweight R wrapper is provided to allow access from R.

Fitnoise uses the Theano deep-learning library for speed.


This iteration of Fitnoise is not yet complete.



Installing
---

Installing dependencies:

    pip install theano
    pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

    R
    install.packages("rPython")
    source("http://bioconductor.org/biocLite.R")
    biocLite("limma")

Installing Fitnoise from source:
    
    python setup.py install     
    R CMD INSTALL fitnoise


Equipping a virtualenv
---

The following creates a virtualenv for both Python and R:

    virtualenv --clear --system-site-packages venv
    mkdir venv/R

    echo 'export R_LIBS=$VIRTUAL_ENV/R' >>venv/bin/activate    
    echo 'import os;os.environ["R_LIBS"]="'`pwd`'/venv/R"' >venv/lib/python*/sitecustomize.py

    echo ". `pwd`/venv/bin/activate && `which R` \$@" >venv/bin/R
    echo ". `pwd`/venv/bin/activate && `which Rscript` \$@" >venv/bin/Rscript
    chmod a+x venv/bin/R venv/bin/Rscript

To install Fitnoise type

    . venv/bin/activate
    
then follow the installation steps above.

If you don't want to use the "activate" script, you can use venv/bin/python and venv/bin/R.

