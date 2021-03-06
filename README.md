Fitnoise
===

Fitnoise is a Python 2 library for statistical analysis of RNA-Seq, PAT-Seq, and microarray data using linear models.

An R wrapper is provided to allow access from R.

Fitnoise uses the Theano deep-learning library for speed.

Fitnoise is developed by Dr. Paul Harrison for the [RNA Systems Biology Laboratory](http://rnasystems.erc.monash.edu), Monash University.

Overview:

* [Poster presented at Lorne Genome Conference 2015](doc/fitnoise-poster-lorne-2015.pdf)

  (An [earlier poster presented at ABiC 2014](http://f1000.com/posters/browse/summary/1097121) describes the previous R based Fitnoise.)

Documentation:

* [What is Fitnoise?](doc/what.md)
* [How to use Fitnoise](doc/how.md)
* [Assessing the quality of a fit](doc/assess.md)
* [Noise models available](doc/models.md)
* [Control genes](doc/controls.md) (and what to do if you don't have replicates)

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

On the python side, Fitnoise requires Theano. On the R side, Fitnoise requires rPython, and jsonlite, and limma.

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

    pip install --upgrade fitnoise

    python -m fitnoise
    # This prints out instructions to install the R component

Alternatively, to install Fitnoise from source:

    python setup.py install
    R CMD INSTALL fitnoise

Alternatively, to install the development version of Fitnoise directly from github:

    pip install --upgrade 'git+https://github.com/pfh/fitnoise.git#egg=fitnoise'

    R
      install.packages("devtools")
      devtools::install_github("pfh/fitnoise", subdir="fitnoise")


References
---

Fitnoise re-implements various features of the limma Bioconductor package:

http://bioinf.wehi.edu.au/limma/



