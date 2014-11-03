Fitnoise
===

Fitnoise comprises Python 2 and R+ packages for statistical analysis of RNA-Seq data.


Installing
---

Installing dependencies:

    pip install theano
    pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

    R CMD INSTALL rPython

Installing Fitnoise:
    
    python setup.py install     
    R CMD INSTALL fitnoise


Equipping a virtualenv
---

This creates a virtualenv supporting both Python and R.

    virtualenv --clear --system-site-packages venv
    mkdir venv/R
    echo 'export R_LIBS=$VIRTUAL_ENV/R' >>venv/bin/activate    
    echo ". `pwd`/venv/bin/activate && `which R` "'$@' >venv/bin/R
    chmod a+x venv/bin/R

To install type

    . venv/bin/activate
    
then follow the installation steps above.

If you don't want to use "activate", you can use venv/bin/python and venv/bin/R.

