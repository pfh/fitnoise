#!/bin/sh
set -u -e

V=$1

virtualenv --clear --system-site-packages $V
[ -d $V/R ] || mkdir $V/R

echo 'export R_LIBS=$VIRTUAL_ENV/R' >>$V/bin/activate

for LIB in $V/lib/python*
do
    echo 'import os,sys; os.environ["R_LIBS"]=sys.prefix+"/R"' >$LIB/sitecustomize.py
done

echo '. `dirname $0`/activate && '`which R`' $@' >$V/bin/R
echo '. `dirname $0`/activate && '`which Rscript`' $@' >$V/bin/Rscript
chmod a+x $V/bin/R $V/bin/Rscript


#===================================================================================
set +u
. $V/bin/activate
set -u
#===================================================================================


pip install --upgrade -r `dirname $0`/requirements.txt

Rscript - <<END
    install.packages(c("rPython","jsonlite"), repos="http://cran.us.r-project.org")
    source("http://bioconductor.org/biocLite.R")
    biocLite("limma")
END

`dirname $0`/freshen.sh $V