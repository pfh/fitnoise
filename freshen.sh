#!/bin/sh
set -u -e

V=$1

#===================================================================================
set +u
. $V/bin/activate
set -u
#===================================================================================

cd `dirname $0`

python setup.py install     

R CMD INSTALL fitnoise

