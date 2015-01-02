#!/bin/bash

set -euo pipefail

sudo -E `which python-lab` setup.py install

sudo -E R CMD INSTALL fitnoise --library=/data/software/R