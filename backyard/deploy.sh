
set -u -e

sudo python setup.py install
sudo chown -R `whoami`:`whoami` build

sudo -H R CMD INSTALL fitnoise


echo Deployed