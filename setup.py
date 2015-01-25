
import os
from os.path import dirname, join

from setuptools import setup

# Get version
directory = dirname(__file__)
with open(join(directory,'fitnoise','__init__.py'),'rU') as f:
    exec f.readline()


setup(
    name="fitnoise",
    version=VERSION,
    description="Statistical analysis of RNA-Seq, PAT-Seq, and microarray data using linear models.",
    author="Paul Harrison",    
    author_email="paul.harrison@monash.edu",
    url="https://github.com/pfh/fitnoise",
    packages=["fitnoise"],
    package_data = {
        "fitnoise" : [
            "DESCRIPTION",
            "NAMESPACE",
            "R/*",
            ],
        },
    )


