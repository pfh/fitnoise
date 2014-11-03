
import os
from os.path import abspath, split, join

from setuptools import setup

os.chdir(abspath(split(__file__)[0]))

setup(
    name="fitnoise",
    author="Paul Harrison",    
    packages=["fitnoise"],
    )


