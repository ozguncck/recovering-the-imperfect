#setup.py

#setup(ext_modules = cythonize('fit_rpn_match.pyx'))


import os
from setuptools import setup, find_packages

include_dirs = []
ext_modules = []

from Cython.Build import cythonize
import numpy as np
include_dirs = [np.get_include()]

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

pyxFile = os.path.join("mrcnn", "fit_rpn_match.pyx")
ext_modules = cythonize(pyxFile)

config = {
    'install_requires': ['numpy', 'matplotlib', 'pillow'],
    'packages': find_packages(),
    'ext_modules': ext_modules,
    'include_dirs': include_dirs
}
setup(**config)
