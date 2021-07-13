#python3 setup.py build_ext --inplace
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("inaccel_track",
              sources=["inaccel_track.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["m"]  # Unix-like specific
              )
]

setup(name="inaccel_track",
      ext_modules=cythonize(ext_modules))
