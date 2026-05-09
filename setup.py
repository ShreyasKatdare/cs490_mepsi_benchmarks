import os
import numpy

from setuptools import find_packages, setup, Extension

# Package meta-data.
NAME = "MEPSI"
DESCRIPTION = "MEPSI: An MDL-based Ensemble Pruning Approach"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.0.1"

# Detect Cython
try:
    import Cython

    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= "0.24"
except ImportError:
    _CYTHON_INSTALLED = False

if not _CYTHON_INSTALLED:
    print("Required Cython version >= 0.24 is not detected!")
    print('Please run "pip install --upgrade cython" first.')
    exit(-1)


REQUIRED = ["numpy>=1.16.0,<1.20.0", "scikit-learn>=0.23,<0.24", "tqdm", "joblib"]

libraries = []
if os.name == "posix":
    libraries.append("m")

from Cython.Build import cythonize

extensions = [
    Extension(
        "mepsi.forest.tree._libs._tree",
        ["mepsi/forest/tree/_libs/_tree.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "mepsi.forest.tree._libs._splitter",
        ["mepsi/forest/tree/_libs/_splitter.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "mepsi.forest.tree._libs._criterion",
        ["mepsi/forest/tree/_libs/_criterion.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "mepsi.forest.tree._libs._utils",
        ["mepsi/forest/tree/_libs/_utils.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        "mepsi.metric.tree_edit._libs.tree_edit",
        ["mepsi/metric/tree_edit/_libs/tree_edit.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3"],
    ),
    Extension(
        "mepsi.pruning._libs.mepsi",
        ["mepsi/pruning/_libs/mepsi.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3"],
    ),
    Extension(
        "mepsi.pruning._libs.kappa_pruning",
        ["mepsi/pruning/_libs/kappa_pruning.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3"],
    ),
]

if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        packages=find_packages(),
        include_package_data=True,
        ext_modules=cythonize(extensions),
        description=DESCRIPTION,
        python_requires=REQUIRES_PYTHON,
        install_requires=REQUIRED,
        setup_requires=["cython", "numpy"],
    )
