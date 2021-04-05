
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
# !cd tools/ && python setup_opera_distance_metric.py build_ext --inplace

import numpy as np
ext_modules = [
    Extension(
        "opera_distance_metric",
        ["opera_distance_metric.pyx"],
        extra_compile_args=['-fopenmp', '-O3'], #'-g'],
        extra_link_args=['-fopenmp'],# '-g'],
        include_dirs=[np.get_include()],
        language="c++"
    )
]

setup(
    name='opera_distance_metric',
    ext_modules=cythonize(ext_modules, annotate=True),
    # gdb_debug=True
)
