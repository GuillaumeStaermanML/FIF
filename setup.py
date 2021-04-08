import sys
import os
import numpy
from Cython.Distutils import build_ext
try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
prjdir = os.path.dirname(__file__)


def read(filename):
    return open(os.path.join(prjdir, filename)).read()




extra_link_args = []
libraries = []
library_dirs = []
include_dirs = []
exec(open('version.py').read())
setup(
    name='pyt-fif',
    version=__version__,
    author='Guillaume Staerman',
    author_email='guillaume.staerman@telecom-paris.fr',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("fif",
                 sources=["_fif.pyx", "fif.cxx"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=['-std=c++11', '-Wcpp'],
                 language="c++")],
    scripts=[],
    py_modules=['version'],
    packages=[],
    license='License.txt',
    include_package_data=True,
    description='Functional Isolation Forest',
    long_description_content_type='text/markdown',
    url='https://github.com/GuillaumeStaermanML/FIF',
    download_url='https://github.com/GuillaumeStaermanML/FIF/archive/refs/tags/1.0.2.tar.gz',
    install_requires=["numpy", "cython"],
)
