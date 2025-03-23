"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')
# all files in data/ and subfolders
data_files = ['config/*'] + [str(p.relative_to(*p.parts[:1])) for p in pathlib.Path(here / 'gym_covid/data').rglob('*/')]

setup(
    name='gym-covid',
    version='0.1.0',
    description='A gym environment for a compartment model modelling the first COVID wave in Belgium.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Mathieu Reymond, Conor F. Hayes, Pieter Libin',
    author_email='mathieu.reymond@vub.be',
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={'gym_covid': 'gym_covid'},
    packages=find_packages(where='./'),
    python_requires='>=3.7, <4',

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={
        'gym_covid': data_files
    },
    install_requires=[
        'gym',
        'pandas',
        'importlib_resources',
        'scipy',
        'matplotlib',
        'numba'
    ]
)
