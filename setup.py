try:
    from setuptools import setup, find_packages
except:
    from distutils.core import setup
import os

long_description = "Please visit https://github.com/TyberiusPrime/pyggplot for full description"
if os.path.exists('README.md'):
    with open('README.md') as op:
        long_description = op.read()

setup(
    name='pyggplot',
    version='27',
    packages=['pyggplot',],
    license='BSD',
    url='https://github.com/TyberiusPrime/pyggplot',
    author='Florian Finkernagel',
    description = "A Pythonic wrapper around R's ggplot",
    author_email='finkernagel@coonabibba.de',
    long_description=long_description,
    package_data={'pyggplot': ['LICENSE.txt', 'README.md']},
    include_package_data=True,    
    install_requires=[
        'pandas>=0.15',
        'plotnine',
        'ordereddict',
        ],
    classifiers=['Development Status :: 3 - Alpha',
         'Intended Audience :: Science/Research',
         'Topic :: Scientific/Engineering',
         'Topic :: Scientific/Engineering :: Visualization',
         'Operating System :: Microsoft :: Windows',
         'Operating System :: Unix',
         'Operating System :: MacOS',
         'Programming Language :: Python',
         'Programming Language :: Python :: 2',
         'Programming Language :: Python :: 2.7',
         'Programming Language :: Python :: 3',
         'Programming Language :: Python :: 3.4'],
)
