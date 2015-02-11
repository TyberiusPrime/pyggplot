try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name='pyggplot',
    version='3',
    packages=['pyggplot',],
    license='BSD',
    #url='http://code.google.com/p/pydataframe/',
    author='Florian Finkernagel',
    description = "A pythonic wrapper around R's ggplot",
    author_email='finkernagel@coonabibba.de',
    long_description=open('README.txt').read(),
    install_requires=[
        'pandas>=0.15',
        'rpy2',
        'ordereddict',
        ]
)
