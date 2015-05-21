pyggplot
========

pyggplot is a Pythonic wrapper around the [R ggplot2 library](http://had.co.nz/ggplot2/).

It is based on a a straightforward *take [Pandas](http://pandas.pydata.org/) data frames and shove them into [R](http://www.r-project.org/) via [rpy2](https://pypi.python.org/pypi/rpy2)* approach.

## Examples
Please visit http://nbviewer.ipython.org/url/tyberiusprime.github.io/pyggplot/pyggplot%20samples.ipynb

## Installation

The easiest installation is via [PyPI](https://pypi.python.org/pypi).

    $ pip install pyggplot

You may be required to update `pandas`, `rpy2`, so you may be required to run

    $ pip install --upgrade pyggplot 

## Usage

    import pandas as pd
    import numpy as np
    import ggplot

    df = pd.DataFrame({'x': np.random.rand(100),
                       'y': np.random.randn(100),
                       'group': ['A','B'] * 50})

    p = pyggplot.Plot(df)
    p.add_scatter('x','y', color='group')
    p.render('output.png')
    ## or if you want to use it in IPython Notebook
    # p.render_notebook()



## Further usage

Takes a `pandas.DataFrame` object, then add layers with the various `add_xyz`
functions (e.g. `add_scatter`).

Refer to the ggplot documentation about the layers (geoms), and simply
replace `geom_*` with `add_*`.
See: http://docs.ggplot2.org/0.9.3.1/index.html

You do not need to separate aesthetics from values - the wrapper
will treat a parameter as value if and only if it is not a column name.
(so `y = 0` is a value, `color = 'blue'` is a value - except if you have a column `'blue'`, then it is a column!.
And `y = 'value'` does not work, but that seems to be a ggplot issue).

When the DataFrame is passed to R:

* row indices are turned into columns with 'reset_index',
* multi level column indices are flattened by concatenating them with `' '`, that is `(X, 'mean')` becomes `'x mean'`.

Error messages are not great - most of them translate to 'one or more columns were not found',
but they can appear as a lot of different actual messages such as

* argument "env" is missing, with no default
* object 'y' not found
* object 'dat_0' not found
* requires the following missing aesthetics: x
* non numeric argument to binary operator

without actually quite pointing at what is strictly the offending value.
Also, the error appears when rendering (or printing in the [IPython Notebook](http://ipython.org/notebook.html)),
not when adding the layer.

## Open questions

* the stat support is not great - it doesn't easily map into pythonic objects. For now, do your stats in pandas - more powerful anyhow! 
* how could error messages be improved?



## Other ggplots' for python

* http://ggplot.yhathq.com/ is a port of ggplot2 for python based on matplotlib - unfortunatly not yet feature complete as of early 2015.
* https://github.com/sirrice/pyplot is another wrapper for ggplot closer to R's syntax, and does not rely on rpy2 - calls command line R.


