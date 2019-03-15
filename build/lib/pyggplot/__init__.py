# Copyright (c) 2009-2015, Florian Finkernagel. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.

# * Neither the name of the Andrew Straw nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""A wrapper around ggplot2 ( http://had.co.nz/ggplot2/ ) and 
plotnine (https://plotnine.readthedocs.io/en/stable/index.html )

Takes a pandas.DataFrame object, then add layers with the various add_xyz
functions (e.g. add_scatter).

Referr to the ggplot/plotnine documentation about the layers (geoms), and simply
replace geom_* with add_*.
See http://docs.ggplot2.org/0.9.3.1/index.html or 
https://plotnine.readthedocs.io/en/stable/index.html<Paste>

You do not need to seperate aesthetics from values - the wrapper
will treat a parameter as value if and only if it is not a column name.
(so y = 0 is a value, color = 'blue 'is a value - except if you have a column
'blue', then it's a column!. And y = 'value' doesn't work, but that seems to be a ggplot issue).

When the DataFrame is passed to the plotting library:
    - row indices are truned into columns with 'reset_index'
    - multi level column indices are flattend by concatenating them with ' '
        -> (X, 'mean) becomes 'x mean'

R Error messages are not great - most of them translate to 'one or more columns were not found',
but they can appear as a lot of different actual messages such as
    - argument "env" is missing, with no defalut
    - object 'y' not found
    - object 'dat_0' not found
    - requires the follewing missing aesthetics: x
    - non numeric argument to binary operator
without actually quite pointing at what is strictly the offending value.
Also, the error appears when rendering (or printing in ipython notebook),
not when adding the layer.
"""
#from .plot_r import Plot, plot_heatmap, multiplot, MultiPagePlot, convert_dataframe_to_r
from .base import _PlotBase
#from . import plot_nine
from .plot_nine import Plot, Expression, Scalar

all = [Plot, Expression, Scalar]
