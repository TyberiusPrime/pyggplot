## Copyright (c) 2009-2015, Florian Finkernagel. All rights reserved.

## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:

##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.

##     * Redistributions in binary form must reproduce the above
##       copyright notice, this list of conditions and the following
##       disclaimer in the documentation and/or other materials provided
##       with the distribution.

##     * Neither the name of the Andrew Straw nor the names of its
##       contributors may be used to endorse or promote products derived
##       from this software without specific prior written permission.

## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""A wrapper around ggplot2 ( http://had.co.nz/ggplot2/ )

Takes a pandas.DataFrame object, then add layers with the various add_xyz
functions (e.g. add_scatter).

Referr to the ggplot documentation about the layers (geoms), and simply
replace geom_* with add_*.
See http://docs.ggplot2.org/0.9.3.1/index.html

You do not need to seperate aesthetics from values - the wrapper
will treat a parameter as value if and only if it is not a column name.
(so y = 0 is a value, color = 'blue 'is a value - except if you have a column
'blue', then it's a column!. And y = 'value' doesn't work, but that seems to be a ggplot issue).

When the DataFrame is passed to R:
    - row indices are truned into columns with 'reset_index'
    - multi level column indices are flattend by concatenating them with ' ' 
        -> (X, 'mean) becomes 'x mean'

Error messages are not great - most of them translate to 'one or more columns were not found',
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

try:
    import exptools
    exptools.load_software('ggplot2')
    import ggplot2
    ggplot2.load_r()
except ImportError:
    pass
import itertools
from ordereddict import OrderedDict
import numpy
import math
import pandas

_r_loaded = False


def load_r():
    """Lazy R loader"""
    global _r_loaded
    if not _r_loaded:
        global NA
        global robjects
        import rpy2.robjects as robjects
        import rpy2.rinterface as rinterface
        NA = robjects.r("NA")
        robjects.r('library(grid)')
        robjects.r('library(ggplot2)')
        robjects.r('library(scales)')
        #apperantly, as_df is missing in some downloaded versions of plyr
        robjects.r("""as_df = function (output)
{
    if (length(output) == 0)
        return(data.frame())
    df <- data.frame(matrix(ncol = 0, nrow = length(output[[1]])))
    for (var in names(output)) {
        df[var] <- output[var]
    }
    df
}
""")
        if robjects.r("exists('Trans')")[0]:  # pre ggplot 0.9 style
            robjects.r("""
TransInvNegLog10 <- Trans$new("InvNegLog10", f = function(x) 10^(-x),
inverse = function(x) -log10(x), labels = function(x) x)
TransInvNegLog10b <- Trans$new("InvNegLog10b",
            f = function(x) -log10(x),
            inverse = function(x) 10^-x,
            labels = function(x) bquote(10^.(-x)))

""")
        else:  # post ggplot 0.9 style
            robjects.r(""" 
TransInvNegLog10 <- scales::trans_new(name="InvNegLog10", 
                transform = function(x) 10^(-x), 
                inverse = function(x) -log10(x), 
                format = function(x) x)
TransInvNegLog10b <- scales::trans_new(name="InvNegLog10b",
            transform = function(x) -log10(x),
            inverse = function(x) 10^-x,
            format = function(x) bquote(10^.(-x)))

""")


def r_expression(expr):
    return robjects.r('expression(%s)' % expr)



class Plot:

    def __init__(self, dataframe, *ignored):
        """Create a new ggplot2 object from DataFrame"""
        load_r()
        self.r = {}
        self.r['ggplot'] = robjects.r['ggplot']
        self.r['aes'] = robjects.r['aes']
        if robjects.r("exists('ggplot2:::\"+.ggplot\"')")[0]:
            self.r['add'] = robjects.r('ggplot2:::"+.ggplot"')
        else:
            self.r['add'] = robjects.r('ggplot2::"%+%"')

        self.r['layer'] = robjects.r['layer']
        self.r['facet_wrap'] = robjects.r['facet_wrap']
        self.r['geom_text'] = robjects.r['geom_text']
        self.r['ggsave'] = robjects.r['ggsave']
        self.old_names = []
        self.lab_rename = {}
        self.dataframe = self._prep_dataframe(dataframe)
        self._other_adds = []
        self.to_rename = {}
        self._add_geom_methods()

    def render(self, output_filename, width=8, height=6, dpi=300):
        """Save the plot to a file"""
        plot = self.r['ggplot'](convert_dataframe_to_r(self.dataframe))
        for obj in self._other_adds:
            plot = self.r['add'](plot, obj)
        for name, value in self.lab_rename.items():
            plot = self.r['add'](
                    plot, robjects.r('labs(%s = "%s")' % (name, value)))
        output_filename = output_filename.replace('%', '%%')  # R tries some kind of integer substitution on these, so we need to double the %
        self.r['ggsave'](filename=output_filename, plot=plot, width=width, height=height, dpi=dpi)

    def render_notebook(self, width=800, height=600):
        """Show the plot in the ipython notebook (ie. return svg formated image data)"""
        import tempfile
        from rpy2.robjects.packages import importr 
        from IPython.core.display import Image
        grdevices = importr('grDevices')
        tf = tempfile.NamedTemporaryFile(suffix='.png')
        fn = tf.name
        grdevices.png(fn, width = width, height = height)
        plot = self.r['ggplot'](convert_dataframe_to_r(self.dataframe))
        for obj in self._other_adds:
            plot = self.r['add'](plot, obj)
        for name, value in self.lab_rename.items():
            plot = self.r['add'](
                    plot, robjects.r('labs(%s = "%s")' % (name, value)))
        robjects.r('plot')(plot)
            
        grdevices.dev_off()
        tf.seek(0,0)
        result = tf.read()
        tf.close()
        return result

    def _repr_png_(self):
        """Ipython notebook representation callback"""
        return self.render_notebook()

    def _prep_dataframe(self, df):
        """prepare the dataframe by renaming all the columns
        (we use this to get around R naming issues - the axis get labled correctly later on)"""
        if 'pydataframe.dataframe.DataFrame' in str(type(df)):
            df = self._convert_pydataframe(df)
        if isinstance(df.columns, pandas.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        #df = dataframe.copy()
        new_names = []
        for name in df.columns:
            if not name in self.old_names:
                new_names.append(name)
        self.old_names.extend(new_names)
        rename = dict([(name, 'dat_%s' % self.old_names.index(name)) for name in df.columns])
        df = df.rename(columns = rename)
        return df

    def _convert_pydataframe(self, pdf):
        """Compability shim for still being able to use old pydataframes with the new pandas interface"""
        d = {}
        for column in pdf.columns_ordered:
            o = pdf.gcv(column)
            if 'pydataframe.factors.Factor' in str(type(o)):
                d[column] = pandas.Series(pandas.Categorical(o.as_levels(), categories = o.levels))
            else:
                d[column] = o
        return pandas.DataFrame(d)

        #if isinstance(o, Factor):
    def _translate_params(self, params):
        """Translate between the original dataframe names and the numbered ones we assign
        to avoid r-parsing issues"""
        aes_params = []
        for aes_name, aes_column in params.items():
            if aes_column in self.old_names:
                new_name = 'dat_%s' % self.old_names.index(aes_column)
                aes_params.append('%s=%s' % (aes_name, new_name))
                if aes_column in self.to_rename:
                    self._fix_axis_label(aes_name, new_name, self.to_rename[aes_column])
                else:
                    self._fix_axis_label(aes_name, new_name, aes_column)
            else:  # a fixeud value
                aes_params.append("%s=%s" % (aes_name, aes_column))
        return aes_params

    def _fix_axis_label(self, aes_name, new_name, real_name):
        """Reapply the correct (or new) labels to the axis, overwriting our dat_%i numbered dataframe
        columns"""
        which_legend = False
        if aes_name == 'x':
            which_legend = 'x'
        elif aes_name == 'y':
            which_legend = 'y'
        elif aes_name == 'color' or aes_name == 'colour':
            which_legend = 'colour'
        elif aes_name == 'fill':
            which_legend = 'fill'
        elif aes_name == 'shape':
            which_legend = 'shape'
        elif aes_name == 'size':
            which_legend = 'size'
        if which_legend:
            self.lab_rename[which_legend] = real_name

    def _build_aesthetic(self, params):
        """Transform a python list of aesthetics to the R aes() object"""
        aes_params = self._translate_params(params)
        aes_params = ", ".join(aes_params)
        return robjects.r('aes(%s)' % aes_params)

    def parse_param(self, name, value, required=True):
        """
        Transform parameters into either aes_params or other_params,
        depending on whether they are in our df.
        if value is None, this parameter is ignored

        """
        if not value is None:
            if isinstance(value, tuple):  # this  allows renaming columns when plotting - why is this here? Is this actually usefully
                new_name = value[1]
                value = value[0]
                self.to_rename[value] = new_name
            if value in self.old_names:
                self.aes_collection[name] = value
            else:
                if value == '..level..':
                    self.aes_collection[name] = '..level..'#robjects.r('expression(..level..)')
                else:
                    self.other_collection[name] = value

    def reset_params(self, data):
        """Prepare the dictionaries used by parse_param"""
        self.aes_collection = {}
        self.other_collection = {}
        if not data is None:
            self.other_collection['data'] = convert_dataframe_to_r(self._prep_dataframe(data))

    def _add(self, name, geom_name, required_mappings, optional_mappings, defaults, args, kwargs):
        """The generic method to add a geom to the ggplot.
        You need to call add_xyz (see _add_geom_methods for a list, with each variable mapping
        being one argument) with the respectivly required parameters (see ggplot documentation).
        You may optionally pass in an argument called data, which will replace the plot-global dataframe
        for this particular geom
        """
        mappings = {}
        all_defined_mappings = required_mappings + optional_mappings
        for a, b in zip(all_defined_mappings, args):  # so that you could in theory also pass the optional_mappings by position...required_mappings
            mappings[a] = b
        mappings.update(kwargs)

        if 'data' in mappings:
            data = mappings['data']
            del mappings['data']
        else:
            data = None
        for mapping in mappings:
            if not mapping in required_mappings and not mapping in optional_mappings:
                raise ValueError("add_%s / %s does not take parameter %s" % (name, geom_name, mapping))
        for mapping in required_mappings:
            if not mapping in mappings:
                raise ValueError("Missing required mapping in add_%s / %s: %s" % (name, geom_name, mapping))
        for mapping in optional_mappings:
            if not mapping in mappings:
                if mapping in defaults:
                    if hasattr(defaults[mapping], '__call__', ):
                        mappings[mapping] = defaults[mapping](mappings)
                    else:
                        mappings[mapping] = defaults[mapping]
                else:
                    mappings[mapping] = None

        self.reset_params(data)
        for param in mappings:
            self.parse_param(param, mappings[param])

        if geom_name.startswith('annotation'):
            self._other_adds.append(robjects.r(geom_name)( **self.other_collection))
        else:
            self._other_adds.append(robjects.r(geom_name)(self._build_aesthetic(self.aes_collection), **self.other_collection))
        return self

    def _add_geom_methods(self):
        """add add_xyz methods for all geoms in ggplot.
        All geoms have required & optional attributes and take an optional data parameter with another
        dataframe
        """
        #python method name (add_ + name), geom (R) name, required attributes, optional attributes, default attribute values
        methods = (
                #geoms

                ('ab_line', 'geom_abline', ['intercept', 'slope'], ['alpha', 'size', 'color', 'linetype'], {}, ''),
                ('area', 'geom_area', ['x', 'y'], ['alpha', 'color', 'fill', 'linetype', 'size', 'position'], {}, ''),
                ('bar', 'geom_bar', ['x', 'y'], ['color', 'group', 'fill', 'position', 'stat', 'order', 'alpha', 'weight', 'width'], {'position': 'dodge', 'stat': 'identity'}, ''),
                ('bin2d', 'geom_bin2d', ['xmin', 'xmax', 'ymin', 'ymax'], ['alpha', 'color', 'fill', 'linetype', 'size', 'weight'], {}, ''),
                ('blank', 'geom_blank', [], [], {}, ''),
                ('box_plot', 'geom_boxplot', ['x', 'y'], ['alpha', 'color', 'fill', 'group', 'linetype', 'shape', 'size', 'weight'], {}, 'a box plot with the default stat (10/25/50/75/90 percentile)'),
                ('box_plot2', 'geom_boxplot', ['x','lower', 'middle','upper','ymin', 'ymax'], ['alpha', 'color', 'fill', 'group', 'linetype', 'shape', 'size', 'weight'], 
                    {'stat': 'identity'}, ' box plot where you define everything manually'),
                ('contour', 'geom_contour', ['x', 'y'], ['alpha',' color', 'linetype', 'size',' weight'], {}, ''),
                ('crossbar', 'geom_crossbar', ['x','y', 'ymin', 'ymax'], ['alpha', 'color', 'fill', 'linetype', 'size'], {}, ''),
                ('density', 'geom_density', ['x', 'y'], ['alpha', 'color', 'fill', 'linetype', 'size',' weight'], 
                    {'bw': lambda mappings: (robjects.r('bw.SJ')(self.dataframe.get_column_view(self.old_names.index(mappings['x']))))
                        }, ''),
                ('density_2d', 'geom_density2d', ['x', 'y'], ['alpha', 'color', 'linetype','fill', 'contour'], {}, ''),
                ('error_bars', 'geom_errorbar', ['x', 'ymin', 'ymax'], ['alpha', 'color', 'group', 'linetype', 'size', 'width'], {'width': 0.25}, ''),
                ('error_barsh', 'geom_errorbarh', ['y', 'xmin', 'xmax'], ['alpha', 'color', 'group', 'linetype', 'size', 'width'], {'width': 0.25}, ''),
                ('freq_poly', 'geom_freq_poly', [], ['alpha', 'color', 'linetype', 'size'], {}, ''),
                ('hex', 'geom_hex', ['x', 'y'], ['alpha', 'color', 'fill', 'size'], {}, ''),
                #  ('histogram', this is it's own function

                ('hline', 'geom_hline', ['yintercept'], ['alpha', 'color', 'linetype', 'size'], {'alpha': 0.5, 'color': 'black', 'size': 1}, ''),
                ('horizontal_line', 'geom_hline', ['yintercept'], ['alpha', 'color', 'linetype', 'size'], {'alpha': 0.5, 'color': 'black', 'size': 1}, 'Renamed hline'), 

                # jitter is it's own function
                ('line', 'geom_line', ['x','y'], ['color', 'group', 'shape', 'alpha', 'size', 'stat', 'fun.y', 'linetype'], {}, ''),
                ('map', 'geom_map', ['map_id'], ['alpha', 'color', 'fill', 'linetype', 'size'], {}, ''),
                ('path', 'geom_path', ['x', 'y'], ['alpha', 'color', 'fill', 'linetype', 'size', 'group'], {}, ''),

                ('point', 'geom_point', ['x','y'], ['color', 'group', 'shape', 'size', 'alpha', 'stat', 'fun.y'], {}, ''),
                ('scatter', 'geom_point', ['x','y'], ['color', 'group', 'shape', 'size', 'alpha', 'stat', 'fun.y'], {}, ''),

                ('pointrange', 'geom_pointrange', ['x', 'y', 'ymin', 'ymax'], ['alpha', 'color',' fill', 'linetype', 'shape', 'size'], {}, ''),
                ('polygon','geom_polygon',  ['x', 'y',], ['alpha', 'color', 'fill', 'linetype', 'size'], {}, ''),
                ('quantile','geom_quantile',  ['x', 'y',], ['alpha', 'color', 'linetype', 'size', 'weight'], {}, ''),
                ('raster', 'geom_raster', ['x', 'y'], ['fill', 'alpha'], {}, ''),
                ('rect', 'geom_rect', ['xmin', 'xmax', 'ymin', 'ymax'], ['alpha', 'color', 'fill', 'linetype', 'size'], {'alpha': 1}, ''),
                ('ribbon', 'geom_ribbon', ['x', 'ymin', 'ymax'], ['alpha', 'color', 'fill', 'linetype', 'size', 'position'], {}, ''),
                ('rug', 'geom_rug', [], ['sides'], {'sides': 'bl'}, ''),
                ('scatter', 'geom_point', ['x','y'], ['color', 'group', 'shape', 'size', 'alpha', 'stat', 'fun.y'], {}, ''),
                ('segment', 'geom_segment', ['x', 'xend', 'y', 'yend'], ['alpha', 'color', 'linetype', 'size'], {'size': 0.5}, ''),
                ('smooth', 'geom_smooth', ['x', 'y'], ['alpha', 'color',' fill', 'linetype', 'size', 'weight', 'method'], {}, ''),
                ('step', 'geom_step', ['x','y'], ['direction', 'stat', 'position', 'alpha', 'color', 'linetype', 'size'], {}, ''),
                ('text', 'geom_text', ['x', 'y', 'label'], ['alpha', 'angle', 'color', 'family', 'fontface', 'hjust', 'lineheight', 'size', 'vjust', 'position'], {'position': 'identity'}, ''),
                ('tile', 'geom_tile', ['x', 'y'], ['alpha', 'color', 'fill', 'size', 'linetype', 'stat'], {}, ''),
                ('violin', 'geom_violin', ['x', 'y'], ['alpha', 'color', 'fill', 'linetype', 'size', 'weight', 'scale', 'stat', 'position', 'trim'], {'stat': 'ydensity'}, ''),

                ('vline', 'geom_vline', ['xintercept'], ['alpha', 'color', 'size', 'linetype'], {'alpha': 0.5, 'color': 'black', 'size': 1}, ''),
                ('vertical_bar', 'geom_vline', ['xintercept'], ['alpha', 'color', 'size', 'linetype'], {'alpha': 0.5, 'color': 'black', 'size': 1}, ''),

                # stats
                ('stat_sum_color', 'stat_sum', ['x', 'y'], ['size'], {'color': '..n..', 'size': 0.5}, ''),
                ('stat_smooth', 'stat_smooth', [], ['method', 'se', 'x', 'y'], {"method": 'lm', 'se': True}, ''),
                ('stat_density_2d', 'stat_density', ['x','y'], ['geom','contour', 'fill'], {}, ''),

                ('stacked_bar_plot', 'geom_bar', ['x', 'y', 'fill'], [], {'position': 'stack'}, ''),  # do we still need this?
                # """A scatter plat that's colored by no of overlapping points"""
                #annotations
                ('annotation_logticks', 'annotation_logticks', [], ['base','sides','scaled', 'short','mid', 'long',],  
                        { 'base' : 10,
                          'sides': "bl",
                          'scaled': True,
                          'short': robjects.r('unit')(0.1, "cm"),
                          'mid': robjects.r('unit')(0.2, "cm"),
                          'long': robjects.r('unit')(0.3, "cm"),
                       }, ''),
                )
        for x in methods:
            if len(x) != 6:
                raise ValueError("Wrong number of arguments: %s" % (x,))

        for (name, geom, required, optional, defaults, doc_str) in methods:
            def define(name, geom, required, optional, defaults):  # we need to capture the variables...
                def do_add(*args, **kwargs):
                    return self._add(name, geom, required, optional, defaults, args, kwargs)
                do_add.__doc__ = doc_str
                return do_add
            setattr(self, 'add_' + name, define(name, geom, required, optional, defaults))

    def add_jitter(self, x,y, color=None, group = None, shape=None, size=None, alpha=None, jitter_x = True, jitter_y = True, fill=None):
        #an api changed necessitates this - jitter_x and jitter_y have been replaced with position_jitter(width, height)...
        aes_params = {'x': x, 'y': y}
        other_params = {}
        position_jitter_params = {}
        if color:
            if color in self.old_names:
                aes_params['colour'] = color
            else:
                other_params['colour'] = color
        if fill:
            if fill in self.old_names:
                aes_params['fill'] = fill
            else:
                other_params['fill'] = fill
        if group:
            aes_params['group'] = group
        if not alpha is None:
            if alpha in self.old_names:
                aes_params['alpha'] = alpha
            else:
                other_params['alpha'] = alpha
        if not shape is None:
            if shape in self.old_names:
                aes_params['shape'] = shape
            else:
                other_params['shape'] = shape
        if size:
            other_params['size'] = size
        if jitter_x is True:
            position_jitter_params['width'] = robjects.r('NULL')
        elif isinstance(jitter_x, float) or isinstance(jitter_x, int):
            position_jitter_params['width'] = jitter_x
        elif jitter_x is False:
            position_jitter_params['width'] = 0
        else:
            raise ValueError("invalid jitter_x value")
        if jitter_y is True:
            position_jitter_params['height'] = robjects.r('NULL')
        elif isinstance(jitter_y, float) or isinstance(jitter_y, int):
            position_jitter_params['height'] = jitter_y
        elif jitter_y is False:
            position_jitter_params['height'] = 0
        else:
            raise ValueError("invalid jitter_y value")
        other_params['position'] = robjects.r('position_jitter')(**position_jitter_params)
        self._other_adds.append(
                robjects.r('geom_jitter')(self._build_aesthetic(aes_params), **other_params)
            )
        return self

    def add_histogram(self, x_column, y_column="..count..", color=None, group=None, fill=None, position="dodge", add_text=False, bin_width=None, alpha=None, size=None):
        aes_params = {'x': x_column}
        other_params = {}
        stat_params = {}
        if fill:
            aes_params['fill'] = fill
        if color:
            if color in self.old_names:
                aes_params['colour'] = color
            else:
                other_params['colour'] = color
        if group:
            aes_params['group'] = group
            #x = x_column, y = y_column)
        if bin_width:
            other_params['binwidth'] = bin_width
        if not alpha is None:
            if alpha in self.old_names:
                aes_params['alpha'] = alpha
            else:
                other_params['alpha'] = alpha
        if size:
            other_params['size'] = size
        if stat_params:
            #other_params.update(stat_params)
            other_params['position'] = position
            #print 'a', other_params
            self._other_adds.append(
                robjects.r('geom_bar')(self._build_aesthetic(aes_params),
                                   **other_params)
            )
        else:
            other_params['position'] = position
            #print 'b', other_params
            print (aes_params)
            self._other_adds.append(
                robjects.r('geom_histogram')(self._build_aesthetic(aes_params), **other_params)
            )
        if add_text:
            self._other_adds.append(
                robjects.r('geom_text')(
                    self._build_aesthetic({'x': x_column, 'y': '..count..', 'label': '..count..'}), stat='bin'))
        return self

    def add_cummulative(self, x_column, ascending=True, percent = False, percentile = 1.0):
        """Add a line showing cumulative % of data <= x.
        if you specify a percentile, all data at the extreme range is dropped
        
        
        """
        total = 0
        current = 0
        try:
            column_name = 'dat_%s' % self.old_names.index(x_column)
        except ValueError:
            raise ValueError("Could not find column %s, available: %s" % (x_column, self.old_names))
        column_data = self.dataframe.get_column(column_name)  # explicit copy!
        column_data = column_data[~numpy.isnan(column_data)]
        column_data = numpy.sort(column_data)
        total = float(len(column_data))
        real_total = total
        if not ascending:
            column_data = column_data[::-1]  # numpy.reverse(column_data)
        if percentile != 1.0:
            if ascending:
                maximum = numpy.max(column_data)
            else:
                maximum =  numpy.min(column_data)
            total = float(total * percentile)
            if total > 0:
                column_data = column_data[:total]
                offset = real_total - total
            else:
                column_data = column_data[total:]
                offset = 2* abs(total)
        else:
            offset = 0
        x_values = []
        y_values = []
        if percent:
            current = 100.0
        else:
            current = total
        for value, group in itertools.groupby(column_data):
            x_values.append(value)
            y_values.append(current + offset)
            if percent:
                current -= (len(list(group)) / total)
            else:
                current -= (len(list(group)))
            #y_values.append(current)
        data = pydataframe.DataFrame({x_column: x_values, ("%" if percent else '#') + ' <=': y_values})
        if percentile > 0:
            self.scale_y_continuous(limits = [0, real_total])
        self.add_line(x_column, ("%" if percent else '#') + ' <=', data=data)
        if percentile != 1.0:
            self.set_title('showing only %.2f percentile, extreme was %.2f' % (percentile, maximum))
        return self

    def add_heatmap(self, x_column, y_column, fill, low="red", mid=None, high="blue", midpoint=0, guide_legend = None, scale_args = None):
        aes_params = {'x': x_column, 'y': y_column}
        aes_params['x'] = x_column
        aes_params['y'] = y_column
        aes_params['fill'] = fill
        self._other_adds.append(
            robjects.r('geom_tile')(self._build_aesthetic(aes_params), stat="identity")
        )
        if scale_args is None:
            scale_args = {}
        if guide_legend:
            scale_args['guide'] = guide_legend
        if mid is None:
            self._other_adds.append(
                    robjects.r('scale_fill_gradient')(low=low, high=high, **scale_args)
                    )
        else:
            self._other_adds.append( robjects.r('scale_fill_gradient2')(low=low, mid=mid, high=high, midpoint=midpoint, **scale_args))
        return self

    def set_title(self, title):
        self._other_adds.append(robjects.r('ggtitle')(title))

    def facet(self, column_one, column_two=None, fixed_x=True, fixed_y=True, ncol=None):
        facet_wrap = robjects.r['facet_wrap']
        if fixed_x and not fixed_y:
            scale = 'free_y'
        elif not fixed_x and fixed_y:
            scale = 'free_x'
        elif not fixed_x and not fixed_y:
            scale = 'free'
        else:
            scale = 'fixed'
        if column_two:
            #params = self._translate_params({column_one: column_two})[0]
            #facet_specification = params.replace('=', '~')
            new_one = 'dat_%s'  % self.old_names.index(column_one)
            new_two = 'dat_%s'  % self.old_names.index(column_two)
            facet_specification = '%s ~ %s' % (new_one, new_two)
            #facet_specification = '%s ~ %s' % (column_one, column_two)
        else:
            params = self._translate_params({"": column_one})[0]
            facet_specification = params.replace('=', '~')
            #facet_specification = '~ %s' % (column_one, )
        params = {
            'scale': scale}
        if ncol:
            params['ncol'] = ncol
        self._other_adds.append(facet_wrap(robjects.r(facet_specification), **params))
        return self

    def facet_grid(self, column_one, column_two=None, fixed_x=True, fixed_y=True, ncol=None):
        facet_wrap = robjects.r['facet_wrap']
        if fixed_x and not fixed_y:
            scale = 'free_y'
        elif not fixed_x and fixed_y:
            scale = 'free_x'
        elif not fixed_x and not fixed_y:
            scale = 'free'
        else:
            scale = 'fixed'
        if column_two:
            new_one = 'dat_%s' % self.old_names.index(column_one)
            new_two = 'dat_%s' % self.old_names.index(column_two)
            facet_specification = '%s ~ %s' % (new_one, new_two)
        else:
            params = self._translate_params({"": column_one})[0]
            facet_specification = '. ' + params.replace('=', '~')
            #facet_specification = '~ %s' % (column_one, )
        params = {
            'scale': scale}
        if ncol:
            params['ncol'] = ncol
        self._other_adds.append(robjects.r('facet_grid')(robjects.r(facet_specification), **params))
        return self

    def greyscale(self):
        self._other_adds.append(robjects.r('scale_colour_grey()'))
        self._other_adds.append(robjects.r('scale_fill_grey()'))

    def theme_bw(self, base_size=None):
        kwargs = {}
        if base_size:
            kwargs['base_size'] = float(base_size)
        self._other_adds.append(robjects.r('theme_bw')(**kwargs))

    def theme_grey(self, base_size=None):
        kwargs = {}
        if base_size:
            kwargs['base_size'] = float(base_size)
        self._other_adds.append(robjects.r('theme_grey')(**kwargs))
    
    def theme_darktalk(self, base_size=None):
        kwargs = {}
        if base_size:
            kwargs['base_size'] = float(base_size)
        robjects.r("""
    theme_darktalk = function (base_size = 28)
{
    structure(
        list(
        axis.line = theme_blank(),
        axis.text.x = theme_text(size = base_size *
            0.8, lineheight = 0.9, colour = "white", vjust = 1),
        axis.text.y = theme_text(size = base_size * 0.8, lineheight = 0.9,
            colour = "white", hjust = 1),
        axis.ticks = theme_segment(colour = "grey40"),
        axis.title.x = theme_text(size = base_size, vjust = 0.5, colour="white"),
        axis.title.y = theme_text(size = base_size, colour="white", angle=90),
        axis.ticks.length = unit(0.15, "cm"),
        axis.ticks.margin = unit(0.1, "cm"),

        legend.background = theme_rect(colour = "black"),
        legend.key = theme_rect(fill = "grey5", colour = "black"),
        legend.key.size = unit(2.2, "lines"),
            legend.text = theme_text(size = base_size * 1, colour="white"),"
            legend.title = theme_text(size = base_size * 1, face = "bold", hjust = 0),
            legend.position = "right",

        panel.background = theme_rect(fill = "black", colour = NA),
        panel.border = theme_blank(),
        panel.grid.major = theme_line(colour = "grey40"),
        panel.grid.minor = theme_line(colour = "grey25", size = 0.25),
        panel.margin = unit(0.25, "lines"),

        strip.background = theme_rect(fill = "grey20", colour = NA),
        strip.label = function(variable, value) value,
        strip.text.x = theme_text(size = base_size * 0.8),
        strip.text.y = theme_text(size = base_size * 0.8, angle = -90),

        plot.background = theme_rect(colour = NA, fill = "black"),
        plot.title = theme_text(size = base_size * 1.2, colour="white"), plot.margin = unit(c(1, 1, 0.5, 0.5), "lines")),

        class = "options")
}

""")
        self._other_adds.append(robjects.r('theme_darktalk')(**kwargs))

    def theme_talk(self, base_size=None):
        kwargs = {}
        if base_size:
            kwargs['base_size'] = float(base_size)
        self._other_adds.append(robjects.r('theme_talk')(**kwargs))

    def set_base_size(self, base_size=10):
        self.theme_grey(base_size=base_size)

    def add_label(self, text, xpos, ypos, size=8, color=None, alpha=None):
        import pydataframe
        data = pydataframe.DataFrame({'x': [xpos], 'y': [ypos], 'text': [text]})
        aes_params = OrderedDict({'x': 'x', 'y': 'y', 'label': 'text'})
        other_params = {'data': data}
        if color:
            other_params['colour'] = color
        if alpha:
            other_params['alpha'] = alpha
        self._other_adds.append(robjects.r('geom_text')(self._build_aesthetic(aes_params), **other_params))
        return self
        self._other_adds.append(
            self.r['geom_text'](
               robjects.r('aes(x=x, y=y, label=text)'),
               data,
                size=size,
                color="black"

            )
        )
        return self

    def scale_x_log_10(self):
        self.scale_x_continuous(trans='log10')
        return self

    def scale_x_continuous(self, breaks=None, minor_breaks=None, trans=None, limits=None, labels=None, expand=None, formatter=None, name=None):
        return self.scale_continuous('x', breaks, minor_breaks, trans, limits, labels, expand, name)

    def scale_y_continuous(self, breaks=None, minor_breaks=None, trans=None, limits=None, labels=None, expand=None, name=None):
        return self.scale_continuous('y', breaks, minor_breaks, trans, limits,  labels, expand, name)

    def scale_continuous(self, scale = 'x', breaks=None, minor_breaks=None, trans=None, limits=None, labels=None, expand=None, name=None):
        other_params = {}
        if not breaks is None:
            if breaks in ('date', 'log', 'pretty', 'trans'):
                other_params['breaks'] = robjects.r('%s_breaks' % breaks)()
                breaks = None
            else:
                other_params['breaks'] = numpy.array(breaks)
        if not minor_breaks is None:
            if minor_breaks == 'waiver()':
                other_params['minor_breaks'] = robjects.r('waiver()')
            else:
                other_params['minor_breaks'] = numpy.array(minor_breaks)
        if trans:
            other_params['trans'] = str(trans)
        if not limits is None:
            other_params['limits'] = numpy.array(limits)
        if not labels is None:
            if labels in ( 'comma', 'dollar', 'percent', 'scientific', 'date', 'parse', 'format', ):
                other_params['labels'] = robjects.r("%s_format" %labels)()
                labels = None
            #elif labels.startswith('math_format') or labels.startswith('trans_format'):
                #other_params['labels'] = robjects.r(labels)
                #labels = None
            elif hasattr(labels, '__iter__'):
                other_params['labels'] = numpy.array(labels)
            else:
                other_params['labels'] = robjects.r(labels)
                labels = None
        if not expand is None:
            other_params['expand'] = numpy.array(expand)
        if not breaks is None and not labels is None:
            if len(breaks) != len(labels):
                raise ValueError("len(breaks) != len(labels)")
        if not name is None:
            other_params['name'] = name

        self._other_adds.append(
            robjects.r('scale_%s_continuous' % scale)(**other_params)
        )
        return self

    def scale_x_discrete(self, breaks=None, minor_breaks=None, trans=None, limits=None, labels=None, expand=None, name = None):
        other_params = {}
        if not breaks is None:
            other_params['breaks'] = numpy.array(breaks)
        if not minor_breaks is None:
            other_params['minor_breaks'] = numpy.array(minor_breaks)
        if trans:
            other_params['trans'] = str(trans)
        if not limits is None:
            other_params['limits'] = numpy.array(limits)
        if not labels is None:
            other_params['labels'] = numpy.array(labels)
        if not expand is None:
            other_params['expand'] = numpy.array(expand)
        if not name is None:
            other_params['name'] = name

        if not breaks is None and not labels is None:
                if len(breaks) != len(labels):
                    raise ValueError("len(breaks) != len(labels)")

        self._other_adds.append(
            robjects.r('scale_x_discrete')(**other_params)
        )
        return self

    def scale_y_discrete(self, breaks=None, minor_breaks=None, trans=None, limits=None, labels=None, expand=None, name = None):
        other_params = {}
        if not breaks is None:
            other_params['breaks'] = numpy.array(breaks)
        if not minor_breaks is None:
            other_params['minor_breaks'] = numpy.array(minor_breaks)
        if trans:
            other_params['trans'] = str(trans)
        if not limits is None:
            other_params['limits'] = numpy.array(limits)
        if not labels is None:
            other_params['labels'] = numpy.array(labels)
        if not expand is None:
            other_params['expand'] = numpy.array(expand)
        if not name is None:
            other_params['name'] = name

        if not breaks is None and not labels is None:
                if len(breaks) != len(labels):
                    raise ValueError("len(breaks) != len(labels)")

        self._other_adds.append(
            robjects.r('scale_y_discrete')(**other_params)
        )
        return self


    def scale_x_reverse(self):
        self._other_adds.append(robjects.r('scale_x_reverse()'))
        return self

    def scale_y_reverse(self):
        self._other_adds.append(robjects.r('scale_y_reverse()'))
        return self

    def turn_x_axis_labels(self, angle=75, hjust=1, size=8, vjust=0, color=None):
        axis_text_x_args = {
                'angle': angle,
                'hjust': hjust,
                'size': size,
                'vjust': vjust,
                'color': color
                }
        for key, value in axis_text_x_args.items():
            if value is None:
                del axis_text_x_args[key]
        kargs = {
            'axis.text.x': robjects.r('theme_text')(**axis_text_x_args)
        }
        self._other_adds.append(robjects.r('opts')(**kargs))
        return self

    def turn_y_axis_labels(self, angle=75, hjust=1, size=8, vjust=0, color=None):
        axis_text_y_args = {
                'angle': angle,
                'hjust': hjust,
                'size': size,
                'vjust': vjust,
                'color': color
                }
        for key, value in axis_text_y_args.items():
            if value is None:
                del axis_text_y_args[key]
        kargs = {
            'axis.text.y': robjects.r('theme_text')(**axis_text_y_args)
        }
        self._other_adds.append(robjects.r('opts')(**kargs))
        return self

    def hide_background(self):
        self._other_adds.append(robjects.r('opts')(**{'panel.background': robjects.r('theme_blank()')}))
        return self

    def hide_y_axis_labels(self):
        self._other_adds.append(robjects.r('opts')(**{"axis.text.y": robjects.r('theme_blank()')}))

    def hide_x_axis_labels(self):
        self._other_adds.append(robjects.r('opts')(**{"axis.text.x": robjects.r('theme_blank()')}))

    def hide_axis_ticks(self):
        self._other_adds.append(robjects.r('opts')(**{"axis.ticks": robjects.r('theme_blank()')}))

    def hide_y_axis_title(self):
        self._other_adds.append(robjects.r('opts')(**{"axis.title.y": robjects.r('theme_blank()')}))

    def hide_x_axis_title(self):
        self._other_adds.append(robjects.r('opts')(**{"axis.title.x": robjects.r('theme_blank()')}))

    def scale_fill_manual(self, list_of_colors, guide = None):
        kwargs = {}
        if guide is not None: 
            kwargs['guide'] = guide
        kwargs['values'] = numpy.array(list_of_colors) 
        self._other_adds.append(robjects.r('scale_fill_manual')(**kwargs))

    def scale_fill_brewer(self, name=None, palette=1, guide = None, typ='qual'):
        other_params = {}
        if not name is None:
            other_params['name'] = name
        if not palette is None:
            other_params['palette'] = palette
        if guide is not None:
            other_params['guide'] = guide
        self._other_adds.append(robjects.r('scale_fill_brewer')(palette = palette, **{'type': typ}))

    def scale_fill_hue(self, h=None, l=None, c=None, limits=None, breaks=None, labels=None, h_start=None, direction=None, guide = None):
        other_params = {}
        if not h is None:
            other_params['h'] = h
        if not l is None:
            other_params['l'] = l
        if not c is None:
            other_params['c'] = c
        if not limits is None:
            other_params['limits'] = numpy.array(limits)
        if not breaks is None:
            other_params['breaks'] = numpy.array(breaks)
        if not labels is None:
            other_params['labels'] = numpy.array(labels)
        if not h_start is None:
            other_params['h.start'] = h_start
        if not direction is None:
            other_params['direction'] = direction
        if guide is not None:
            other_params['guide'] = guide
        self._other_adds.append(robjects.r('scale_fill_hue')(**other_params))

    def scale_fill_gradient(self, low, high, mid=None, midpoint=None, name=None, space='rgb', breaks=None, labels=None, limits=None, trans=None, guide = None):
        other_params = {}
        other_params['low'] = low
        other_params['high'] = high
        if midpoint is not None and mid is None:
            raise ValueError("If you pass in a midpoint, you also need to set a value for mid")
        if mid is not None:
            other_params['mid'] = mid
        if midpoint is not None:
            other_params['midpoint'] = midpoint
        if name is not None:
            other_params['name'] = name
        if space is not None:
            other_params['space'] = space
        if breaks is not None:
            other_params['breaks'] = breaks
        if limits is not None:
            other_params['limits'] = limits
        if trans is not None:
            other_params['trans'] = trans
        if guide is not None:
            other_params['guide'] = guide
        
        if mid is not None:
            self._other_adds.append(robjects.r('scale_fill_gradient2')(**other_params))
        else:
            self._other_adds.append(robjects.r('scale_fill_gradient')(**other_params))
        return self
    def scale_fill_gradientn(self, *args):
        self._other_adds.append(robjects.r('scale_fill_gradientn')(colours = list(args)))


    def scale_fill_rainbow(self, number_of_steps = 7):
        self._other_adds.append(robjects.r('scale_fill_gradientn')(colours = robjects.r('rainbow')(number_of_steps)))


    def coord_flip(self):
        self._other_adds.append(robjects.r('coord_flip()'))
        return self

    def coord_polar(self, theta="x", start=0, direction=1):
        self._other_adds.append(robjects.r('coord_polar')(
            theta=theta,
            start=start,
            direction=direction,
           # expand=expand
            ))
        return self

    def legend_position(self, value):
        if type(value) is tuple:
            self._other_adds.append(robjects.r('opts(legend.position = c(%i,%i))' % value))
        else:
            self._other_adds.append(robjects.r('opts(legend.position = "%s")' % value))

    def hide_legend(self):
        self.legend_position('none')

    def guide_legend(self,**kwargs):
        r_args = {}
        for arg_name in [
            "title",
            "title_position",
            "title_theme",
            "title_hjust",
            "title_vjust",
            "label",
            "label_position",
            "label_theme",
            "label_hjust",
            "label_vjust",
            "keywidth",
            "keyheight",
            "direction",
            "default_unit",
            "override_aes",
            "nrow",
            "ncol",
            "byrow",
            "reverse", ]:
            if arg_name in kwargs and kwargs[arg_name] is not None: 
                r_args[arg_name.replace('_','.')] = kwargs[arg_name]
        return robjects.r('guide_legend')(**kwargs)
        

    def guide_colourbar(self,**kwargs):
        r_args = {}
        for arg_name in [
            "title",
            "title_position",
            "title_theme",
            "title_hjust",
            "title_vjust",
            "label",
            "label_position",
            "label_theme",
            "label_hjust",
            "label_vjust",
            "keywidth",
            "keyheight",
            "direction",
            "default_unit",
            "override_aes",
            "nrow",
            "ncol",
            "byrow",
            "reverse",
            'nbin'
            ]:
            if arg_name in kwargs and kwargs[arg_name] is not None: 
                r_args[arg_name.replace('_','.')] = kwargs[arg_name]
        return robjects.r('guide_colourbar')(**kwargs)
       
    def hide_panel_border(self):
        self._other_adds.append(robjects.r('opts(panel.border=theme_rect(fill=NA, colour=NA))'))

    def set_axis_color(self, color=None):
        if color is None:
            self._other_adds.append(robjects.r('opts(axis.line = theme_segment())'))
        else:
            self._other_adds.append(robjects.r('opts(axis.line = theme_segment(colour = "%s"))' % color))

    def hide_grid(self):
        self._other_adds.append(robjects.r('opts(panel.grid.major = theme_blank())'))
        self._other_adds.append(robjects.r('opts(panel.grid.minor = theme_blank())'))

    def hide_grid_minor(self):
        #self._other_adds.append(robjects.r('opts(panel.grid.major = theme_line(colour = NA))'))
 #       self._other_adds.append(robjects.r('opts(panel.grid.minor = theme_line(colour = NA))'))
        self._other_adds.append(robjects.r('opts(panel.grid.minor = theme_blank())'))

    def smaller_margins(self):
        self._other_adds.append(robjects.r('opts(panel.margin = unit(0.0, "lines"))'))
        self._other_adds.append(robjects.r('opts(axis.ticks.margin = unit(0.0, "cm"))'))
        self.plot_margin(0, 0, 0, 0)

    def plot_margin(self, top, left, bottom, right):
        self._other_adds.append(robjects.r('opts(plot.margin = unit(c(%i,%i,%i,%i), "lines"))' % (top, left, bottom, right)))

    def scale_shape_manual(self, values):
        self._other_adds.append(robjects.r('scale_shape_manual')(values=values))

    def scale_shape_identity(self):
        self._other_adds.append(robjects.r('scale_shape_identity')())

    def scale_shape(self, solid=True):
        self._other_adds.append(robjects.r('scale_shape')(solid=solid))

    def scale_colour_manual(self, values, guide = None):
        kwargs = {}
        if guide is not None:
            kwargs['guide'] = guide
        self._other_adds.append(robjects.r('scale_colour_manual')(values=numpy.array(values), **kwargs))

    def scale_colour_manual_labels(self, vals, labels, guide = None):
        kwargs = {}
        if guide is not None:
            kwargs['guide'] = guide
        self._other_adds.append(robjects.r("""
        scale_colour_manual
        """)(values=numpy.array(vals), labels = numpy.array(labels), **kwargs))

    def scale_color_manual(self, *args, **kwargs):
        return self.scale_colour_manual(*args, **kwargs)

    def scale_color_identity(self, guide = None):
        kwargs = {}
        if guide is not None:
            kwargs['guide'] = guide
        self._other_adds.append(robjects.r('scale_colour_identity')(**kwargs))

    def scale_color_hue(self, guide = None):
        kwargs = {}
        if guide is not None:
            kwargs['guide'] = guide
        self._other_adds.append(robjects.r('scale_colour_hue')(**kwargs))

    def scale_color_brewer(self, name=None, palette='Set1', guide = None):
        other_params = {}
        if not name is None:
            other_params['name'] = name
        if not palette is None:
            other_params['palette'] = palette
        if guide is not None:
            other_params['guide'] = guide
        self._other_adds.append(robjects.r('scale_colour_brewer')(**other_params))

    def scale_colour_grey(self, guide = None):
        kwargs = {}
        if guide is not None:
            kwargs['guide'] = guide
        self._other_adds.append(robjects.r('scale_colour_grey')(**kwargs))

    def scale_color_gradient(self, low, high, mid=None, midpoint=None, name=None, space='rgb', breaks=None, labels=None, limits=None, trans=None, guide = None):
        other_params = {}
        other_params['low'] = low
        other_params['high'] = high
        if midpoint is not None and mid is None:
            raise ValueError("If you pass in a midpoint, you also need to set a value for mid")
        if mid is not None:
            other_params['mid'] = mid
        if midpoint is not None:
            other_params['midpoint'] = midpoint
        if name is not None:
            other_params['name'] = name
        if space is not None:
            other_params['space'] = space
        if breaks is not None:
            other_params['breaks'] = breaks
        if limits is not None:
            other_params['limits'] = limits
        if trans is not None:
            other_params['trans'] = trans
        if guide is not None:
            other_params['guide'] = guide
        
        if mid is not None:
            self._other_adds.append(robjects.r('scale_colour_gradient2')(**other_params))
        else:
            self._other_adds.append(robjects.r('scale_colour_gradient')(**other_params))
        return self

    def scale_fill_grey(self, guide = None):
        kwargs = {}
        if guide is not None:
            kwargs['guide'] = guide
        self._other_adds.append(robjects.r('scale_fill_grey')(**kwargs))


class MultiPagePlot(Plot):
    """A plot job that splits faceted variables over multiple pages.

    Bug: The last page may get fewer variables and the plots get a different size than the other pages
    
    """
    def __init__(self, dataframe, facet_variable_x, facet_variable_y = None, ncol_per_page = 3, n_rows_per_page = 5, fixed_x = False, fixed_y = True, facet_style = 'wrap'):
        Plot.__init__(self, dataframe)
        self.facet_variable_x = facet_variable_x
        self.facet_variable_y = facet_variable_y
        if facet_variable_x not in dataframe.columns_ordered:
            raise ValueError("facet_variable_x %s not in dataframe.columns_ordered" % facet_variable_x)
        if facet_variable_y and facet_variable_y not in dataframe.columns_ordered:
            raise ValueError("facet_variable_y %s not in dataframe.columns_ordered" % facet_variable_y)
        if facet_style not in ('wrap', 'grid'):
            raise ValueError("facet_style must be one of wrap, grid")
        self.facet_style = facet_style
        self.fixed_x = fixed_x
        self.fixed_y = fixed_y
        self.ncol_per_page = ncol_per_page
        no_of_x_variables = len(dataframe.get_column_unique(self.facet_variable_x))
        if self.facet_variable_y:
            no_of_y_variables = len(dataframe.get_column_unique(self.facet_variable_y))
            no_of_plots = no_of_x_variables * no_of_y_variables
        else:
            no_of_plots = no_of_x_variables
        self.plots_per_page = ncol_per_page * n_rows_per_page
        pages_needed = math.ceil(no_of_plots / float(self.plots_per_page))
        self.width = 8.26772
        self.height = 11.6929
        self.no_of_pages = pages_needed

    def _iter_by_pages(self):
        def grouper(iterable, n, fillvalue=None):
            "Collect data into fixed-length chunks or blocks"
            # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
            args = [iter(iterable)] * n
            return itertools.izip_longest(fillvalue=fillvalue, *args)
        if self.facet_variable_y:
            raise NotImplemented("The splitting into two variable sub_dfs is not implemented yet")
        else:
            x_column = 'dat_%s' % self.old_names.index(self.facet_variable_x)
            x_values = self.dataframe.get_column_unique(x_column)
            for group in grouper(sorted(x_values), self.plots_per_page):
                keep = numpy.zeros((len(self.dataframe)), dtype=numpy.bool)
                for value in group:
                    if value:
                        keep[self.dataframe.get_column_view(x_column) == value] = True
                yield self.dataframe[keep, :]

    def render(self, output_filename, width=8, height=6, dpi=300):
        if not output_filename.endswith('.pdf'):
            raise ValueError("MultiPagePlots only for pdfs")
        if self.facet_variable_y:
               new_one = 'dat_%s' % self.old_names.index(self.facet_variable_x)
               new_two = 'dat_%s' % self.old_names.index(self.facet_variable_y)
               facet_specification = '%s ~ %s' % (new_one, new_two)
        else:
               params = self._translate_params({"": self.facet_variable_x})[0]
               facet_specification = params.replace('=', '~')
        
        if self.fixed_x and not self.fixed_y:
            scale = 'free_y'
        elif not self.fixed_x and self.fixed_y:
            scale = 'free_x'
        elif not self.fixed_x and not self.fixed_y:
            scale = 'free'
        else:
            scale = 'fixed'
        if self.facet_style == 'grid':
            self._other_adds.append(robjects.r('facet_grid')(robjects.r(facet_specification), scale = scale, ncol=self.ncol_per_page))
        elif self.facet_style == 'wrap':
            self._other_adds.append(robjects.r('facet_wrap')(robjects.r(facet_specification), scale = scale, ncol=self.ncol_per_page))

        robjects.r('pdf')(output_filename, width = 8.26, height = 11.69)
        page_no = 0
        for sub_df in self._iter_by_pages():
            page_no += 1
            plot = self.r['ggplot'](sub_df)
            for obj in self._other_adds:
                plot = self.r['add'](plot, obj)
            for name, value in self.lab_rename.items():
                plot = self.r['add'](
                        plot, robjects.r('labs(%s = "%s")' % (name, value)))
            print self._other_adds
            robjects.r('print')(plot)
        robjects.r('dev.off')()

    def facet_grid(self, column_one, column_two=None, fixed_x=True, fixed_y=True, ncol=None):
        raise ValueError("MultiPagePlots specify faceting on construction")

    def facet(self, column_one, column_two=None, fixed_x=True, fixed_y=True, ncol=None):
        raise ValueError("MultiPagePlots specify faceting on construction")


def plot_heatmap(output_filename, data, infinity_replacement_value = 10, low='blue', high = 'red', mid='white', nan_color='grey', hide_genes = True, array_cluster_method = 'cosine',
        x_label = 'Condition', y_label = 'Gene', colors = None, hide_tree = False, exclude_those_with_too_many_nans_in_y_clustering = False, width = None, row_order = False, column_order = False):
    """This code plots a heatmap + dendrogram.
    (unlike add_heatmap, which just does the squares on an existing plot)
    @data is a df of {'gene':, 'condition':, 'expression_change'}
    nan, is translated to 0 (but plotted grey), infinity to infinity_replacement_value (or -1 * infinity_replacement_value for negative infinity).

    Clustering is performed using the cosine distance - on the genes.

    If there is a gene name occuring twice, we use it's median!

    @low, high, mid, nan_color allow you to modify the colors
    @hide_genes hides the y-axis labels,
    @array_cluster_method may be cosine or hamming_on_0 (threshold on 0, then hamming)
    @x_label and @y_label are the axis labels,
    keep_column_order enforces the order in the df
    keep_row_order does the same.
    @colors let's you supply colors - TODO: What format?
    @hide_tree hides the tree
    @exclude_those_with_too_many_nans_in_y_clustering removes elements with more than 25% nans from deciding the order in the y-clustering

    It's using ggplot and ggdendro... in the end, this code breads insanity"""
    column_number = len(set(data.get_column_view('condition')))
    row_number = len(data) / column_number
    keep_column_order = False
    if column_order is not False:
        keep_column_order = True
    keep_row_order = False
    if row_order is not False:
        keep_row_order = True
    load_r()
    valid_array_cluster = 'hamming_on_0', 'cosine'
    if not array_cluster_method in valid_array_cluster:
        raise ValueError("only accepts array_cluster_method methods %s" % valid_array_cluster)
    df = data
    if colors == None:
        colors = ['grey' for x in range(len(data))]
    
   
    #R's scale NaNs everything on any of these values...
    #df[numpy.isnan(df.get_column_view('expression_change')), 'expression_change'] = 0 #we do this part in R now.
    df[numpy.isposinf(df.get_column_view('expression_change')), 'expression_change'] = infinity_replacement_value
    df[numpy.isneginf(df.get_column_view('expression_change')), 'expression_change'] = -1 * infinity_replacement_value
    
    if len(df.get_column_unique('condition')) < 2 or len(df.get_column_unique('gene')) < 2:
        op = open(output_filename,'wb')
        op.write("not enough dimensions\n")
        op.close()
        return
    file_extension = output_filename[-3:].lower()
    if not (file_extension == 'pdf' or file_extension == 'png'):
        raise ValueError('File extension must be .pdf or .png, outfile was '+output_filename)
    robjects.r("""
    normvec<-function(x) 
    {
       sqrt(x%*%x)
    }
    cosine_distance = function(a, b)
    {
       1 - ((a %*% b) / ( normvec(a) * normvec(b)))[1]
    }

    dist_cosine = function(x)
    {
       x = as.matrix(x)
       N = nrow(x)
       res = matrix(0, nrow = N, ncol= N)
       for (i in 1:N)
       {
           for (t in 1:N)
           {
              res[i,t] = cosine_distance(x[i,], x[t,])
           }
       }
       as.dist(res)
    }

    hamming_distance = function(a, b)
    {
        sum(a != b)
    }

    dist_hamming = function(x)
    {
       x = as.matrix(x)
       N = nrow(x)
       res = matrix(0, nrow = N, ncol= N)
       for (i in 1:N)
       {
           for (t in 1:N)
           {
              res[i,t] = hamming_distance(x[i,], x[t,])
           }
       }
       as.dist(res)
    }
    """)
    robjects.r("""
    library(ggplot2)
    library(reshape)
    library(ggdendro) 
    library(grid) 

    do_tha_funky_heatmap = function(outputfilename, df, 
                        low, mid, high, nan_color, 
                        hide_genes, width, height, array_cluster_method, 
                        keep_column_order, keep_row_order, colors, hide_tree, exclude_those_with_too_many_nans_in_y_clustering, row_order, column_order)
    {
        df$condition <- factor(df$condition)
        options(expressions = 50000) #allow more recursion
        
        #transform df into a rectangualr format

        df_cast = cast(df, gene ~ condition, value='expression_change', fun.aggregate=median) 
        col_names = names(df_cast)
        row_names = df_cast$gene
        df_cast = df_cast[do.call(order,df_cast['gene']),]
        df_scaled = as.matrix(scale(df_cast))
        df_scaled[is.nan(df_scaled)] = 0
        df_scaled[is.na(df_scaled)] = 0
        #do the row clustering
        if (!keep_row_order)
        {
            if (exclude_those_with_too_many_nans_in_y_clustering) #when clustering genes, leave out those samples with too many nans
            {
                df_scaled_with_nans = as.matrix(scale(df_cast)) #we need it a new, with nans
                nan_count_per_column = colSums(is.na(df_scaled_with_nans))
                too_much = dim(df_scaled_with_nans)[1] / 4.0
                exclude = nan_count_per_column >= too_much
                keep = !exclude
                df_scaled_with_nans = df_scaled_with_nans[, keep]
                df_scaled_with_nans[is.nan(df_scaled_with_nans)] = 0
                df_scaled_with_nans[is.na(df_scaled_with_nans)] = 0
                dd.row <- as.dendrogram(hclust(dist_cosine(df_scaled_with_nans)))
            }
            else
            {
                dd.row <- as.dendrogram(hclust(dist_cosine(df_scaled)))
            }
        }
        #do the column clustering.
        if(!keep_column_order){
            if (array_cluster_method == 'cosine')
            {
                dd.col <- as.dendrogram(hclust(dist_cosine(t(df_scaled))))
            }
            else if (array_cluster_method == 'hamming_on_0') 
            {
                df_hamming = as.matrix(df_cast) > 0
                df_hamming[is.nan(df_hamming)] = 0
                df_hamming[is.na(df_hamming)] = 0
                dd.col <- as.dendrogram(hclust(dist_hamming(t(df_hamming))))
            }
        }
        if (keep_row_order)
        {
            row.ord = 1:length(row_order)
            for(i in 1:length(row_order)){
                row.ord[i] = which(row_names==row_order[i])
            }
            row.ord = rev(row.ord)
        }
        else
        {
            row.ord <- order.dendrogram(dd.row) 
        }
        if (keep_column_order)
        {
            tmp = 1:length(column_order)
            for(i in 1:length(column_order)){
                tmp[i] = which(col_names==column_order[i])-1
            }
            col.ord <- tmp
        }
        else
        {
            col.ord <- order.dendrogram(dd.col)
        }
        xx <- scale(df_cast, FALSE, FALSE)[row.ord, col.ord]
        xx_names <- attr(xx, 'dimnames')
        df <- as.data.frame(xx)
        colnames(df) <- xx_names[[2]]
        df$gene <- xx_names[[1]]
        df$gene <- with(df, factor(gene, levels=gene, ordered=TRUE))
        mdf <- melt(df, id.vars="gene")
        
        tmp = c()
        i = 1
        for (gene in df$gene)
        {
            index = which(colors$gene == gene)
            colll <- as.character(colors$color[index])
            tmp[i] = colll
            i = i +1    
        }
        colors = tmp
        if(!keep_column_order){
            ddata_x <- dendro_data(dd.col)
            }
        if(!keep_row_order)
        {
            ddata_y <- dendro_data(dd.row)
        }
        ### Set up a blank theme
        
        theme_none <- theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            axis.title.x = element_text(colour=NA),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.line = element_blank(),
            axis.ticks = element_blank()
            )

        ### Create plot components ###    
        # Heatmap
        p1 <- ggplot(mdf, aes(x=variable, y=gene)) + 
            geom_tile(aes(fill=value)) + scale_fill_gradient2(low=low,mid=mid, high=high, na.value=nan_color) + theme(axis.text.x = element_text(angle=90, size=8, hjust=0, vjust=0, colour="black"),
            axis.title.y = element_blank(), axis.title.x = element_blank(),
            axis.text.y = element_text(colour="black"))
        if (hide_genes)
            p1 = p1 + theme(axis.text.y = element_blank())
        else
        {
            p1 = p1 + theme(strip.background = element_rect(colour = 'NA', fill = 'NA'), axis.text.y = element_text(colour=colors))
        }
        if (!keep_column_order && !hide_tree)
        {
            # Dendrogram 1
            p2 <- ggplot(segment(ddata_x)) + 
                geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) + 
                theme_none + theme(axis.title.x=element_blank())
        }

        if(!keep_row_order && !hide_tree)
        {
            # Dendrogram 2
            p3 <- ggplot(segment(ddata_y)) + 
                geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) + 
                coord_flip() + theme_none
        }

        if (grepl('png$', outputfilename))
            png(outputfilename, width=width * 72, height=height * 72)
        else if (grepl('pdf$', outputfilename))
            pdf(outputfilename, width=width, height=height)
        else
            error("Don't know that file format")
        grid.newpage()
        if (hide_tree)
            vp = viewport(1, 1, x=0.5, y=0.5)
        else
            vp = viewport(0.8, 0.8, x=0.4, y=0.4)

        #print(p1, vp=vp)
        if (!keep_column_order && !hide_tree)
        {
            #print(p2, vp=viewport(0.60, 0.2, x=0.4, y=0.9))
        }
        if (!keep_row_order && !hide_tree)
        {
            #print(p3, vp=viewport(0.2, 0.86, x=0.9, y=0.4))
        }
        dev.off()
    }
    """)
    if not width:
        width = len(df.get_column_unique('condition')) * 0.4 + 5
    height = len(df.get_column_unique('gene')) * 0.15 + 3
    robjects.r('do_tha_funky_heatmap')(output_filename, df, low, mid, high, nan_color, hide_genes, width, height, array_cluster_method, keep_column_order, keep_row_order, colors, hide_tree, exclude_those_with_too_many_nans_in_y_clustering, row_order, column_order)


def EmptyPlot(text_to_display = 'No data'):
    p = Plot(pandas.DataFrame({'x': [0], 'y': [0], 'text': [text_to_display]}))
    p.add_text('x', 'y', 'text')
    return p


try:
    import rpy2.robjects as ro
    import rpy2.robjects.conversion
    import rpy2.rinterface as rinterface
    import rpy2.robjects.numpy2ri

    def numpy2ri_vector(o):
        """Convert a numpy 1d array to an R vector.

        Unlike the original conversion which converts into a list, apperantly."""
        if len(o.shape) != 1:
            raise ValueError("Dataframe.numpy2ri_vector can only convert 1d arrays")
        #if isinstance(o, Factor):
            #res = ro.r['factor'](o.as_levels(), levels=o.levels, ordered=True)
        elif isinstance(o, numpy.ndarray):
            if not o.dtype.isnative:
                raise(ValueError("Cannot pass numpy arrays with non-native byte orders at the moment."))

            # The possible kind codes are listed at
            #   http://numpy.scipy.org/array_interface.shtml
            kinds = {
                # "t" -> not really supported by numpy
                "b": rinterface.LGLSXP,
                "i": rinterface.INTSXP,
                # "u" -> special-cased below
                "f": rinterface.REALSXP,
                "c": rinterface.CPLXSXP,
                # "O" -> special-cased below
                "S": rinterface.STRSXP,
                "U": rinterface.STRSXP,
                # "V" -> special-cased below
                }
            # Most types map onto R arrays:
            if o.dtype.kind in kinds:
                # "F" means "use column-major order"
    #            vec = rinterface.SexpVector(o.ravel("F"), kinds[o.dtype.kind])
                vec = rinterface.SexpVector(numpy.ravel(o,"F"), kinds[o.dtype.kind])
                res = vec
            # R does not support unsigned types:
            elif o.dtype.kind == "u":
                o = numpy.array(o, dtype=numpy.int64)
                return numpy2ri_vector(o)
                #raise(ValueError("Cannot convert numpy array of unsigned values -- R does not have unsigned integers."))
            # Array-of-PyObject is treated like a Python list:
            elif o.dtype.kind == "O":
                all_str = True
                all_bool = True
                for value in o:
                    if (
                            not type(value) is str and 
                            not type(value) is unicode and 
                            not type(value) is numpy.string_ and 
                            not (type(value) is numpy.ma.core.MaskedArray and value.mask == True) and
                            not (type(value) is numpy.ma.core.MaskedConstant and value.mask == True)
                                
                                ):
                        all_str = False
                        break
                    if not type(value) is bool or type(value) is numpy.bool_:
                        all_bool = False
                if (not all_str) and (not all_bool):
                    raise(ValueError("numpy2ri_vector currently does not handle object vectors: %s %s" % (value, type(value))))
                else:
                    #since we keep strings as objects
                    #we have to jump some hoops here
                    vec = rinterface.SexpVector(numpy.ravel(o,"F"), kinds['S'])
                    return vec
                    #res = ro.conversion.py2ri(list(o))
            # Record arrays map onto R data frames:
            elif o.dtype.kind == "V":
                raise(ValueError("numpy2ri_vector currently does not handle record arrays"))
            # It should be impossible to get here:
            else:
                raise(ValueError("Unknown numpy array type."))
        else:
            raise(ValueError("Unknown input to numpy2ri_vector."))
        return res

    # I can't figuer out how to do colnames(x) = value without a helper function
    ro.r("""
        set_colnames = function(df, names)
        {
            colnames(df) = names
            df
        }
        """)
       
    def convert_dataframe_to_r(o):
        #print 'converting', o, type(o)
        if isinstance(o, pandas.DataFrame):
            #print 'dataframe'
            dfConstructor = ro.r['data.frame']
            names = []
            parameters = []
            kw_params = {}
            for column_name in o.columns:
                try:
                    names.append(str(column_name))
                    if str(o[column_name].dtype) == 'category':  # There has to be a more elegant way to specify this...
                        col = ro.r('factor')(numpy.array(o[column_name]), list(o[column_name].cat.categories), ordered=True )

                    else:
                        col = numpy.array(o[column_name])
                        col = numpy2ri_vector(col)
                    parameters.append(col)
                except ValueError, e:
                    raise ValueError(str(e) + ' Offending column: %s, dtype: %s, content: %s' %( column_name, col.dtype, col[:10]))
            #kw_params['row.names'] = numpy2ri_vector(numpy.array(o.index))
            try:
                res = dfConstructor(*parameters, **kw_params)
                res = ro.r('set_colnames')(res, names)
            except TypeError:
                print parameters.keys()
                raise
        elif isinstance(o, numpy.ndarray):
            res = numpy2ri_vector(o)
        else:
            res =  ro.default_py2ri(o)
        return res

except ImportError: #guess we don't have rpy
    pass


all = [Plot, plot_heatmap]
