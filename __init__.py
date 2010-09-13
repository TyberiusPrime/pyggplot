import exptools
from ordereddict import OrderedDict
import sys
from hilbert import hilbert_plot, hilbert_to_image
import rpy2.robjects as robjects
exptools.ensureSoftwareVersion('pyvenn','tip')
import pyvenn
from square_euler import SquareEuler
import re
from sequence_logos import plot_sequences
try:
    from pwmlocationplotter import PWMLocationPlotter
except ImportError:
    pass

_r_loaded = False
def load_r():
    global _r_loaded
    if not _r_loaded:
        robjects.r('library(ggplot2)')
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
TransInvNegLog10 <- Trans$new("InvNegLog10", f = function(x) 10^(-x), 
inverse = function(x) -log10(x), labels = function(x) x)
TransInvNegLog10b <- Trans$new("InvNegLog10b", 
            f = function(x) -log10(x),
            inverse = function(x) 10^-x, 
            labels = function(x) bquote(10^.(-x)))

""")

import numpy


def r_expression(expr):
    return robjects.r('expression(%s)' % expr)

NA = robjects.r("NA")


class Plot:

    def __init__(self, dataframe, *ignored):
        load_r()
        self.r = {}
        self.r['ggplot'] = robjects.r['ggplot']
        self.r['aes'] = robjects.r['aes']
        self.r['add'] = robjects.r('ggplot2:::"+.ggplot"')
        self.r['layer'] = robjects.r['layer']
        self.r['facet_wrap'] = robjects.r['facet_wrap']
        self.r['geom_text'] = robjects.r['geom_text']
        self.r['ggsave'] = robjects.r['ggsave']
        self.old_names = []
        self.lab_rename = {}
        self.dataframe = self._prep_dataframe(dataframe)
        self._other_adds = []

    def _prep_dataframe(self, dataframe):
        df = dataframe.copy()
        new_names = []
        for name in df.columns_ordered:
            if not name in self.old_names:
                new_names.append(name)
        self.old_names.extend(new_names)
        for name in df.columns_ordered[:]:
            df.rename_column(name, 'dat_%s' % self.old_names.index(name))
        return df

    def render(self, output_filename, width=8, height=6, dpi=300):
        #print self.dataframe
        try:
            plot = self.r['ggplot'](self.dataframe)
        except ValueError:
            print self.old_names
            raise
        for obj in self._other_adds:
            plot = self.r['add'](plot, obj)
        for name, value in self.lab_rename.items():
            plot = self.r['add'](plot, robjects.r('labs(%s = "%s")' % (name, value)))
        #plot = self.r['add'](plot, self.r['layer'](geom="point"))
        #robjects.r('options( error=recover )')
        self.r['ggsave'](filename=output_filename,plot=plot, width=width, height=height, dpi=dpi)

    def add_aesthetic(self, name, column_name):
        self._aesthetics[name] = column_name

    def add_scatter(self, x_column, y_column, color=None, group=None, shape=None, size=None, alpha=None):
        aes_params = {'x': x_column, 'y': y_column}
        other_params = {}
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        if shape:
            aes_params['shape'] = shape
        if not alpha is None:
            if type(alpha) == int or type(alpha) == float:
                other_params['alpha'] = alpha
            else:
                aes_params['alpha'] = str(alpha)
        if not size is None:
            if type(size) is int or type(size) is float:
                other_params['size'] = size
            else:
                aes_params['size'] = str(size)
        self._other_adds.append(robjects.r('geom_point')(self._build_aesthetic(aes_params), **other_params))
        return

    def add_jitter(self, x_column, y_column, color=None, jitter_x = None, jitter_y = None, shape=None, size=None, data=None):
        aes_params = {'x': x_column, 'y': y_column}
        other_params = {}
        if color:
            aes_params['colour'] = color
        position_params = {}
        if not jitter_x is None:
            position_params['width'] = jitter_x
        if not jitter_y is None:
            position_params['height'] = jitter_y
        if shape:
            aes_params['shape'] = shape
        if not size is None:
            if type(size) is int or type(size) is float:
                other_params['size'] = size
            else:
                aes_params['size'] = str(size)
        if not data is None:
            other_params['data'] = self._prep_dataframe(data)

        if position_params:
            self._other_adds.append(robjects.r('geom_jitter')(self._build_aesthetic(aes_params), position=robjects.r('position_jitter')(**position_params), **other_params))
        else:
            self._other_adds.append(robjects.r('geom_jitter')(self._build_aesthetic(aes_params), **other_params))

    def add_stacked_bar_plot(self, x_column, y_column, fill):
        aes_params  = {'x': x_column, 'y': y_column, 'fill': fill}
        self._other_adds.append(robjects.r('geom_bar')(self._build_aesthetic(aes_params), position='stack'))


    def add_histogram(self, x_column, y_column = "..count..", color=None, group = None, fill=None, position="dodge", add_text = False):
        aes_params = {'x': x_column}
        if fill:
            aes_params['fill'] = fill
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
            #x = x_column, y = y_column)
        self._other_adds.append(
            robjects.r('geom_bar')(self._build_aesthetic(aes_params), stat='bin', position=position)
        )
        if add_text:
            self._other_adds.append(
                robjects.r('geom_text')(self._build_aesthetic({'x': x_column, 'y':'..count..', 'label':'..count..'}),stat='bin' ))


    def add_bar(self, *args, **kwargs):
        return self.add_bar_plot(*args, **kwargs)
    def add_bar_plot(self, x_column, y_column, color=None, group = None, fill=None, position="dodge", data=None):
        aes_params = {'x': x_column, 'y': y_column}
        other_params = {
                'stat':  'identity',
                'position': position}
        if not fill is None:
            if fill in self.old_names:
                aes_params['fill'] = fill
            else:
                other_params['fill'] = fill
        if not color is None:
            if color in self.old_names:
                aes_params['colour'] = color
            else:
                other_params['colour'] = color
        if group:
            aes_params['group'] = group
        if not data is None:
            other_params['data'] = self._prep_dataframe(data)

        self._other_adds.append(
            robjects.r('geom_bar')(self._build_aesthetic(aes_params),  **other_params)
        )

    def add_box_plot(self, x_column, y_column, color=None, group = None, fill=None):
        aes_params = {'x': x_column, 'y': y_column}
        if fill:
            aes_params['fill'] = fill
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        self._other_adds.append(
            robjects.r('geom_boxplot')(self._build_aesthetic(aes_params))
        )

    def add_heatmap(self, x_column, y_column, fill, low="red", mid=None, high="blue", midpoint=0):
        aes_params = {}
        aes_params['x'] = x_column
        aes_params['y'] = y_column
        aes_params['fill'] = fill
        self._other_adds.append(
            robjects.r('geom_tile')(self._build_aesthetic(aes_params), stat="identity")
        )
        if mid is None:
            self._other_adds.append(
                    robjects.r('scale_fill_gradient')(low=low, high=high)
                    )
        else:
            self._other_adds.append(
                robjects.r('scale_fill_gradient2(low="%s", mid="%s", high="%s", midpoint=%.f)' % (
                low, mid, high, midpoint))
            )
    
    def _fix_axis_label(self, aes_name, new_name, real_name):
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

    def _translate_params(self, params):
        aes_params = []
        for aes_name, aes_column in params.items():
            if aes_column in self.old_names:
                new_name = 'dat_%s'  % self.old_names.index(aes_column)
                aes_params.append('%s=%s' % (aes_name, new_name))
                self._fix_axis_label(aes_name, new_name, aes_column)
            else: #a fixeud value
                aes_params.append("%s=%s" % (aes_name, aes_column))
        return aes_params

    def _build_aesthetic(self, params):
        aes_params = self._translate_params(params)
        aes_params = ", ".join(aes_params)
        print aes_params
        return robjects.r('aes(%s)' % aes_params)


    def add_line(self, x_column, y_column, color=None, group=None, shape=None, alpha=1.0, size=None, data=None):
        aes_params = {'x': x_column, 'y': y_column}
        other_params = {}
        if color:
            if color in self.old_names:
                aes_params['colour'] = color
            else:
                other_params['colour'] = color
        if group:
            aes_params['group'] = group
        if shape:
            aes_params['shape'] = shape
        if type(alpha) == int or type(alpha) == float:
            other_params['alpha'] = alpha
        else:
            aes_params['alpha'] = str(alpha)
        if not size is None:
            if type(size) is int or type(size) is float:
                other_params['size'] = size
            else:
                aes_params['size'] = str(size)
        if not data is None:
            other_params['data'] = self._prep_dataframe(data)
        self._other_adds.append(robjects.r('geom_line')(self._build_aesthetic(aes_params), **other_params))

    def add_area(self, x_column, y_column, color=None, fill=None, linetype=1, alpha=1.0, size=None, data=None, position=None):
        aes_params = {'x': x_column, 'y': y_column}
        other_params = {}
        if not color is None:
            if color in self.old_names:
                aes_params['colour'] = color
            else:
                other_params['colour'] = color
        if not fill is None:
            if fill in self.old_names:
                aes_params['fill'] = fill
            else:
                other_params['fill'] = fill
        if not alpha is None:
            if alpha in self.old_names:
                aes_params['alpha'] = alpha
            else:
                other_params['alpha'] = alpha
        if not size is None:
            if size in self.old_names:
                aes_params['size'] = size
            else:
                other_params['size'] = size
        if not linetype is None:
            if linetype in self.old_names:
                aes_params['linetype'] = linetype
            else:
                other_params['linetype'] = linetype
        if not position is None:
            other_params['position'] = position
        if not data is None:
            other_params['data'] = self._prep_dataframe(data)

        self._other_adds.append(robjects.r('geom_area')(self._build_aesthetic(aes_params), **other_params))

    def add_ribbon(self, x, ymin, ymax, color = None, fill = None, size = 0.5, linetype = 1, alpha = 1, position=None, data=None):
        aes_params = {'x': x}
        other_params = {}
        if ymin in self.old_names:
            aes_params['ymin'] = ymin
        else:
            other_params['ymin'] = ymin
        if ymax in self.old_names:
            aes_params['ymax'] = ymax
        else:
            other_params['ymax'] = ymax
        if not color is None:
            if color in self.old_names:
                aes_params['colour'] = color
            else:
                other_params['colour'] = color
        if not fill is None:
            if fill in self.old_names:
                aes_params['fill'] = fill
            else:
                other_params['fill'] = fill
        if not alpha is None:
            if alpha in self.old_names:
                aes_params['alpha'] = alpha
            else:
                other_params['alpha'] = alpha
        if not size is None:
            if size in self.old_names:
                aes_params['size'] = size
            else:
                other_params['size'] = size
        if not linetype is None:
            if linetype in self.old_names:
                aes_params['linetype'] = linetype
            else:
                other_params['linetype'] = linetype
        if not position is None:
            other_params['position'] = position
        if not data is None:
            other_params['data'] = self._prep_dataframe(data)

        self._other_adds.append(robjects.r('geom_ribbon')(self._build_aesthetic(aes_params), **other_params))

    def add_error_bars(self, x_column, ymin, ymax, color=None, group=None, position=None, width=0.25,alpha=1):
        aes_params = {'x': x_column, 'ymin': ymin, 'ymax':ymax}
        other_params = {}
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        if type(alpha) == int or type(alpha) == float:
            other_params['alpha'] = alpha
        else:
            aes_params['alpha'] = str(alpha)
        if position:
            other_params['position'] = position
        other_params['width'] = width
        self._other_adds.append(robjects.r('geom_errorbar')(self._build_aesthetic(aes_params), **other_params))

    def add_error_barsh(self, x_column, ypos, xmin, xmax, color=None, group=None, position=None, width=0.25, alpha=1):
        aes_params = {'x': x_column, 'y': ypos, 'xmin': xmin, 'xmax':xmax}
        other_params = {}
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        if type(alpha) == int or type(alpha) == float:
            other_params['alpha'] = alpha
        else:
            aes_params['alpha'] = str(alpha)
        if position:
            other_params['position'] = position
        other_params['width'] = width

        self._other_adds.append(robjects.r('geom_errorbarh')(self._build_aesthetic(aes_params), **other_params))

    def add_ab_line(self, intercept, slope, alpha = None, size = None):
        other_params = {}
        aes_params = {}
        if not alpha is None:
            if type(alpha) == int or type(alpha) == float:
                other_params['alpha'] = alpha
            else:
                aes_params['alpha'] = str(alpha)
        if not size is None:
            if type(size) == int or type(size) == float:
                other_params['size'] = size
            else:
                aes_params['size'] = str(size)
        other_params['intercept'] = intercept
        other_params['slope'] = slope
        self._other_adds.append(robjects.r('geom_abline')(self._build_aesthetic(aes_params), **other_params))

    def add_density(self, x_column, y_column = None, color = None):
        """add a kernel estimated density plot - gauss kernel and bw.SJ estimation of bandwith"""
        aes_params = {'x': x_column}
        if y_column:
            aes_params['y'] =  y_column

        if color:
            aes_params['colour'] = color
        self._other_adds.append(robjects.r('geom_density')(
            self._build_aesthetic(aes_params),
            bw = robjects.r('bw.SJ')(self.dataframe.get_column_view(self.old_names.index(x_column)))
            )
        )

    def add_density_2d(self, x_column, y_column, color = None):
        """add a kernel estimated density plot - gauss kernel and bw.SJ estimation of bandwith"""
        aes_params = {'x': x_column}
        aes_params['y'] =  y_column
        if color:
            aes_params['colour'] = color
        self._other_adds.append(robjects.r('geom_density2d')(
            self._build_aesthetic(aes_params),
            )
        )

    def add_stat_sum_color(self, x_column, y_column, size = 0.5):
        """A scatter plat that's colored by no of overlapping points"""
        aes_params = {'x': x_column}
        aes_params['y'] =  y_column
        aes_params['color'] = '..n..'
        self._other_adds.append(robjects.r('stat_sum')(self._build_aesthetic(aes_params), size = size))

    def add_rect(self, x_min_column, x_max_column, y_min_colum, y_max_column, color=None, fill = None, data = None, alpha = None):
        aes_params = {'xmin': x_min_column, 'xmax': x_max_column, 'ymin': y_min_colum, 'ymax': y_max_column}
        other_params = {}
        if not data is None:
            other_params['data'] = self._prep_dataframe(data)
        if color:
            aes_params['colour'] = color
        if fill:
            aes_params['fill'] = fill
        if type(alpha) == int or type(alpha) == float:
            other_params['alpha'] = alpha
        else:
            aes_params['alpha'] = str(alpha)

        print self.old_names
        obj = robjects.r('geom_rect')(self._build_aesthetic(aes_params), **other_params)
        self._other_adds.append(obj)

    def add_stat_smooth(self):
        self._other_adds.append(robjects.r('stat_smooth(method="lm", se=FALSE)'))

    def set_title(self, title):
        self._other_adds.append(robjects.r('opts(title = "%s")' %  title))

    def add_vertical_bar(self, xpos, alpha=0.5, color='black'):
        self._other_adds.append(
            robjects.r('geom_vline(aes(xintercept = %s),  alpha=%f, color="%s")' % (xpos, alpha, color))
        )

    def add_horizontal_bar(self, ypos, alpha=0.5, color='black'):
        self._other_adds.append(
            robjects.r('geom_hline(aes(yintercept = %f),  alpha=%f, color="%s")' % (ypos, alpha,color))
        )

    def add_segment(self, xstart, xend, ystart, yend, color, alpha = 1.0, size=0.5):
        self._other_adds.append(
            robjects.r('geom_segment')
            (
                robjects.r('aes(x=x, y=y, xend=xend, yend=yend)'),
                exptools.DataFrame.DataFrame({"x": [xstart], 'xend': [xend], 'y': [ystart], 'yend': [yend]}),
                colour=color,
                alpha = alpha,
                size = 0.5

            )
        )
    
    def add_text(self, x, y, label, data = None, angle=None, alpha=None, size=None, hjust=None, vjust=None, fontface = None, color=None):
        aes_params = {
            'x': x,
            'y': y,
            'label': label
        }
        other_params = {}
        if not data is None:
            other_params['data'] = self._prep_dataframe(data)
        if not angle is None:
            if type(angle) == int or type(angle) == float:
                other_params['angle'] = angle
            else:
                aes_params['angle'] = str(angle)
        if not alpha is None:
            if type(alpha) == int or type(alpha) == float:
                other_params['alpha'] = alpha
            else:
                aes_params['alpha'] = str(alpha)
        if not size is None:
            if type(size) == int or type(size) == float:
                other_params['size'] = size
            else:
                aes_params['size'] = str(size)
        if not vjust is None:
            if type(vjust) == int or type(vjust) == float:
                other_params['vjust'] = vjust
            else:
                aes_params['vjust'] = str(vjust)
        if not hjust is None:
            if type(hjust) == int or type(hjust) == float:
                other_params['hjust'] = hjust
            else:
                aes_params['hjust'] = str(hjust)
        if not color is None:
            if color in self.old_names:
                aes_params['colour'] = color
            else:
                other_params['colour'] = color

        if not fontface is None:
            other_params['fontface'] = fontface
        print aes_params, other_params
        self._other_adds.append(
            robjects.r('geom_text')(self._build_aesthetic(aes_params), **other_params)
        )

    def facet(self, column_one, column_two = None, fixed_x = True, fixed_y = True, ncol=None):
        facet_wrap = robjects.r['facet_wrap']
        if fixed_x and not fixed_y:
            scale = 'free_y'
        elif not fixed_x and fixed_y:
            scale = 'free_x'
        elif not fixed_x and fixed_y:
            scale = 'free'
        else:
            scale = 'fixed'
        if column_two:
            new_one = 'dat_%s'  % self.old_names.index(column_one)
            new_two = 'dat_%s'  % self.old_names.index(column_two)
            facet_specification = '%s ~ %s' % (new_one, new_two)
        else:
            params = self._translate_params({"":column_one})[0]
            facet_specification = params.replace('=', '~')
            #facet_specification = '~ %s' % (column_one,)
        print facet_specification
        params = {
            'scale': scale}
        if ncol:
            params['ncol'] = ncol
        self._other_adds.append(facet_wrap(robjects.r(facet_specification), **params))


    def greyscale(self):
        self._other_adds.append( robjects.r('scale_colour_grey()'))
        self._other_adds.append( robjects.r('scale_fill_grey()'))

    def theme_bw(self, base_size = None):
        kwargs = {}
        if base_size:
            kwargs['base_size'] = float(base_size)
        self._other_adds.append(robjects.r('theme_bw')(**kwargs))

    def set_base_size(self, base_size = 10):
        self.theme_bw(base_size = base_size)
        
    def add_label(self, text, xpos, ypos):
        import exptools
        data = exptools.DataFrame.DataFrame({'x': [xpos], 'y': [ypos], 'text': [text]})
        self._other_adds.append(
            self.r['geom_text'](
               robjects.r('aes(x=x, y=y, label=text)'),
               data,
                size=3,
                color="black"

            )
        )

    def scale_x_log_10(self):
        self.scale_x_continuous(trans = 'log10')

    def scale_x_continuous(self, breaks = None, minor_breaks = None, trans = None, limits=None, labels=None, expand=None):
        other_params = {}
        if not breaks is None:
            other_params['breaks'] = numpy.array(breaks)
        if not minor_breaks is None:
            other_params['minor_breaks'] = numpy.array(minor_breaks)
        if trans:
            other_params['trans'] = str(trans)
        if not limits is None:
            other_params['limits'] = numpy.array(limits)
        if not limits is None:
            other_params['limits'] = numpy.array(limits)
        if not labels is None:
            other_params['labels'] = numpy.array(labels)
        if not expand is None:
            other_params['expand'] = numpy.array(expand)
        if not breaks is None and not labels is None:
                if len(breaks) != len(labels):
                    raise ValueError("len(breaks) != len(labels)")

        self._other_adds.append(
            robjects.r('scale_x_continuous')(**other_params)
        )
    def scale_x_discrete(self, breaks = None, minor_breaks = None, trans = None, limits=None, labels=None, expand=None):
        other_params = {}
        if not breaks is None:
            other_params['breaks'] = numpy.array(breaks)
        if not minor_breaks is None:
            other_params['minor_breaks'] = numpy.array(minor_breaks)
        if trans:
            other_params['trans'] = str(trans)
        if not limits is None:
            other_params['limits'] = numpy.array(limits)
        if not limits is None:
            other_params['limits'] = numpy.array(limits)
        if not labels is None:
            other_params['labels'] = numpy.array(labels)
        if not expand is None:
            other_params['expand'] = numpy.array(expand)
        if not breaks is None and not labels is None:
                if len(breaks) != len(labels):
                    raise ValueError("len(breaks) != len(labels)")

        self._other_adds.append(
            robjects.r('scale_x_discrete')(**other_params)
        )

    def scale_y_continuous(self, breaks = None, minor_breaks = None, trans = None, limits=None, labels=None, expand=None, name = None):
        other_params = {}
        if not breaks is None:
            other_params['breaks'] = numpy.array(breaks)
        if not minor_breaks is None:
            other_params['minor_breaks'] = numpy.array(minor_breaks)
        if not trans is None:
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
            robjects.r('scale_y_continuous')(**other_params)
        )

    def scale_y_continuous(self, breaks = None, minor_breaks = None, trans = None, limits=None, labels=None, expand=None, name = None):
        other_params = {}
        if not breaks is None:
            other_params['breaks'] = numpy.array(breaks)
        if not minor_breaks is None:
            other_params['minor_breaks'] = numpy.array(minor_breaks)
        if not trans is None:
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
            robjects.r('scale_y_continuous')(**other_params)
        )

    def scale_x_reverse(self):
        self._other_adds.append(robjects.r('scale_x_reverse()'))

    def scale_y_reverse(self):
        self._other_adds.append(robjects.r('scale_y_reverse()'))

    def turn_x_axis_labels(self,  angle=75, hjust=1, size=8, vjust=0):
        kargs = {
            'axis.text.x': robjects.r('theme_text')(angle = angle, hjust=hjust, size=size, vjust=0)
        }
        self._other_adds.append( robjects.r('opts')(**kargs))

    def hide_background(self):
        self._other_adds.append(robjects.r('opts')(**{'panel.background': robjects.r('theme_blank()')}))

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

    def scale_fill_manual(self, list_of_colors):
        self._other_adds.append(robjects.r('scale_fill_manual')(values = numpy.array(list_of_colors)))

    def scale_fill_brewer(self, name = None, palette = 'Set1'):
        other_params = {}
        if not name is None:
            other_params['name'] = name
        if not palette is None:
            other_params['palette'] = palette
        self._other_adds.append(robjects.r('scale_fill_brewer')(**other_params))

    def scale_fill_hue(self, h = None, l = None, c = None, limits = None, breaks = None, labels = None, h_start = None, direction = None):
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
        self._other_adds.append(robjects.r('scale_fill_hue')(**other_params))



    def coord_flip(self):
        self._other_adds.append(robjects.r('coord_flip()'))

    def coord_polar(self, theta="x", start=0, direction=1, expand=False):
        self._other_adds.append(robjects.r('coord_polar')(
            theta = theta,
            start = start,
            direction = direction,
            expand = expand))


    def legend_position(self, value):
        if type(value) is tuple:
            self._other_adds.append(robjects.r('opts(legend.position = c(%i,%i))' % value))
        else:
            self._other_adds.append(robjects.r('opts(legend.position = "%s")' % value))

    def hide_legend(self):
        self.legend_position('none')

    def hide_panel_border(self):
        self._other_adds.append(robjects.r('opts(panel.border=theme_rect(fill=NA, colour=NA))'))

    def set_axis_color(self, color=None):
        if color is None:
            self._other_adds.append(robjects.r('opts(axis.line = theme_segment())'))
        else:
            self._other_adds.append(robjects.r('opts(axis.line = theme_segment(colour = "%s"))' % color))

    def hide_grid(self):
        self._other_adds.append(robjects.r('opts(panel.grid.major = theme_line(colour = NA))'))
        self._other_adds.append(robjects.r('opts(panel.grid.minor = theme_line(colour = NA))'))

    def hide_grid_minor(self):
        #self._other_adds.append(robjects.r('opts(panel.grid.major = theme_line(colour = NA))'))
        self._other_adds.append(robjects.r('opts(panel.grid.minor = theme_line(colour = NA))'))

    def smaller_margins(self):
        self._other_adds.append(robjects.r('opts(panel.margin = unit(0.0, "lines"))'))
        self._other_adds.append(robjects.r('opts(axis.ticks.margin = unit(0.0, "cm"))'))
        self.plot_margin(0,0,0,0)

    def plot_margin(self, top, left, bottom, right):
        self._other_adds.append(robjects.r('opts(plot.margin = unit(c(%i,%i,%i,%i), "lines"))' % (top, left, bottom, right)))


    def scale_shape_manual(self, values):
        self._other_adds.append(robjects.r('scale_shape_manual')(values=values))

    def scale_colour_manual(self, values):
        self._other_adds.append(robjects.r('scale_colour_manual')(values=numpy.array(values)))

    def scale_color_brewer(self, name = None, palette = 'Set1'):
        other_params = {}
        if not name is None:
            other_params['name'] = name
        if not palette is None:
            other_params['palette'] = palette
        self._other_adds.append(robjects.r('scale_colour_brewer')(**other_params))


    def scale_colour_grey(self):
        self._other_adds.append(robjects.r('scale_colour_grey')())

def plot_top_k_overlap(lists, output_filename, until_which_k = sys.maxint):
    if exptools.output_file_exists(output_filename):
        return
    for s in lists:
        until_which_k = min(len(s), until_which_k)
    max_k = until_which_k
    first = lists[0]
    lists = lists[1:]
    plot_data = {"k": [], 'overlap': []}
    print 'building overlap assoc plot'
    tr = exptools.TimeRemainingGuestimator(max_k)
    for k in xrange(1, max_k + 1):
        plot_data['k'].append(k)
        my_set = set(first[:k])
        for l in lists:
            my_set = my_set.intersection(set(l[:k]))
        plot_data['overlap'].append(len(my_set) / float(k))
        tr.step()
    tr.finished()
    plot = Plot(exptools.DF.DataFrame(plot_data), 'k', 'overlap')
    plot.add_line('k','overlap')
    plot.render(output_filename)

def plot_venn(sets, output_filename, width=8, height=8):
    df = pyvenn.VennDiagram(sets)
    df.plot_normal(output_filename, width)

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def intersection(list_of_sects):
    if not list_of_sects:
        return set() 
    final_set = list_of_sects[0]
    for k in list_of_sects[1:]:
        final_set = final_set.intersection(k)
    return final_set

def union(list_of_sects):
    if not list_of_sects:
        return set() 
    final_set = list_of_sects[0]
    for k in list_of_sects[1:]:
        final_set = final_set.union(k)
    return final_set 

def _no_annotation(set_name, set_entries):
    return { set_name: set_entries}

def venn_to_dataframes(sets, annotator = None, ordered = None):
    if annotator is None:
        annotator = _no_annotation
    dfs = {}
    if ordered is None:
        ordered = sets.keys()
        ordered.sort()
    sets = dict((k,set(v)) for (k, v) in sets.items())
    one_letter_names = []
    current_letter = 'A'
    for name in ordered:
        one_letter_names.append(current_letter)
        current_letter = chr(ord(current_letter) + 1)

    for subset in powerset(sets.keys()):
        if not subset:
            continue
        not_in_set = [x for x in sets.keys() if not x in subset]
        name_of_subset = [] 
        long_name = []
        for one_letter, name in zip(one_letter_names, ordered):
            if name in subset:
                name_of_subset.append(one_letter)
                long_name.append(name)
            else:
                name_of_subset.append("~%s" % one_letter)
                long_name.append("(NOT %s)" % name)
        name_of_subset = "".join(name_of_subset)
        long_name = " AND ".join(long_name)
        #print 'name_of_subset', name_of_subset
        actual_set = intersection([sets[x] for x in subset]).difference(union([sets[x] for x in not_in_set]))
        data = annotator(long_name, actual_set)
        df = exptools.DF.DataFrame(data, [long_name])
        dfs[name_of_subset] = df
    overview = {"Short name": [], 'Set name': []} 
    for one_letter, name in zip(one_letter_names, ordered):
        overview["Short name"].append(one_letter)
        overview["Set name"].append(name)
    dfs['Overview'] = exptools.DF.DataFrame(overview)
    return dfs

def dataframes_to_venn(dfs):
    sets = OrderedDict()
    lookup = {}
    for row in dfs['Overview'].iter_rows():
        lookup[row['Short name']] = row['Set name']
    for set_name in dfs:
        if set_name == 'Overview':
            continue
        goes_where = re.sub("~[A-Z]", '', set_name)
        to_add = set(dfs[set_name].get_column_view(0)) 
        for letter in goes_where:
            if not lookup[letter] in sets:
                sets[lookup[letter]] = set()
            sets[lookup[letter]].update(to_add)
    return sets

def dump_venn(sets, output_filename, annotator = None):
    dfs = venn_to_dataframes(sets, annotator)
    exptools.DF.DF2Excel().write(dfs, output_filename)


def PlotPiechartHistogram(df, column_name, include_counts = True, include_percentage = False):
    df = df.copy()
    column = df.get_column_view(column_name)
    if isinstance(column, exptools.DF.factors.Factor):
        ordered = column.levels
    else:
        ordered = list(sorted(df.get_column_unique(column_name)))
    positions = []
    counts = []
    current = 0
    for value in ordered:
        cc = numpy.sum([column == value])
        counts.append(cc)
        positions.append(current + cc / 2.0)
        current += cc
    total = current
    labels = []
    for cc in counts:
        if include_counts and include_percentage:
            l = "%i\n(%.2f%%)" % (cc, float(cc) / total * 100)
        elif include_counts:
            l = "%i" % cc
        elif include_percentage:
            l = "%.2f%%" % (float(cc) / total * 100,)
        else:
            l = ""
        labels.append(l)


    count_column = 'count'
    while count_column in df.columns_ordered:
        count_column += 'c'
    plot_df = exptools.DF.DataFrame({
        'Dummy': [1] * len(counts), 
        'Counts': counts,
        'Values': [str(x) for x in ordered],
        'Labels': labels,
        'y': positions
    })
    plot = Plot(plot_df, 'Dummy')
    plot.add_bar_plot("Dummy", 'Counts', fill="Values", position="stack")
    if include_percentage or include_counts:
        plot.add_text('Dummy','y', 'Labels')
    plot.coord_polar(theta='y')
    plot.hide_x_axis_labels()
    plot.hide_x_axis_title()
    plot.hide_axis_ticks()
    plot.hide_y_axis_labels()
    plot.hide_y_axis_title()
    return plot



 

def doGGBarPlot(dataframe,title, xaxis, yaxis, color, facet, output_filename):
    load_r()
    robjects.r('library(ggplot2)')
    ggplot = robjects.r['ggplot']
    aes = robjects.r['aes']
    add = robjects.r('ggplot2:::"+.ggplot"')
    layer = robjects.r['layer']
    facet_wrap = robjects.r['facet_wrap']
    ggsave = robjects.r['ggsave']
    plot = ggplot(dataframe, robjects.r('aes(%s, %s,fill=%s)' % (xaxis, yaxis, color )))
    plot = add(plot, robjects.r('theme_bw()'))
    plot = add(plot, layer(geom="bar",stat='identity', position='dodge'))
    plot = add(plot, facet_wrap(robjects.r('~ %s'% facet), scale='free_x'))
    plot = add(plot, robjects.r('scale_x_discrete()'))
    plot = add(plot, robjects.r('scale_fill_brewer("Set1")'))
    #plot = add(plot, robjects.r('scale_fill_grey()'))
    #robjects.r('grob = ggplotGrob(%s)' % plot)
    plot = add(plot, robjects.r('opts(axis.ticks.x = theme_blank())'))
    plot = add(plot, robjects.r('opts(axis.text.x = theme_blank())'))
    #robjects.r('grid.gedit(gPath("xaxis", "labels"), gp=gpar(fontsize=6))')
    plot = add(plot, robjects.r('opts(title = "%s")' %  title))
    ggsave(filename=output_filename,plot=plot, width=10, height=10)


def doHistogramPlot(dataframe,title, xaxis, color= None,group=None,  facet = None,position='dodge',  output_filename = False):
    load_r()
    if not output_filename:
        raise ValueError("You have to specify an output_filename")
    robjects.r('library(ggplot2)')
    ggplot = robjects.r['ggplot']
    aes = robjects.r['aes']
    add = robjects.r('ggplot2:::"+.ggplot"')
    layer = robjects.r['layer']
    facet_wrap = robjects.r['facet_wrap']
    ggsave = robjects.r['ggsave']

    aes_params = []
    aes_params.append('%s' % xaxis)


    if color:
        aes_params.append('fill=%s'  % color)
    if group:
        aes_params.append('group=%s' % group)
    aes_params = ", ".join(aes_params)
    plot = ggplot(dataframe, robjects.r('aes(%s)' % (aes_params, )))
    #plot = add(plot, robjects.r('theme_bw()'))
    #plot = add(plot, robjects.r('scale_x_discrete()'))
    plot = add(plot, layer(geom="bar", stat='bin', position=position))
    if (facet):
        plot = add(plot, facet_wrap(robjects.r('~ %s'% facet), scale='free_x'))
    #plot = add(plot, robjects.r('scale_fill_brewer("Set1")'))
    #plot = add(plot, robjects.r('scale_fill_grey()'))
    #robjects.r('grob = ggplotGrob(%s)' % plot)
    #plot = add(plot, robjects.r('opts(axis.ticks.x = theme_blank())'))
    #plot = add(plot, robjects.r('opts(axis.text.x = theme_blank())'))
    #robjects.r('grid.gedit(gPath("xaxis", "labels"), gp=gpar(fontsize=6))')
    plot = add(plot, robjects.r('opts(title = "%s")' %  title))
    ggsave(filename=output_filename,plot=plot, width=8, height=6)

def doScatterPlot(dataframe, title, xaxis, yaxis, color=None, output_filename = False):
    load_r()
    if not output_filename:
        raise ValueError("You have to specify an output_filename")
    robjects.r('library(ggplot2)')
    ggplot = robjects.r['ggplot']
    aes = robjects.r['aes']
    add = robjects.r('ggplot2:::"+.ggplot"')
    layer = robjects.r['layer']
    facet_wrap = robjects.r['facet_wrap']
    ggsave = robjects.r['ggsave']

    aes_params = []
    aes_params.append('x=%s' % xaxis)
    aes_params.append('y=%s' % yaxis)
    if color:
        aes_params.append('color=%s'  % color)
    aes_params = ", ".join(aes_params)

    plot = ggplot(dataframe, robjects.r('aes(%s)' % (aes_params, )))
    plot = add(plot, layer(geom="point"))
    if color:
        plot = add(plot, robjects.r('scale_colour_gradient()'))
    #plot = add(plot, robjects.r('scale_fill_brewer("Set1")'))
    #plot = add(plot, robjects.r('opts(title = "%s")' %  title))
    ggsave(filename=output_filename,plot=plot, width=8, height=6)

def plotArray(numpy_array, title, xaxis_name = 'x', yaxis_name = 'y', xaxis_offset = 0, color=None,  output_filename = False):
    load_r()
    if not output_filename:
        raise ValueError("You have to specify an output_filename")
    xs = []
    ys = []
    for (x, y) in enumerate(numpy_array):
        xs.append(x + xaxis_offset)
        ys.append(y)
    import exptools
    df = exptools.DataFrame.DataFrame({xaxis_name: xs, yaxis_name: ys})
    robjects.r('library(ggplot2)')
    ggplot = robjects.r['ggplot']
    aes = robjects.r['aes']
    add = robjects.r('ggplot2:::"+.ggplot"')
    layer = robjects.r['layer']
    facet_wrap = robjects.r['facet_wrap']
    ggsave = robjects.r['ggsave']

    aes_params = []
    aes_params.append('x=%s' % xaxis_name)
    aes_params.append('y=%s' % yaxis_name)
    if color:
        aes_params.append('color=%s'  % color)
    aes_params = ", ".join(aes_params)
    plot = ggplot(dataframe, robjects.r('aes(%s)' % (aes_params, )))
    plot = add(plot, layer(geom="point"))
    if color:
        plot = add(plot, robjects.r('scale_colour_gradient()'))
    ggsave(filename=output_filename)

def doJitterPlot(dataframe, title, xaxis, yaxis, group=None, color=None, size=None,shape=None, output_filename = False):
    load_r()
    if not output_filename:
        raise ValueError("You have to specify an output_filename")
    robjects.r('library(ggplot2)')
    ggplot = robjects.r['ggplot']
    aes = robjects.r['aes']
    add = robjects.r('ggplot2:::"+.ggplot"')
    layer = robjects.r['layer']
    facet_wrap = robjects.r['facet_wrap']
    position_jitter = robjects.r['position_jitter']
    ggsave = robjects.r['ggsave']

    aes_params = []
    aes_params.append('x=%s' % xaxis)
    aes_params.append('y=%s' % yaxis)
    if color:
        aes_params.append('color=%s'  % color)
    if group:
        aes_params.append('group=%s' % group)
    if size:
        aes_params.append('size=%s' % size)
    if shape:
        aes_params.append('shape=%s' % shape)
        
    aes_params = ", ".join(aes_params)

    plot = ggplot(dataframe, robjects.r('aes(%s)' % (aes_params, )))
    plot = add(plot, layer(geom="jitter", position=position_jitter(width=0.5)))
 #   if color:
  #      plot = add(plot, robjects.r('scale_colour_gradient()'))
    #plot = add(plot, robjects.r('scale_fill_brewer("Set1")'))
    #plot = add(plot, robjects.r('opts(title = "%s")' %  title))
    ggsave(filename=output_filename,plot=plot, width=8, height=6)

