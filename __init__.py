import exptools
import itertools
from ordereddict import OrderedDict
import pydataframe
import sys
from hilbert import hilbert_plot, hilbert_to_image
import rpy2.robjects as robjects
exptools.load_software('pyvenn')
import pyvenn
from square_euler import SquareEuler
import re
from sequence_logos import plot_sequences, plot_sequence_alignment
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
        self.to_rename = {}

        self._add_geom_methods()


    def render(self, output_filename, width=8, height=6, dpi=300):
        #print self.dataframe
        try:
            plot = self.r['ggplot'](self.dataframe)
            for obj in self._other_adds:
                plot = self.r['add'](plot, obj)
            for name, value in self.lab_rename.items():
                plot = self.r['add'](plot, robjects.r('labs(%s = "%s")' % (name, value)))
            #plot = self.r['add'](plot, self.r['layer'](geom="point"))
            #robjects.r('options( error=recover )')
            self.r['ggsave'](filename=output_filename,plot=plot, width=width, height=height, dpi=dpi)
        except ValueError:
            print 'old names', self.old_names
            raise

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

    def _translate_params(self, params):
        """Translate between the original dataframe names and the numbered ones we assign
        to avoid r-parsing issues"""

        aes_params = []
        for aes_name, aes_column in params.items():
            if aes_column in self.old_names:
                new_name = 'dat_%s'  % self.old_names.index(aes_column)
                aes_params.append('%s=%s' % (aes_name, new_name))
                if aes_column in self.to_rename:
                    self._fix_axis_label(aes_name, new_name, self.to_rename[aes_column])
                else:
                    self._fix_axis_label(aes_name, new_name, aes_column)
            else: #a fixeud value
                aes_params.append("%s=%s" % (aes_name, aes_column))
        return aes_params

    def _fix_axis_label(self, aes_name, new_name, real_name):
        """Reapply the correct (or new) labels to the axis, overwriting our dat_%i numbered dataframe 
        columns"""
        which_legend = False
        import pypipegraph
        logger = pypipegraph.util.start_logging('pyggplot2')
        logger.info('fixing %s to %s' % (aes_name, real_name))
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
        """Tarnsform a pythhon list of aesthetics to the R aes() object"""
        aes_params = self._translate_params(params)
        aes_params = ", ".join(aes_params)
        return robjects.r('aes(%s)' % aes_params)

    def parse_param(self, name, value, required = True):
        """
        Transform parameters into either aes_params or other_params, 
        depending on whether they are in our df.
        if value is None, this parameter is ignored
        
        """
        if not value is None:
            if isinstance(value, tuple):
                new_name = value[1]
                value = value[0]
                self.to_rename[value] = new_name
            if value in self.old_names:
                self.aes_collection[name] = value
            else:
                self.other_collection[name] = value

    def reset_params(self, data):
        self.aes_collection = {}
        self.other_collection = {}
        if not data is None:
            self.other_collection['data'] = self._prep_dataframe(data)

    def _add(self, name, geom_name, required_mappings, optional_mappings, defaults, args, kwargs):
        """The generic method to add a geom to the ggplot.
        You need to call add_xyz (see _add_geom_methods for a list, with each variable mapping
        being one argument) with the respectivly required parameters (see ggplot documentation).
        You may optionally pass in an argument called data, which will replace the plot-global dataframe
        for this particular geom
        """
        mappings = {}
        all_defined_mappings = required_mappings + optional_mappings
        for a, b in zip(all_defined_mappings, args): #so that you could in thery also pass the optional_mappings by position...required_mappings
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
                    if hasattr(defaults[mapping], '__call__',):
                        mappings[mapping] = defaults[mapping](mappings)
                    else:
                        mappings[mapping] = defaults[mapping]
                else:
                    mappings[mapping] = None

        self.reset_params(data)
        for param in mappings:
            self.parse_param(param, mappings[param])

        self._other_adds.append(robjects.r(geom_name)(self._build_aesthetic(self.aes_collection), **self.other_collection))
        return self

    def _add_geom_methods(self):
        """add add_xyz methods for all geoms in ggplot.
        All geoms have required & optional attributes and take an optional data parameter with another
        dataframe
        """
        #python method name (add_ + name), geom (R) name, required attributes, optional attributes
        methods = (
                #geoms
                ('scatter', 'geom_point', ['x','y'], ['color', 'group', 'shape', 'size', 'alpha'], {}),
                ('jitter', 'geom_jitter', ['x','y'], ['color', 'group', 'shape', 'size', 'alpha', 'jitter_x', 'jitter_y'], {}),
                ('bar', 'geom_bar', ['x','y'], ['color','group', 'fill','position', 'stat'], {'position': 'dodge', 'stat': 'identity'}),
                ('box_plot', 'geom_boxplot', ['x','y'], ['color','group','fill', 'alpha'], {}),
                ('box_plot2', 'geom_boxplot', ['x','lower', 'middle','upper','ymin', 'ymax'], ['color','group','fill', 'alpha', 'stat'], {'stat': 'identity'}),
                ('line', 'geom_line', ['x','y'], ['color', 'group', 'shape', 'alpha', 'size'], {}),
                ('area', 'geom_area', ['x','y'], ['color','fill', 'linetype', 'alpha', 'size', 'position'], {}),
                ('ribbon', 'geom_ribbon', ['x', 'ymin', 'ymax'], ['color', 'fill', 'size', 'linetype', 'alpha', 'position'], {}),
                ('error_bars', 'geom_errorbar', ['x','ymin', 'ymax'], ['color', 'group', 'width', 'alpha'], {'width': 0.25}),
                ('error_barsh', 'geom_errorbarh', ['y','xmin', 'xmax'], ['color', 'group', 'width', 'alpha'], {'width': 0.25}),
                ('ab_line', 'geom_abline', ['intercept', 'slope'], ['alpha', 'size', 'color'], {}),
                ('density', 'geom_density', ['x'], ['y', 'color'], 
                    {'bw': lambda mappings: robjects.r('bw.SJ')(self.dataframe.get_column_view(self.old_names.index(mappings['x'])))}),
                ('density_2d', 'geom_density2d', ['x','y'], ['color', 'alpha'], {}),
                ('rect', 'geom_rect', ['xmin','xmax','ymin', 'ymax'], ['color', 'fill', 'alpha'], {}),
                ('tile', 'geom_tile', ['x','y'], ['color', 'fill', 'size', 'linetype', 'alpha'], {}),
                ('vertical_bar', 'geom_vline', ['xintercept'], ['alpha', 'color', 'size'], {'alpha': 0.5, 'color': 'black', 'size': 1}),
                ('horizontal_bar', 'geom_hline', ['yintercept'], ['alpha', 'color', 'size'], {'alpha': 0.5, 'color': 'black', 'size': 1}),
                ('segment', 'geom_segment', ['x', 'xend', 'y', 'yend'], ['color', 'alpha', 'size'], {'size': 0.5}),
                ('text', 'geom_text', ['x','y','label'], ['angle','alpha', 'size', 'hjust', 'vjust', 'fontface', 'color'], {}),



                #stast
                ('stat_sum_color', 'stat_sum', ['x','y'], ['size'], {'color': '..n..', 'size': 0.5}),
                ('stat_smooth', 'stat_smooth', [], ['method', 'se', 'x', 'y'], {"method": 'lm', 'se': True}),

                ('stacked_bar_plot','geom_bar', ['x','y','fill'], [], {'position': 'stack'}), #do we still need this?
                #"""A scatter plat that's colored by no of overlapping points"""
                )

        for (name, geom, required, optional, defaults) in methods:
            def define(name, geom, required , optional, defaults): #we need to capture the variables...
                def do_add(*args, **kwargs):
                    return self._add(name, geom, required, optional, defaults, args, kwargs)
                return do_add
            setattr(self, 'add_' + name, define(name, geom, required, optional, defaults))


    def add_histogram(self, x_column, y_column = "..count..", color=None, group = None, fill=None, position="dodge", add_text = False, bin_width = None, alpha = None):
        aes_params = {'x': x_column}
        other_params = {}
        stat_params = {}
        if fill:
            aes_params['fill'] = fill
        if color:
            aes_params['colour'] = color
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
            self._other_adds.append(
                robjects.r('geom_histogram')(self._build_aesthetic(aes_params), **other_params)
            )
        if add_text:
            self._other_adds.append(
                robjects.r('geom_text')(
                    self._build_aesthetic({'x': x_column, 'y':'..count..', 'label':'..count..'}),stat='bin' ))
        return self

    def add_cummulative(self, x_column, ascending = True):
        """Add a line showing cumulative % of data <= x"""
        total = 0
        current = 0
        try:
            column_name = 'dat_%s'  % self.old_names.index(x_column)
        except ValueError:
            raise ValueError("Could not find column %s, available: %s" % (x_column, self.old_names))
        column_data = self.dataframe.get_column(column_name) #explicit copy!
        column_data .sort()
        x_values = []
        y_values = []
        for value, group in itertools.groupby(column_data):
            x_values.append(value)
            y_values.append(len(list(group)))
        data = pydataframe.DataFrame({x_column: x_values, '% <=': y_values})
        return self.add_line(x_column, '% <=', data=data)




    def add_heatmap(self, x_column, y_column, fill, low="red", mid=None, high="blue", midpoint=0):
        aes_params = {'x': x_column, 'y': y_column}
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
        return self











    def set_title(self, title):
        self._other_adds.append(robjects.r('opts(title = "%s")' %  title))






 
    def facet(self, column_one, column_two = None, fixed_x = True, fixed_y = True, ncol=None):
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
            params = self._translate_params({column_one: column_two})[0]
            facet_specification = params.replace('=', '~')
            #facet_specification = '%s ~ %s' % (column_one, column_two)
        else:
            params = self._translate_params({"":column_one})[0]
            facet_specification = params.replace('=', '~') 
            #facet_specification = '~ %s' % (column_one,)
        params = {
            'scale': scale}
        if ncol:
            params['ncol'] = ncol
        self._other_adds.append(facet_wrap(robjects.r(facet_specification), **params))

    def facet_grid(self, column_one, column_two = None, fixed_x = True, fixed_y = True, ncol=None):
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
            new_one = 'dat_%s'  % self.old_names.index(column_one)
            new_two = 'dat_%s'  % self.old_names.index(column_two)
            facet_specification = '%s ~ %s' % (new_one, new_two)
        else:
            params = self._translate_params({"":column_one})[0]
            facet_specification = '. ' + params.replace('=', '~')
            #facet_specification = '~ %s' % (column_one,)
        params = {
            'scale': scale}
        if ncol:
            params['ncol'] = ncol
        self._other_adds.append(robjects.r('facet_grid')(robjects.r(facet_specification), **params))



    def greyscale(self):
        self._other_adds.append( robjects.r('scale_colour_grey()'))
        self._other_adds.append( robjects.r('scale_fill_grey()'))

    def theme_bw(self, base_size = None):
        kwargs = {}
        if base_size:
            kwargs['base_size'] = float(base_size)
        self._other_adds.append(robjects.r('theme_bw')(**kwargs))


    def theme_darktalk(self, base_size = None):
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
            legend.text = theme_text(size = base_size * 1, colour="white"), 
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
        plot.title = theme_text(size = base_size * 1.2,colour="white"), plot.margin = unit(c(1, 1, 0.5, 0.5), "lines")), 

        class = "options")
}

""")
        self._other_adds.append(robjects.r('theme_darktalk')(**kwargs))

    def theme_talk(self, base_size = None):
        kwargs = {}
        if base_size:
            kwargs['base_size'] = float(base_size)
        self._other_adds.append(robjects.r('theme_talk')(**kwargs))

    def set_base_size(self, base_size = 10):
        self.theme_bw(base_size = base_size)
        
    def add_label(self, text, xpos, ypos, size=8, color=None, alpha = None):
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
        self.scale_x_continuous(trans = 'log10')
        return self

    def scale_x_continuous(self, breaks = None, minor_breaks = None, trans = None, limits=None, labels=None, expand=None, formatter = None, name = None):
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
        if not formatter is None:
            other_params['formatter'] = formatter
        if not name is None:
            other_params['name'] = name

        self._other_adds.append(
            robjects.r('scale_x_continuous')(**other_params)
        )
        return self

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
        return self

    def scale_y_continuous(self, breaks = None, minor_breaks = None, trans = None, limits=None, labels=None, expand=None, name = None):
        other_params = {}
        if not breaks is None:
            if breaks != 'NA':
                other_params['breaks'] = numpy.array(breaks)
            else:
                other_params['breaks'] = robjects.r("NA")
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
        return self

    def scale_x_reverse(self):
        self._other_adds.append(robjects.r('scale_x_reverse()'))
        return self

    def scale_y_reverse(self):
        self._other_adds.append(robjects.r('scale_y_reverse()'))
        return self

    def turn_x_axis_labels(self,  angle=75, hjust=1, size=8, vjust=0):
        kargs = {
            'axis.text.x': robjects.r('theme_text')(angle = angle, hjust=hjust, size=size, vjust=0)
        }
        self._other_adds.append( robjects.r('opts')(**kargs))
        return self

    def turn_y_axis_labels(self,  angle=75, hjust=1, size=8, vjust=0):
        kargs = {
            'axis.text.y': robjects.r('theme_text')(angle = angle, hjust=hjust, size=size, vjust=0)
        }
        self._other_adds.append( robjects.r('opts')(**kargs))
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


    def scale_fill_gradient(self, low, high, mid = None,midpoint = None, name = None, space = 'rgb', breaks = None, labels = None, limits = None, trans = None):
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
        if mid is not None:
            self._other_adds.append(robjects.r('scale_fill_gradient2')(**other_params))
        else:
            raise ValueError("Gradient 1")
            self._other_adds.append(robjects.r('scale_fill_gradient')(**other_params))
        return self



    def coord_flip(self):
        self._other_adds.append(robjects.r('coord_flip()'))
        return self

    def coord_polar(self, theta="x", start=0, direction=1, expand=False):
        self._other_adds.append(robjects.r('coord_polar')(
            theta = theta,
            start = start,
            direction = direction,
            expand = expand))
        return self


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
 #       self._other_adds.append(robjects.r('opts(panel.grid.minor = theme_line(colour = NA))'))
        self._other_adds.append(robjects.r('opts(panel.grid.minor = theme_blank())'))

    def smaller_margins(self):
        self._other_adds.append(robjects.r('opts(panel.margin = unit(0.0, "lines"))'))
        self._other_adds.append(robjects.r('opts(axis.ticks.margin = unit(0.0, "cm"))'))
        self.plot_margin(0,0,0,0)


    def plot_margin(self, top, left, bottom, right):
        self._other_adds.append(robjects.r('opts(plot.margin = unit(c(%i,%i,%i,%i), "lines"))' % (top, left, bottom, right)))


    def scale_shape_manual(self, values):
        self._other_adds.append(robjects.r('scale_shape_manual')(values=values))
    def scale_shape(self, solid = True):
        self._other_adds.append(robjects.r('scale_shape')(solid=solid))

    def scale_colour_manual(self, values):
        self._other_adds.append(robjects.r('scale_colour_manual')(values=numpy.array(values)))

    def scale_color_identity(self):
        self._other_adds.append(robjects.r('scale_colour_identity')())

    def scale_color_hue(self):
        self._other_adds.append(robjects.r('scale_colour_hue')())

    def scale_color_brewer(self, name = None, palette = 'Set1'):
        other_params = {}
        if not name is None:
            other_params['name'] = name
        if not palette is None:
            other_params['palette'] = palette
        self._other_adds.append(robjects.r('scale_colour_brewer')(**other_params))


    def scale_colour_grey(self):
        self._other_adds.append(robjects.r('scale_colour_grey')())

    def scale_fill_grey(self):
        self._other_adds.append(robjects.r('scale_fill_grey')())

def plot_top_k_overlap(lists, output_filename, until_which_k = sys.maxint):
    if exptools.output_file_exists(output_filename):
        return
    for s in lists:
        until_which_k = min(len(s), until_which_k)
    max_k = until_which_k
    first = lists[0]
    lists = lists[1:]
    plot_data = {"k": [], 'overlap': []}
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

def plot_venn(sets, output_filename, width=8, height=8, proportional = False):
    df = pyvenn.VennDiagram(sets)
    if proportional:
        df.plot_proportional(output_filename, width)
    else:
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
    label_positions = []
    for ii, cc in enumerate(counts):
        if include_counts and include_percentage:
            l = "%i (%.2f%%)" % (cc, float(cc) / total * 100)
        elif include_counts:
            l = "%i" % cc
        elif include_percentage:
            l = "%.2f%%" % (float(cc) / total * 100,)
        else:
            l = ""
        labels.append(l)
        label_positions.append([2, 2.25, 2.5][ii % 3])


    count_column = 'count'
    while count_column in df.columns_ordered:
        count_column += 'c'
    plot_df = exptools.DF.DataFrame({
        'Dummy': [1] * len(counts), 
        'Counts': counts,
        'Values': [str(x) for x in ordered],
        'Labels': labels,
        'Label_positions': label_positions,
        'y': positions
    })
    plot = Plot(plot_df, 'Dummy')
    plot.add_bar("Dummy", 'Counts', fill="Values", position="stack")
    if include_percentage or include_counts:
        plot.add_text('Label_positions','y', 'Labels', color="Values")
        #plot.add_segment(0, 'Label_positions', 1, 'y', color="Values")
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
    df = pydataframe.DataFrame({xaxis_name: xs, yaxis_name: ys})
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

