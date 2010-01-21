import exptools
import rpy2.robjects as robjects
import numpy
robjects.r('library(ggplot2)')
robjects.r('library(Vennerable)')

class Plot:

    def __init__(self, dataframe, xaxis='X', yaxis=None):
        self.r = {}
        self.r['ggplot'] = robjects.r['ggplot']
        self.r['aes'] = robjects.r['aes']
        self.r['add'] = robjects.r('ggplot2:::"+.ggplot"')
        self.r['layer'] = robjects.r['layer']
        self.r['facet_wrap'] = robjects.r['facet_wrap']
        self.r['geom_text'] = robjects.r['geom_text']
        self.r['ggsave'] = robjects.r['ggsave']
        self.dataframe = dataframe.copy()
        self.old_names = self.dataframe.columns_ordered[:]
        self.lab_rename = {}
        for ii in xrange(0, len(self.old_names)):
            self.dataframe.rename_column(self.old_names[ii], 'dat_%s' % ii)
        self._aesthetics = {} 
        self._aesthetics['x'] = xaxis
        if yaxis:
            self._aesthetics['y'] = yaxis
        self._other_adds = []

    def render(self, output_filename, width=8, height=6):
        aes = self._build_aesthetic(self._aesthetics)
        plot = self.r['ggplot'](self.dataframe, aes)
        for obj in self._other_adds:
            plot = self.r['add'](plot, obj)
        for name, value in self.lab_rename.items():
            plot = self.r['add'](plot, robjects.r('labs(%s = "%s")' % (name, value)))
        #plot = self.r['add'](plot, self.r['layer'](geom="point"))
        #robjects.r('options( error=recover )')
        self.r['ggsave'](filename=output_filename,plot=plot, width=width, height=height, dpi=300)

    def add_aesthetic(self, name, column_name):
        self._aesthetics[name] = column_name

    def add_scatter(self, x_column, y_column, color=None, group=None, shape=None, size=None):
        aes_params = {'x': x_column, 'y': y_column}
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        if shape:
            aes_params['shape'] = shape
        if size:
            aes_params['size'] = size
        self._other_adds.append(robjects.r('geom_point')(self._build_aesthetic(aes_params)))
        return
        self._other_adds.append(self.r['layer'](geom="point"))
        self.add_aesthetic('x',x_column)
        self.add_aesthetic('y',y_column)

    def add_jitter(self, x_column, y_column, color=None):
        aes_params = {'x': x_column, 'y': y_column}
        if color:
            aes_params['colour'] = color
        self._other_adds.append(robjects.r('geom_jitter')(self._build_aesthetic(aes_params)))

    def add_stacked_bar_plot(self, x_column, y_column, fill):
        aes_params  = {'x': x_column, 'y': y_column, 'fill': fill}
        self._other_adds.append(robjects.r('geom_bar')(self._build_aesthetic(aes_params), position='stack'))

    #def add_bar_plot(self, x_column, y_column, fill, position = 'dodge'):
        #aes_params  = {'x': x_column, 'y': y_column, 'fill': fill}
        #self._other_adds.append(robjects.r('geom_bar')(self._build_aesthetic(aes_params), position=position,
                                                      #stat='identity'))

    def add_histogram(self, x_column, y_column = "..count..", color=None, group = None, fill=None, position="dodge"):
        aes_params = {'x': x_column}
        if fill:
            aes_params['fill'] = fill
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        stat = robjects.r['stat_bin']()
            #x = x_column, y = y_column)
        self._other_adds.append(
            robjects.r('geom_bar')(self._build_aesthetic(aes_params), stat=stat, position=position)
        )

    def add_bar_plot(self, x_column, y_column, color=None, group = None, fill=None, position="dodge"):
        aes_params = {'x': x_column, 'y': y_column}
        if fill:
            aes_params['fill'] = fill
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        self._other_adds.append(
            robjects.r('geom_bar')(self._build_aesthetic(aes_params), stat="identity", position=position)
        )

    def add_box_plot(self, x_column, y_column):
        aes_params = {'x': x_column, 'y': y_column}
        self._other_adds.append(
            robjects.r('geom_boxplot')(self._build_aesthetic(aes_params))
        )

    def add_heatmap(self, fill):
        aes_params = {}
        aes_params['fill'] = fill
        self._other_adds.append(
            robjects.r('geom_tile')(self._build_aesthetic(aes_params), stat="identity")
        )
        self._other_adds.append(
            robjects.r('scale_fill_gradient2(low="red", mid="white", high="blue", midpoint=0)')
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


    def _build_aesthetic(self, params):
        aes_params = []
        for aes_name, aes_column in params.items():
            if aes_column in self.old_names:
                new_name = 'dat_%s'  % self.old_names.index(aes_column)
                aes_params.append('%s=%s' % (aes_name, new_name))
                self._fix_axis_label(aes_name, new_name, aes_column)
            else: #a fixeud value
                aes_params.append("%s=%s" % (aes_name, aes_column))
        aes_params = ", ".join(aes_params)
        return robjects.r('aes(%s)' % aes_params)


    def add_line(self, x_column, y_column, color=None, group=None, shape=None, alpha=1.0):
        aes_params = {'x': x_column, 'y': y_column}
        other_params = {}
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        if shape:
            aes_params['shape'] = shape
        if type(alpha) == int or type(alpha) == float:
            other_params['alpha'] = alpha
        else:
            aes_params['alpha'] = str(alpha)
        self._other_adds.append(robjects.r('geom_line')(self._build_aesthetic(aes_params), **other_params))
        #self.add_aesthetic('x',x_column)
        ###self.add_aesthetic('y',y_column)

    def add_ab_line(self, intercept, slope):
        self._other_adds.append(robjects.r('geom_abline(intercept=%f, slope=%f)' % (intercept, slope)))

    def add_density(self, x_column, color = None):
        """add a kernel estimated density plot - gauss kernel and bw.SJ estimation of bandwith"""
        aes_params = {'x': x_column}
        if color:
            aes_params['colour'] = color
        self._other_adds.append(robjects.r('geom_density')(
            self._build_aesthetic(aes_params),
            bw = robjects.r('bw.SJ')(self.dataframe.get_column_view(self.old_names.index(x_column)))
            )
        )


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
            facet_specification = '%s ~ %s' % (column_one, column_two)
        else:
            facet_specification = '~ %s' % (column_one,)
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
        
    def add_text(self, text, xpos, ypos):
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
        self._other_adds.append(
            robjects.r('scale_x_continuous(trans="log10")')
        )

    def turn_x_axis_labels(self,  angle=75, hjust=1, size=8):
        kargs = {
            'axis.text.x': robjects.r('theme_text')(angle = angle, hjust=hjust, size=size)
        }
        self._other_adds.append( robjects.r('opts')(**kargs))

    def set_fill(self, list_of_colors):
        self._other_adds.append(robjects.r('scale_fill_manual')(values = numpy.array(list_of_colors)))

    def coord_flip(self):
        self._other_adds.append(robjects.r('coord_flip()'))

    def legend_position(self, value):
        if type(value) is tuple:
            self._other_adds.append(robjects.r('opts(legend.position = c(%i,%i))' % value))
        else:
            self._other_adds.append(robjects.r('opts(legend.position = "%s")' % value))

    def smaller_margins(self):
        self._other_adds.append(robjects.r('opts(panel.margin = unit(0.0, "lines"))'))
        self._other_adds.append(robjects.r('opts(plot.margin = unit(c(0,0,0,0), "lines"))'))
        self._other_adds.append(robjects.r('opts(axis.ticks.margin = unit(0.0, "cm"))'))

    def scale_shape_manual(self, values):
        self._other_adds.append(robjects.r('scale_shape_manual')(values=values))


def plot_venn(sets, output_filename, width=8, height=8):
    _venn_plot_weights(sets ,output_filename, width, height)
 
def _venn_plot_sets(sets, output_filename, width=8, height=8):
    """Plot a venn diagram into the pdf file output_filename.
    Takes a dictionary of sets and passes them straight on to R"""
    robjects.r('pdf')(output_filename, width=width, height=height)
    x = robjects.r('Venn')(Sets = [numpy.array(list(x)) for x in sets.values()], SetNames=sets.keys())
    robjects.r('plot')(x, **{'type': 'squares', 'doWeights': False})
    robjects.r('dev.off()')

def _venn_plot_weights(sets, output_filename, width=8, height=8):
    """Plot a venn diagram into the pdf file output_filename.
    Takes a dictionary of sets and does the intersection calculation in python
    (which hopefully is a lot faster than passing 10k set elements to R)
    (and anyhow, we have the smarter code)"""
    weights = [0]
    sets_by_power_of_two = {}
    for ii, kset in enumerate(sorted(sets.keys())):
        iset = sets[kset]
        sets_by_power_of_two[2**ii] = set(iset)
    for i in xrange(1, 2**len(sets)):
        sets_to_intersect = []
        to_exclude = set()
        for ii in xrange(0, len(sets)):
            if (i & (2**ii)):
                sets_to_intersect.append(sets_by_power_of_two[i & (2**ii)])
            else:
                to_exclude = to_exclude.union(sets_by_power_of_two[(2**ii)])
        final = set.intersection(*sets_to_intersect) - to_exclude
        weights.append( len(final))
    robjects.r('pdf')(output_filename, width=width, height=height)
    x = robjects.r('Venn')(Weight = numpy.array(weights), SetNames=sorted(sets.keys()))
    if len(sets) <= 3:
        venn_type = 'circles'
    else:
        venn_type = 'squares'

    robjects.r('plot')(x, **{'type': venn_type, 'doWeights': False})
    robjects.r('dev.off()')


 

def doGGBarPlot(dataframe,title, xaxis, yaxis, color, facet, output_filename):
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

