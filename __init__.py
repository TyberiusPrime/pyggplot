import exptools
import rpy2.robjects as robjects
robjects.r('library(ggplot2)')

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
        self.dataframe = dataframe
        self._aesthetics = {} 
        self._aesthetics['x'] = xaxis
        if yaxis:
            self._aesthetics['y'] = yaxis
        self._other_adds = []

    def render(self, output_filename, width=8, height=6):
        aes_params = []
        for aes_name, aes_column in self._aesthetics.items():
            aes_params.append('%s=%s' % (aes_name, aes_column))
        aes_params = ", ".join(aes_params)
        plot = self.r['ggplot'](self.dataframe, robjects.r('aes(%s)' % (aes_params,)))
        for obj in self._other_adds:
            plot = self.r['add'](plot, obj)
        #plot = self.r['add'](plot, self.r['layer'](geom="point"))
        #robjects.r('options( error=recover )')
        self.r['ggsave'](filename=output_filename,plot=plot, width=width, height=height)

    def add_aesthetic(self, name, column_name):
        self._aesthetics[name] = column_name

    def add_scatter(self, x_column, y_column, color=None, group=None):
        aes_params = {'x': x_column, 'y': y_column}
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        print self._build_aesthetic(aes_params)
        self._other_adds.append(robjects.r('geom_point')(self._build_aesthetic(aes_params)))
        return
        self._other_adds.append(self.r['layer'](geom="point"))
        self.add_aesthetic('x',x_column)
        self.add_aesthetic('y',y_column)

    def add_stacked_bar_plot(self, x_column, y_column, fill):
        aes_params  = {'x': x_column, 'y': y_column, 'fill': fill}
        self._other_adds.append(robjects.r('geom_bar')(self._build_aesthetic(aes_params), position='stack'))

    def add_histogram(self, x_column, color=None, group = None, fill=None, position="dodge"):
        aes_params = {'x': x_column}
        if fill:
            aes_params['fill'] = fill
        if color:
            aes_params['colour'] = color
        if group:
            aes_params['group'] = group
        self._other_adds.append(
            robjects.r('geom_bar')(self._build_aesthetic(aes_params), stat="bin", position=position)
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

    def _build_aesthetic(self, params):
        aes_params = []
        for aes_name, aes_column in params.items():
            aes_params.append('%s=%s' % (aes_name, aes_column))
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
        print self._build_aesthetic(aes_params)
        print other_params
        self._other_adds.append(robjects.r('geom_line')(self._build_aesthetic(aes_params), **other_params))
        #self.add_aesthetic('x',x_column)
        ###self.add_aesthetic('y',y_column)

    def set_title(self, title):
        self._other_adds.append(robjects.r('opts(title = "%s")' %  title))

    def add_vertical_bar(self, xpos, alpha=0.5):
        self._other_adds.append(
            robjects.r('geom_vline(aes(xintercept = %i),  alpha=%f)' % (xpos, alpha))
        )

    def add_segment(self, xstart, xend, ystart, yend, color, alpha = 1):
        self._other_adds.append(
            robjects.r('geom_segment')
            (
                robjects.r('aes(x=x, y=y, xend=xend, yend=yend)'),
                exptools.DataFrame.DataFrame({"x": [xstart], 'xend': [xend], 'y': [ystart], 'yend': [yend]}),
                colour=color,
                alpha = alpha

            )
        )

    def facet(self, column_one, column_two = None, fixed_x = True, fixed_y = True):
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
        self._other_adds.append(facet_wrap(robjects.r(facet_specification), scale=scale))


    def greyscale(self):
        self._other_adds.append( robjects.r('scale_colour_grey()'))
        self._other_adds.append( robjects.r('scale_fill_grey()'))

    def theme_bw(self):
        self._other_adds.append(robjects.r('theme_bw()'))
        
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

    def turn_x_axis_labels(self,  angle=75, hjust=0):
        kargs = {
            'axis.text.x': robjects.r('theme_text')(angle = angle, hjust=hjust)
        }
        self._other_adds.append( robjects.r('opts')(**kargs))


        


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
    print aes_params
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
    print dataframe
    ggsave(filename=output_filename,plot=plot, width=8, height=6)

