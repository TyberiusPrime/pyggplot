import rpy2.robjects as robjects

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
    print dataframe
    ggsave(filename=output_filename,plot=plot, width=8, height=6)


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

