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

