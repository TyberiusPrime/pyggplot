import os
import tempfile
import pandas as pd
import numpy as np
import itertools
from .base import _PlotBase
#try:
    #import exptools
    #exptools.load_software('palettable')
    #exptools.load_software('descartes')
    #exptools.load_software('mizani')
    #exptools.load_software('patsy')
    #exptools.load_software('plotnine')

#except ImportError:
    #pass
import matplotlib
matplotlib.use('agg')
import plotnine as p9
from plotnine import stats


class Expression:
    def __init__(self, expr_str):
        self.expr_str = expr_str


class Scalar:
    def __init__(self, scalar_str):
        self.scalar_str = scalar_str


class Plot(_PlotBase):
    def __init__(self, dataframe):
        self.dataframe = self._prep_dataframe(dataframe)
        self.ipython_plot_width = 600
        self.ipython_plot_height = 600
        self.plot = p9.ggplot(self.dataframe)
        self._add_geoms()
        self._add_scales_and_cords_and_facets_and_themes()
        self._add_positions_and_stats()

    def __add__(self, other):
        if hasattr(other, '__radd__'): 
            # let's assume that it's a plotnine object that knows how to radd it
            # self to our plot
            self.plot = self.plot + other
        return self


    def _prep_dataframe(self, df):
        """prepare the dataframe by making sure it's a pandas dataframe,
        has no multi index columns, has no index etc
        """
        if 'pydataframe.dataframe.DataFrame' in str(type(df)):
            df = self._convert_pydataframe(df)
        elif isinstance(df, dict):
            df = pd.DataFrame(df)
        elif isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        return df

    def _convert_pydataframe(self, pdf):
        """Compability shim for still being able to use old pydataframes with the new pandas interface"""
        d = {}
        for column in pdf.columns_ordered:
            o = pdf.gcv(column)
            if 'pydataframe.factors.Factor' in str(type(o)):
                d[column] = pd.Series(
                    pd.Categorical(o.as_levels(), categories=o.levels))
            else:
                d[column] = o
        return pd.DataFrame(d)

    def _repr_png_(self, width=None, height=None):
        """Show the plot in the ipython notebook (ie. return png formated image data)"""
        if width is None:
            width = self.ipython_plot_width
            height = self.ipython_plot_height
        try:
            handle, name = tempfile.mkstemp(
                suffix=".png"
            )  # mac os for some reason would not read back again from a named tempfile.
            os.close(handle)
            self.plot.save(
                name,
                width=width / 72.,
                height=height / 72.,
                dpi=72,
                verbose=False)
            tf = open(name, "rb")
            result = tf.read()
            tf.close()
            return result
        finally:
            os.unlink(name)

    def _repr_svg_(self, width=None, height=None):
        """Show the plot in the ipython notebook (ie. return svg formated image data)"""
        if width is None:
            width = self.ipython_plot_width / 150. * 72
            height = self.ipython_plot_height / 150. * 72
        try:
            handle, name = tempfile.mkstemp(
                suffix=".svg"
            )  # mac os for some reason would not read back again from a named tempfile.
            os.close(handle)
            self.plot.save(
                name,
                width=width / 72.,
                height=height / 72.,
                dpi=72,
                verbose=False)
            tf = open(name, "rb")
            result = tf.read().decode('utf-8')
            tf.close()
            # newer jupyters need the height attribute
            # otherwise the iframe is only one line high and
            # the figure tiny
            result = result.replace("viewBox=",
                                    'height=\'%i\' viewBox=' % (height))
            return result, {"isolated": True}
        finally:
            os.unlink(name)

    #def geom_scatter(self, x, y):
    #self.plot += p9.geom_point(p9.aes(x, y))

    def _add(self, geom_class, args, kwargs):
        """The generic method to add a geom to the ggplot.
        You need to call add_xyz (see _add_geom_methods for a list, with each variable mapping
        being one argument) with the respectivly required parameters (see ggplot documentation).
        You may optionally pass in an argument called data, which will replace the plot-global dataframe
        for this particular geom
        """

        if 'data' in kwargs:
            data = self._prep_dataframe(kwargs['data'])
        else:
            data = None
        if 'stat' in kwargs:
            stat = kwargs['stat']
        else:
            stat = geom_class.DEFAULT_PARAMS['stat']
        if isinstance(stat, str):
            stat = getattr(
                p9.stats, 'stat_' + stat
                if not stat.startswith('stat_') else stat)
        mapping = {}
        out_kwargs = {}
        all_defined_mappings = list(
            stat.REQUIRED_AES) + list(geom_class.REQUIRED_AES) + list(
                geom_class.DEFAULT_AES)  + ['group'] # + list(geom_class.DEFAULT_PARAMS)
        if 'x' in geom_class.REQUIRED_AES:
            if len(args) > 0:
                kwargs['x'] = args[0]
            if 'y' in geom_class.REQUIRED_AES or 'y' in stat.REQUIRED_AES:
                if len(args) > 1:
                    kwargs['y'] = args[1]
                if len(args) > 2:
                    raise ValueError(
                        "We only accept x&y by position, all other args need to be named"
                    )
            else:
                if len(args) > 1:
                    raise ValueError(
                        "We only accept x by position, all other args need to be named"
                    )
        elif 'xmin' and 'ymin' in geom_class.REQUIRED_AES:
            if len(args) > 0:
                kwargs['xmin'] = args[0]
            if len(args) > 1:
                kwargs['xmax'] = args[1]
            if len(args) > 2:
                kwargs['ymin'] = args[2]
            if len(args) > 3:
                kwargs['ymax'] = args[3]
            elif len(args) > 4:
                raise ValueError(
                    "We only accept xmin,xmax,ymin,ymax by position, all other args need to be named"
                )

        for a, b in kwargs.items():
            if a in all_defined_mappings:
                # if it is an expression, keep it that way
                # if it's a single value, treat it as a scalar
                # except if it looks like an expression (ie. has ()
                is_kwarg = False
                if isinstance(b, Expression):
                    b = b.expr_str
                    is_kwarg = True
                elif isinstance(b, Scalar):
                    b = b.scalar_str
                    is_kwarg = True
                elif (((data is not None and b not in data.columns) or
                       (data is None and b not in self.dataframe.columns))
                      and not '(' in str(
                          b)  # so a true scalar, not a calculated expression
                      ):
                    b = b  # which will tell it to treat it as a scalar!
                    is_kwarg = True
                if not is_kwarg:
                    mapping[a] = b
                else:
                    out_kwargs[a] = b

        #mapping.update({x: kwargs[x] for x in kwargs if x in all_defined_mappings})

        out_kwargs['data'] = data
        for a in geom_class.DEFAULT_PARAMS:
            if a in kwargs:
                out_kwargs[a] = kwargs[a]

        self.plot += geom_class(mapping=p9.aes(**mapping), **out_kwargs)
        return self

    def _add_geoms(self):
        # allow aliases
        name_to_geom = {}
        for name in dir(p9):
            if name.startswith('geom_'):
                geom = getattr(p9, name)
                name_to_geom[name] = geom
        name_to_geom['geom_scatter'] = name_to_geom['geom_point']
        for name, geom in name_to_geom.items():

            def define(geom):
                def do_add(*args, **kwargs):
                    return self._add(geom_class=geom, args=args, kwargs=kwargs)

                do_add.__doc__ = geom.__doc__
                return do_add

            method_name = 'add_' + name[5:]
            if not hasattr(self, method_name):
                setattr(self, method_name, define(geom))

        return self

    def _add_scales_and_cords_and_facets_and_themes(self):
        for name in dir(p9):
            if name.startswith('scale_') or name.startswith(
                    'coord_') or name.startswith('facet') or name.startswith(
                        'theme'):
                method_name = name
                if not hasattr(self, method_name):
                    scale = getattr(p9, name)

                    def define(scale):
                        def add_(*args, **kwargs):
                            self.plot += scale(*args, **kwargs)
                            return self

                        add_.__doc__ = scale.__doc__
                        return add_

                    setattr(self, method_name, define(scale))

    def _add_positions_and_stats(self):
        for name in dir(p9):
            if name.startswith('stat_') or name.startswith('position'):
                setattr(self, name, getattr(p9, name))

    def title(self, text):
        """Set plot title"""
        self.plot += p9.ggtitle(text)

    def set_title(self, text):
        """Set plot title"""
        self.title(text)

    def facet(self, *args, **kwargs):
        """Compability to old calling style"""
        if 'free_y' in kwargs['scales']:
            self.plot += p9.theme(subplots_adjust={'wspace':0.2})
        return self.facet_wrap(*args, **kwargs)

    def add_jitter(self, x, y, jitter_x=True, jitter_y=True, **kwargs):
        # an api changed in ggplot necessitates this - jitter_x and jitter_y have been replaced with position_jitter(width, height)...
        kwargs['position'] = self.position_jitter(
            width=0.4 if jitter_x is True else float(jitter_x),
            height=0.4 if jitter_y is True else float(jitter_y),
        )
        self.add_point(x, y, **kwargs)

    def add_cummulative(self,
                        x_column,
                        ascending=True,
                        percent=False,
                        percentile=1.0):
        """Add a line showing cumulative % of data <= x.
        if you specify a percentile, all data at the extreme range is dropped


        """
        total = 0
        current = 0
        column_data = self.dataframe[x_column].copy()  # explicit copy!
        column_data = column_data[~np.isnan(column_data)]
        column_data = np.sort(column_data)
        total = float(len(column_data))
        real_total = total
        if not ascending:
            column_data = column_data[::-1]  # numpy.reverse(column_data)
        if percentile != 1.0:
            if ascending:
                maximum = np.max(column_data)
            else:
                maximum = np.min(column_data)
            total = float(total * percentile)
            if total > 0:
                column_data = column_data[:total]
                offset = real_total - total
            else:
                column_data = column_data[total:]
                offset = 2 * abs(total)
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
            # y_values.append(current)
        data = pd.DataFrame({
            x_column: x_values,
            ("%" if percent else '#') + ' <=': y_values
        })
        if percentile > 0:
            self.scale_x_continuous(
                limits=[0, real_total if not percent else 100])
        self.add_line(x_column, ("%" if percent else '#') + ' <=', data=data)
        if percentile != 1.0:
            self.set_title('showing only %.2f percentile, extreme was %.2f' %
                           (percentile, maximum))
        return self

    def add_heatmap(self,
                    x_column,
                    y_column,
                    fill,
                    low="red",
                    mid=None,
                    high="blue",
                    midpoint=0,
                    guide_legend=None,
                    scale_args=None):
        self.add_tile(x_column, y_column, fill=fill)
        if mid is None:
            self.scale_fill_gradient(low=low, high=high, **scale_args)
        else:
            self.scale_fill_gradient2(
                low=low, mid=mid, high=high, midpoint=midpoint, **scale_args)

    def add_alternating_background(self,
                                   x_column,
                                   fill_1="#EEEEEE",
                                   fill_2="#FFFFFF",
                                   vertical=False,
                                   alpha=0.5,
                                   log_y_scale=False,
                                   facet_column=None):
        """Add an alternating background to a categorial (x-axis) plot.
        """
        self.scale_x_discrete()
        if log_y_scale:
            self._expected_y_scale = 'log'
        else:
            self._expected_y_scale = 'normal'
        if facet_column is None:
            sub_frames = [(False, self.dataframe)]
        else:
            sub_frames = self.dataframe.groupby(facet_column)

        for facet_value, facet_df in sub_frames:
            no_of_x_values = len(facet_df[x_column].unique())
            df_rect = pd.DataFrame({
                'xmin':
                np.array(range(no_of_x_values)) - .5 + 1,
                'xmax':
                np.array(range(no_of_x_values)) + .5 + 1,
                'ymin':
                -np.inf if not log_y_scale else 0,
                'ymax':
                np.inf,
                'fill':
                ([fill_1, fill_2] * (no_of_x_values // 2 + 1))[:no_of_x_values]
            })
            if facet_value is not False:
                df_rect.insert(0, facet_column, facet_value)
            #df_rect.insert(0, 'alpha', alpha)
            left = df_rect[df_rect.fill == fill_1]
            right = df_rect[df_rect.fill == fill_2]
            if not vertical:
                if len(left):
                    self.add_rect(
                        'xmin',
                        'xmax',
                        'ymin',
                        'ymax',
                        fill='fill',
                        data=left,
                        alpha=alpha)
                if len(right):
                    self.add_rect(
                        'xmin',
                        'xmax',
                        'ymin',
                        'ymax',
                        fill='fill',
                        data=right,
                        alpha=alpha)
            else:
                if len(left):
                    self.add_rect(
                        'ymin',
                        'ymax',
                        'xmin',
                        'xmax',
                        fill='fill',
                        data=left,
                        alpha=alpha)
                if len(right):
                    self.add_rect(
                        'ymin',
                        'ymax',
                        'xmin',
                        'xmax',
                        fill='fill',
                        data=right,
                        alpha=alpha)
        self.scale_fill_identity()
        self.scale_alpha_identity()
        return self

    def turn_x_axis_labels(self,
                           angle=90,
                           hjust='center',
                           vjust='top',
                           size=None,
                           color=None):
        return self.turn_axis_labels('axis_text_x', angle, hjust, vjust, size,
                                     color)

    def turn_y_axis_labels(self,
                           angle=90,
                           hjust='hjust',
                           vjust='center',
                           size=None,
                           color=None):
        return self.turn_axis_labels('axis_text_y', angle, hjust, vjust, size,
                                     color)

    def _change_theme(self, what, t):
        if self.plot.theme is None:
            self.theme_grey()
        self.plot.theme += p9.theme(**{what: t})
        return self

    def turn_axis_labels(self, ax, angle, hjust, vjust, size, color):
        t = p9.themes.element_text(
            rotation=angle, ha=hjust, va=vjust, size=size, color=color)
        return self._change_theme(ax, t)

    def hide_background(self):
        return self._change_theme('panel_background', p9.element_blank())

    def hide_y_axis_labels(self):
        return self._change_theme('axis_text_y', p9.element_blank())

    def hide_x_axis_labels(self):
        return self._change_theme('axis_text_x', p9.element_blank())

    def hide_axis_ticks(self):
        return self._change_theme('axis_ticks', p9.element_blank())

    def hide_axis_ticks_x(self):
        return self._change_theme('axis_ticks_major_x', p9.element_blank())

    def hide_axis_ticks_y(self):
        return self._change_theme('axis_ticks_major_y', p9.element_blank())

    def hide_y_axis_title(self):
        return self._change_theme('axis_title_y', p9.element_blank())

    def hide_x_axis_title(self):
        return self._change_theme('axis_title_x', p9.element_blank())

    def hide_facet_labels(self):
        self._change_theme('strip_background', p9.element_blank())
        return self._change_theme('strip_text_x', p9.element_blank())

    def hide_legend_key(self):
        raise ValueError("plotnine doesn't do 'hide_legend' - pass show_legend=False to the geoms instead")
        
    _many_cat_colors = [
        "#1C86EE",
        "#E31A1C",  # red
        "#008B00",
        "#6A3D9A",  # purple
        "#FF7F00",  # orange
        "#4D4D4D",
        "#FFD700",
        "#7EC0EE",
        "#FB9A99",  # lt pink
        "#90EE90",
        # "#CAB2D6", # lt purple
        "#0000FF",
        "#FDBF6F",  # lt orange
        "#B3B3B3",
        "EEE685",
        "#B03060",
        "#FF83FA",
        "#FF1493",
        "#0000FF",
        "#36648B",
        "#00CED1",
        "#00FF00",
        "#8B8B00",
        "#CDCD00",
        "#8B4500",
        "#A52A2A"
    ]

    def scale_fill_many_categories(self, offset=0, **kwargs):
        self.scale_fill_manual((self._many_cat_colors + self._many_cat_colors
                                )[offset:offset + len(self._many_cat_colors)], **kwargs)
        return self

    def scale_color_many_categories(self, offset=0, **kwargs):
        self.scale_color_manual((self._many_cat_colors + self._many_cat_colors
                                 )[offset:offset + len(self._many_cat_colors)], **kwargs)
        return self

    def render(self,
               output_filename,
               width=8,
               height=6,
               dpi=300,
               din_size=None):
        if din_size == 'A4':
            width = 8.267
            height = 11.692
        self.plot += p9.theme(dpi=dpi)
        self.plot.save(filename=output_filename, width=width, height=height, verbose=False)


save_as_pdf_pages = p9.save_as_pdf_pages
