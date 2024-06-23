# -*- coding: utf-8 -*-
"""
INTRO

Yi Zhang (C)
Created on Wed Apr 17 22:34:33 2019
Aerodynamics, AE
TU Delft
"""
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from typing import List
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})


class MITRVisualize:
    def ___plot_MTIR___(
            self,
            plot_type,
            line_var,       
            res2plot,
            prime='input2',   # sequence of lines
            hcp=None,   # h-convergence parameter
            show_order=False, order_text_size=18, plot_order_triangle=None,  # h-convergence plot related
            title=None, left=0.15, bottom=0.15,
            ylabel=None, yticks=None,
            xlabel=None, xticks=None,
            linewidth=1.2, corlormap='viridis',
            styles=None, colors=None, COLORS=None,
            labels=None, legend_local='best', legend_frame=False,
            minor_tick_length=6, major_tick_length=12, tick_pad=8,
            tick_size=20, label_size=20, legend_size=20, title_size=20, title_pad=12,
            figsize=(7.5, 5), usetex=False, saveto=None,
            xlim=None, ylim=None,
            return_order=False,
    ):
        """ 
        IMPORTANT: this plotter only works for input's criterion is 'standard'.
        
        Here we wrap `plot`, `semilogx`, `semilogy`, `loglog`.
        
        We use the `matplotlib` to do the plot.
        
        Parameters
        ----------
        plot_type : str
            This is the indicator of which type of plot will it be.
        line_var :
            Notice `line_var` is not the x-axis name. It refers to how many lines we 
            want to have. It can only be `input_names[0]` or `input_names[1]`. When
            it is `input_names[0]`, the x-axis is `input_names[1]`, and when it is [1]
            the x-axis is [0].
        res2plot :
            y-axis. We plot `res2plot` against `line_var`. 
        prime : str, optional
            To decide the sequence of the lines in the plot.

            If `prime` == 'line_var': we use `data_sequence_line_var`.
            If `prime` == 'input2'  : we will use `data_sequence__input2`.
        hcp :
            `h-convergence parameter`. We will use `x_data` = `hcp`/`x_data`. This is
            usually used for h-convergence plot, so we call it `hcp`.
        plot_order_triangle :
            If it is not False, we will plot triangles to illustrate the order of plot
            lines.
            
            We will parse `plot_order_triangle` to get parameters for the triangle. For
            the data structure of `plot_order_triangle`, we refer to method
            `___plot_MTIR_plot_order_triangle___`
        COLORS:
            The color sequence of all lines regardless of the group of the line.
            
        """
        # noinspection PyUnresolvedReferences
        D = self._data_
        # _____________check `line_var` and `res2plot`___________________________________
        # noinspection PyUnresolvedReferences
        input_names = self._dfw_._runner_.input_names
        # noinspection PyUnresolvedReferences
        output_names = self._dfw_._runner_.output_names
        if isinstance(res2plot, str):  # we make it possible to plot more than one result.
            res2plot = (res2plot,)
        assert all([res2plot[i] in output_names for i in range(len(res2plot))]), \
            " <RunnerVisualize> : res2plot={} is wrong.".format(res2plot)
        assert line_var in input_names[:2], " <RunnerVisualize> : line_var={} is wrong.".format(line_var)
        line_var_index = input_names.index(line_var)
        x_index = 0 if line_var_index == 1 else 1
        x_name = input_names[x_index]
        # group the plot data: the data will be grouped into a tuple called `data_sequence`
        data_sequence_line_var = ()  # `i2` changes in each `line_var`.
        data_sequence_inputs2 = ()  # `line_var` changes in each `i2`.
        if prime == 'line_var':
            for ai in set(D[line_var]):
                sub_rdf = D[D[line_var] == ai]
                for i2i in set(D[input_names[2]][D[line_var] == ai]):
                    data_sequence_line_var += (sub_rdf[sub_rdf[input_names[2]] == i2i],)
            data_sequence = data_sequence_line_var
        elif prime == 'input2':
            for i2i in set(D[input_names[2]]):
                sub_rdf = D[D[input_names[2]] == i2i]
                for ai in set(D[line_var][D[input_names[2]] == i2i]):
                    data_sequence_inputs2 += (sub_rdf[sub_rdf[line_var] == ai],)
            data_sequence = data_sequence_inputs2
        else:
            raise Exception(" <RunnerVisualize> : prime={} is wrong.".format(prime))
        # _____ prepare styles, colors, labels If they are none.________________________
        num_res2plot = len(res2plot)
        num_lines = len(data_sequence)
        # noinspection PyUnresolvedReferences
        line_groups = self._dfw_._runner_.input_shape[2]
        num_lines_per_group = int(num_lines/line_groups)
        if styles is None:
            styles = ('-^', '-x', '-o', '-s', '-*', '-8', '->', '-p', 
                      '-H', '-h', '->', '-D', '-d', '-P', '-v') * 5
        if colors is None:
            color = cm.get_cmap(corlormap, line_groups)
            colors = []
            for j in range(line_groups):
                colors.append(color(j))
        
        if labels is None:
            labels = ()
            for i in range(line_groups):
                for j in range(num_lines_per_group):
                    k = j + i * num_lines_per_group
                    list_line_var = data_sequence[k][line_var].tolist()
                    assert all([i == list_line_var[0] for i in list_line_var[1:]]), \
                        " <RunnerVisualize> : data grouping is wrong for %r" % data_sequence[k]
                    list_input2 = data_sequence[k][input_names[2]].tolist()
                    assert all([i == list_input2[0] for i in list_input2[1:]]), \
                        " <RunnerVisualize> : data grouping is wrong for %r" % data_sequence[k]
                    if num_res2plot == 1:
                        labels += (line_var+'='+str(list_line_var[0]) + ', ' +
                                   input_names[2]+'='+str(list_input2[0]).replace('_', '-'),)
                    else:
                        for m in range(num_res2plot):
                            labels += (line_var+'='+str(list_line_var[0])+', ' +
                                       input_names[2]+'='+str(list_input2[0]) + ', ' +
                                       res2plot[m].replace('_', '-'),)
            labels = list(labels)
        elif labels is False:
            if show_order:
                labels = ['' for _ in range(num_lines)]
            else:
                pass
        else:
            pass

        # ___ preparing orders _________________________________________________________
        orders: List[float] = [0.0 for _ in range(num_lines)]
        # ___ pre-parameterize the plot_________________________________________________
        if saveto is not None:
            matplotlib.use('Agg')
        else:
            matplotlib.use('TkAgg')

        plt.rcParams.update({
            "text.usetex": usetex,
            "font.family": "DejaVu sans",
            # "font.serif": "Times New Roman",
        })

        if usetex:
            plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

        fig, ax = plt.subplots(figsize=figsize)
        plt.gcf().subplots_adjust(left=left)
        plt.gcf().subplots_adjust(bottom=bottom)
        # __ find the range of x_data___________________________________________________
        xd_max = ()
        xd_min = ()
        ploter = getattr(plt, plot_type)  # we get the plotter from matplotlib
        for i in range(line_groups):
            for j in range(num_lines_per_group):
                k = j + i*num_lines_per_group
                xdata2plot = np.array(data_sequence[k][x_name].tolist())
                if hcp is not None:
                    xdata2plot = hcp / xdata2plot
                xd_max += (np.max(xdata2plot),)
                xd_min += (np.min(xdata2plot),)
        xd_max = np.min(xd_max)
        xd_min = np.min(xd_min)
        x_range = (xd_max, xd_min)
        # ___ do THE PLOT_______________________________________________________________
        for i in range(line_groups):
            for j in range(num_lines_per_group):
                k = j + i*num_lines_per_group
                xdata2plot = np.array(data_sequence[k][x_name].tolist())
                if hcp is not None:
                    xdata2plot = hcp / xdata2plot
                for m in range(num_res2plot):
                    n = m + j*num_res2plot + i*num_res2plot*num_lines_per_group
                    J = m + j*num_res2plot
                    N = m + j*num_res2plot + i*num_res2plot*num_lines_per_group
                    ydata2plot = data_sequence[k][res2plot[m]]

                    if show_order or return_order:
                        try:
                            # noinspection PyUnboundLocalVariable
                            orders[n] = (np.log10(ydata2plot.values[-1]) - np.log10(ydata2plot.values[-2])) /\
                                        (np.log10(xdata2plot[-1])-np.log10(xdata2plot[-2]))
                        except IndexError:
                            orders[n] = float('nan')
                    # __add order to label______________________________________________
                    if show_order:
                        labels[n] += ', order$={}$'.format('%0.2f' % (orders[n]))
                    # ___ get data and plot the triangle that shows the order___________
                    if plot_order_triangle is not None:
                        # __ check_____________________________________________________
                        assert isinstance(plot_order_triangle, dict), \
                            " <RunnerVisualize> : plot_order_triangle needs to be a dict."
                        # __compute_data________________________________________________
                        if n in plot_order_triangle:
                            potn = plot_order_triangle[n]
                            assert isinstance(potn, dict), \
                                " <RunnerVisualize> : plot_order_triangle[{}] needs to be a dict.".format(n)
                            c0, c1, c2, textpos, ordern = self.___plot_MTIR_plot_order_triangle___(
                                plot_type, x_range, potn, xdata2plot, ydata2plot.values)
                        # ___plot triangle______________________________________________
                            c0x, c0y = c0
                            c1x, c1y = c1
                            c2x, c2y = c2
                            plt.fill_between([c0x, c1x], [c0y, c1y], [c0y, c2y], color='grey', alpha=0.5)
                            if isinstance(ordern, int):
                                plt.text(textpos[0], textpos[1], "${}$".format(ordern), fontsize=order_text_size)
                            else:
                                plt.text(textpos[0], textpos[1], "${}$".format('%0.2f' % ordern),
                                         fontsize=order_text_size)
                        # --------------------------------------------------------------
                    # ------------------------------------------------------------------
                    if COLORS is None:
                        if labels is False:
                            ploter(xdata2plot, ydata2plot, styles[J],
                                   color=colors[i], linewidth=linewidth)
                        else:
                            ploter(xdata2plot, ydata2plot, styles[J],
                                   color=colors[i], label=labels[n], linewidth=linewidth)
                    else:
                        if labels is False:
                            ploter(xdata2plot, ydata2plot, styles[J],
                                   color=COLORS[N], linewidth=linewidth)
                        else:
                            ploter(xdata2plot, ydata2plot, styles[J],
                                   color=COLORS[N], label=labels[n], linewidth=linewidth)
        # ___ post-parameterize the plot________________________________________________
        plt.tick_params(which='both', labeltop=False, labelright=False, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=minor_tick_length)
        plt.tick_params(axis='both', which='major', direction='in', length=major_tick_length)
        plt.tick_params(axis='both', which='both', labelsize=tick_size)
        plt.tick_params(axis='x', which='both', pad=tick_pad)
        plt.tick_params(axis='y', which='both', pad=tick_pad)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if labels is False:
            pass
        else:
            plt.legend(fontsize=legend_size, loc=legend_local, frameon=legend_frame)
        if xlabel is not None: 
            plt.xlabel(xlabel, fontsize=label_size)
        else:
            if hcp is None:
                plt.xlabel(x_name, fontsize=label_size)
            else:
                plt.xlabel(str(hcp)+'/'+x_name, fontsize=label_size)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=label_size)

        if xticks is not None:
            if isinstance(xticks, dict):  # we are configuring the minor ticks and their labels.
                xtick_para = xticks
                ax.set_xticks(xtick_para['xticks'], labels=xtick_para['labels'], minor=xtick_para['minor'])
            else:
                plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)

        if title is None: 
            if len(res2plot) == 1:
                plt.title(r'' + res2plot[0].replace('_', '-'), fontsize=title_size, pad=title_pad)
            else:
                plt.title(r'' + str(res2plot).replace('_', '-'), fontsize=title_size, pad=title_pad)
        elif title is False:
            pass
        else:
            plt.title(r'' + title, fontsize=title_size, pad=title_pad)

        plt.tight_layout()
        if saveto is not None and saveto != '':
            if saveto[:-4] == 'pdf':
                plt.savefig(saveto, bbox_inches='tight')
            else:
                plt.savefig(saveto, dpi=210, bbox_inches='tight')

        else:
            plt.show()

        plt.close()
        # ------------------------------------------------------------------------------
        if return_order:
            return orders
        else:
            return 0
    
    def ___plot_MTIR_plot_order_triangle___(self, plot_type, x_range, potn, xd, yd):
        """ 
        We plot a triangle along the line indicated by `xd`, `yd`.
        
        Parameters
        ----------
        plot_type :
        x_range :
            The x_range of whole xd's for all lines.
        potn : dict or True
            The information that is going to use for the triangle.
            
            It should have keys (optional):
                "p" : a tuple of shape (2,), like (a, b) and 0 <= a, b <= 1. (a, b) 
                    represents the pointing corner of the triangle. And (a, b) is the 
                    distance from the last value point.
                "tp":
                    A tuple of shape (2,), like (c,d), (c,d) decides the position of 
                    the text (the order number).
                "l": the length of the bottom line: -1 <= l <= 1.
                    It means we will draw a horizontal line, starting from 'p',
                    going toward right (l>0) or left (l<0), whose length is 
                    abs(l)*`total_length_of_plot_horizontal_length`.
                "h": the height; the order. # NOT USED KEY!!!
                    if "h" > 0, the height goes up, if "h" < 0, it goes down. Both from
                    the edge point of the horizontal line.
                    "h" means the height of the height line is `l*h`, of course, the 
                    scale of x-y axes are considered.
                "order": the order text shown besides the triangle.
                    
        xd :
            The x-axis data to plot for this line.
        yd :
            The y-axis data to plot for this line.
                    
        """
        # ___default "p"________________________________________________________________
        if "p" not in potn:
            potn["p"] = (0, -0.3)
        if "tp" not in potn:
            potn["tp"] = (0.02, 0.2)
        # ___default "l"________________________________________________________________
        if "l" not in potn:
            potn["l"] = 0.1
        # ___default "h"________________________________________________________________
        if "order" not in potn:
            potn["order"] = (np.log10(yd[-1]) - np.log10(yd[-2])) / (np.log10(xd[-1]) - np.log10(xd[-2]))
        # ___ loglog____________________________________________________________________
        if plot_type == 'loglog':
            x_max, x_min = x_range
            x_range = np.log10(x_max) - np.log10(x_min)
            origin = (xd[-1], yd[-1])
            otc0x = np.log10(origin[0]) + x_range*potn["p"][0] 
            otc0x = 10**otc0x 
            otc0y = np.log10(origin[1]) + x_range*potn["p"][1] 
            otc0y = 10**otc0y 
            otc0 = (otc0x, otc0y)  # order_triangle_corner_0
            otc1x = np.log10(otc0x) + x_range*potn["l"]
            otc1x = 10**otc1x 
            otc1 = (otc1x, otc0y)  # order_triangle_corner_0
            otc2y = np.log10(otc0y) + x_range*potn["l"]*potn["order"]
            otc2y = 10**otc2y
            otc2 = (otc1x, otc2y)  # order_triangle_corner_0
            ttps_x, ttps_y = potn["tp"]
            textpos_x = 10**(np.log10(otc1x) + x_range*ttps_x)
            textpos_y = 10**(np.log10(otc0y) + x_range*potn["l"]*potn["order"]*ttps_y)
            order = potn["order"]
            return otc0, otc1, otc2, (textpos_x, textpos_y), order
        # ...
        else:
            raise NotImplementedError(" <plot_order_triangle> does not work for {} plot".format(plot_type))
