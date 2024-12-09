# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm


def __matplot__(
        plot_type,
        # data, and linewidth
        x, y, num_lines=1, linewidth=1.2, markersize=None,
        # style, color, and labels
        style=None, color=None, label=False,
        styles=None, colors=None, labels=None, mfcs=None,  # mfc: maker face color
        # config
        usetex=True, saveto=None, pad_inches=0.1, colormap='Dark2',
        # figure
        figsize=(5.5, 4), left=0.15, bottom=0.15,
        # title
        title=None, title_size=20, title_pad=12,
        # labels
        xlabel=None, ylabel=None, label_size=16,
        xlabel_pad=None, ylabel_pad=None,
        # ticks
        tick_style: [] = "sci", xticks=None, yticks=None, tick_direction='in',
        tick_size=16, tick_pad=6, minor_tick_length=4, major_tick_length=8,

        labelleft=True,
        labelbottom=True,

        # legend
        legend_size=18, legend_local='best', legend_frame=False,
        xlim=None, ylim=None,
        y_scientific=True,
        legend_ncol=1,
        scatter=None,
        scatter_kwargs=None,

):
    """

    Parameters
    ----------
    plot_type
    x :
        If num_lines > 1, we plot (x[i], y[i]) for i in range(num_lines).
    y :
        If num_lines > 1, we plot (x[i], y[i]) for i in range(num_lines).
    num_lines
    style
    color
    label : bool, None, str
        The arg affects when `num_lines` == 1.

        If it is False (default): we will turn off the label.
        If it is None: we will use a default label.
        If it is a str: we will it as the label of the single line.

    styles :
    colors :
    labels : list, tuple, bool
    linewidth
    markersize
    usetex
    saveto
    pad_inches
    colormap
    figsize
    left
    bottom
    title
    title_size
    title_pad
    xlabel
    ylabel
    label_size
    xlabel_pad
    ylabel_pad
    tick_style : {'sci', 'scientific', 'plain'}
    xticks
    yticks
    tick_size : int
        The font size of the ticks.
    tick_pad : int
        The gap size between the ticks and the axes.
    minor_tick_length
    major_tick_length
    labelleft
    labelbottom
    legend_size
    legend_local
    legend_frame
    legend_ncol:
    xlim
    ylim
    y_scientific
    scatter
    scatter_kwargs

    Returns
    -------

    """
    # - config matplotlib -------------------------------------------------------------------------1
    if saveto is not None:
        matplotlib.use('Agg')

    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "DejaVu sans",
        # "font.serif": "Times New Roman",
    })

    if usetex:
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath, amssymb}"

    _, ax = plt.subplots(figsize=figsize)
    plt.gcf().subplots_adjust(left=left)
    plt.gcf().subplots_adjust(bottom=bottom)

    plt.tick_params(labelleft=labelleft, labelbottom=labelbottom)

    # --------- default styles, colors, and labels -------------------------------------------------

    if num_lines == 1:  # single line
        if style is None:
            style = '-^'
        if color is None:
            color = 'darkgray'
        assert label in (None, False) or isinstance(label, str), f"label must be a str, False, or None."
        if label is None:
            label = r'$\mathrm{line}\#0$'

    elif num_lines > 1:  # multiple lines

        if styles is None:
            styles = ('-^', '-x', '-o', '-s', '-v', '-*', '-8', '->', '-p',
                      '-H', '-h', '-D', '-d', '-P') * 5
        elif isinstance(styles, str):
            styles = [styles for _ in range(num_lines)]
        else:
            pass

        if colors is None:
            color = cm.get_cmap(colormap, num_lines)
            colors = []
            for j in range(num_lines):
                colors.append(color(j))
        elif isinstance(colors, str):
            colors = [colors for _ in range(num_lines)]
        else:
            pass

        if labels is None:
            labels = [r'$\mathrm{line}\#' + str(_) + '$' for _ in range(num_lines)]

        assert isinstance(styles, (list, tuple)), \
            f"put correct amount of styles in list pls."
        assert isinstance(colors, (list, tuple)), \
            f"put correct amount of colors in list pls."

        if labels is False:
            pass
        else:
            assert isinstance(labels, (list, tuple)), f"put labels in list pls."
            assert len(labels) == num_lines, f"I need {num_lines} labels, now I get {len(labels)}."
            for i, lab in enumerate(labels):
                if lab is not None:
                    assert isinstance(lab, str), f"labels[{i}] = {lab} is not str."

    else:
        raise Exception()

    # - do the plot -------------------------------------------------------------------------------1
    plotter = getattr(plt, plot_type)

    if num_lines == 1:
        if label is not False:
            plotter(x, y, style, color=color, label=label, markersize=markersize, linewidth=linewidth)
        else:
            plotter(x, y, style, color=color, markersize=markersize, linewidth=linewidth)

    else:
        if mfcs is None:
            if labels is False:
                for i in range(num_lines):
                    plotter(x[i], y[i], styles[i], markersize=markersize, color=colors[i], linewidth=linewidth)
            else:
                for i in range(num_lines):
                    plotter(x[i], y[i], styles[i],
                            markersize=markersize, color=colors[i], label=labels[i], linewidth=linewidth)

        else:
            if labels is False:
                for i in range(num_lines):
                    plotter(x[i], y[i], styles[i],
                            color=colors[i], markersize=markersize, linewidth=linewidth, mfc=mfcs[i],)
            else:
                for i in range(num_lines):
                    plotter(x[i], y[i], styles[i],
                            color=colors[i], markersize=markersize, label=labels[i], linewidth=linewidth, mfc=mfcs[i],)

    # ------ customize figure ---------------------------------------------------------------------1
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)

    plt.tick_params(which='both', labeltop=False, labelright=False, top=True, right=True)
    plt.tick_params(axis='both', which='minor', direction=tick_direction, length=minor_tick_length)
    plt.tick_params(axis='both', which='major', direction=tick_direction, length=major_tick_length)
    plt.tick_params(axis='both', which='both', labelsize=tick_size)
    plt.tick_params(axis='x', which='both', pad=tick_pad)
    plt.tick_params(axis='y', which='both', pad=tick_pad)

    if plot_type in ('semilogy', 'loglog'):
        pass
    else:
        plt.ticklabel_format(style=tick_style, axis='y', scilimits=(0, 0))
        tx = ax.yaxis.get_offset_text()
        tx.set_fontsize(tick_size)

        if not y_scientific:
            ax.get_yaxis().get_major_formatter().set_scientific(y_scientific)

    if scatter is None:
        pass
    else:
        ax.scatter(*scatter, **scatter_kwargs)

    if xlabel is not None:
        if xlabel_pad is None:
            plt.xlabel(xlabel, fontsize=label_size)
        else:
            plt.xlabel(xlabel, fontsize=label_size, labelpad=xlabel_pad)

    if ylabel is not None:
        if ylabel_pad is None:
            plt.ylabel(ylabel, fontsize=label_size)
        else:
            plt.ylabel(ylabel, fontsize=label_size, labelpad=ylabel_pad)

    if title is None:
        pass
    elif title is False:
        pass
    else:
        plt.title(r'' + title, fontsize=title_size, pad=title_pad)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # ----legend ----------------------------------------------------------------------------------1
    if num_lines == 1 and label is False:
        # turn off legend when num_line==1 and label is False
        pass
    elif num_lines > 1 and labels is False:
        # turn off legend when num_line > 1 and labels is False
        pass
    else:
        plt.legend(fontsize=legend_size, loc=legend_local, ncol=legend_ncol,
                   frameon=legend_frame)

    # ---------------- save the figure ------------------------------------------------------------1
    plt.tight_layout()
    if saveto is not None and saveto != '':
        plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches)
        plt.close()
    else:
        from src.config import _setting
        plt.show(block=_setting['block'])
    # =============================================================================================1
    return 0


def plot(*args, **kwargs):
    return __matplot__('plot', *args, **kwargs)


def semilogy(*args, **kwargs):
    return __matplot__('semilogy', *args, **kwargs)


def loglog(*args, **kwargs):
    return __matplot__('loglog', *args, **kwargs)
