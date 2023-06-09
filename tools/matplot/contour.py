# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 3:12 PM on 5/2/2023
"""
import numpy as np
import sys

if './' not in sys.path:
    sys.path.append('./')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm


def ___set_contour_levels___(v, num_levels):
    """ We set the `num_levels` according to the values `v` to be plotted. """
    v_min = list()
    v_max = list()
    for i in v:
        v_min.append(np.min(v[i]))
        v_max.append(np.max(v[i]))
    v_min = min(v_min)
    v_max = max(v_max)

    if abs(v_min - v_max) < 1e-3:
        if abs(v_min - 0) < 1e-3:
            v_max = 0.1
        else:
            if v_min > 0:
                v_max = v_min * 1.01
            else:
                v_max = v_min * 0.99
        num_levels = 2
    levels = np.linspace(v_min, v_max, num_levels)
    return levels


def contourf(*args, **kwargs):
    """contourf"""
    return contour(*args, **kwargs, plot_type='contourf')


def contour(
        x, y, v,
        figsize=(8, 6),
        levels=None, num_levels=20, linewidth=1, linestyle=None,
        usetex=True, colormap='coolwarm',
        show_colorbar=True,
        colorbar_label=None, colorbar_orientation='vertical', colorbar_aspect=20,
        colorbar_labelsize=12.5, colorbar_extend='both',
        ticksize=12,
        labelsize=15,
        title=True,
        saveto=None,
        colorbar_only=False,
        pad_inches=0,
        dpi=150,
        plot_type='contour',
):
    """

    Parameters
    ----------
    x:
        x-coo data. Put in dict for multiple patches. Each patch must be a 2d-ndarray
    y:
        y-coo data. Put in dict for multiple patches. Each patch must be a 2d-ndarray
    v:
        value data. Put in dict for multiple patches. Each patch must be a 2d-ndarray
    figsize
    levels
    num_levels
    linewidth
    linestyle:
        {None, 'solid', 'dashed', 'dashdot', 'dotted'}

    usetex
    colormap
    show_colorbar
    colorbar_label
    colorbar_orientation:
        None or {'vertical', 'horizontal'}

        The orientation of the colorbar. It is preferable to set the location of the colorbar,
        as that also determines the orientation; passing incompatible values for location and
        orientation raises an exception.

    colorbar_aspect
    colorbar_labelsize
    colorbar_extend:
        {'neither', 'both', 'min', 'max'}

        If not 'neither', make pointed end(s) for out-of- range values. These are set for a
        given colormap using the colormap set_under and set_over methods.

    ticksize
    labelsize
    title
    saveto
    plot_type:
        {'contour', 'contourf'}
    colorbar_only
    pad_inches
    dpi

    Returns
    -------

    """
    x = {0: x} if not isinstance(x, dict) else x
    y = {0: y} if not isinstance(y, dict) else y
    v = {0: v} if not isinstance(v, dict) else v

    if saveto is not None:
        matplotlib.use('Agg')

    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "DejaVu sans",
    })

    if usetex:
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

    if colormap is not None:
        plt.rcParams['image.cmap'] = colormap
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    # ------- label and  ticks -------------------------------------------------------
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.xlabel('$x$', fontsize=labelsize)
    plt.ylabel('$y$', fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)

    if levels is None:
        levels = ___set_contour_levels___(v, num_levels)
    else:
        pass
    # -------------- contour plot --------------------------------------------------------
    for patch in v:
        if plot_type == 'contour':
            plt.contour(x[patch], y[patch], v[patch], levels=levels, linewidths=linewidth, linestyles=linestyle)
        elif plot_type == 'contourf':
            VAL = v[patch]
            VAL[VAL > levels[-1]] = levels[-1]
            VAL[VAL < levels[0]] = levels[0]
            plt.contourf(x[patch], y[patch], VAL, levels=levels)
        else:
            raise Exception(f"plot_type={plot_type} is wrong. Should be one of ('contour', 'contourf')")

    # --------------- title -----------------------------------------------------
    if title is None:
        pass
    else:
        plt.title(title)
    # -------------------------------- color bar ---------------------------------
    if show_colorbar:
        mappable = cm.ScalarMappable()
        mappable.set_array(np.array(levels))
        cb = plt.colorbar(mappable, ax=ax,
                          extend=colorbar_extend,
                          aspect=colorbar_aspect,
                          orientation=colorbar_orientation)

        if colorbar_label is not None:
            cb.set_label(colorbar_label, labelpad=10, size=15)

        cb.ax.tick_params(labelsize=colorbar_labelsize)

    # ---------------------- save to ---------------------------------------------
    if colorbar_only:
        ax.remove()
    else:
        pass
    if saveto is None:
        from src.config import _matplot_setting
        plt.show(block=_matplot_setting['block'])
    else:
        plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)

    # --------------------------------------------------------------------------
    return fig
