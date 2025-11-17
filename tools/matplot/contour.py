# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.use('TkAgg')
plt.rcParams.update({
    "font.family": "DejaVu sans",
})


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
        levels=None, level_range=None, num_levels=20, linewidth=1,
        linestyles=None,
        usetex=True,

        colormap='bwr',
        colors=None,

        show_colorbar=True,
        colorbar_only=False,
        colorbar_label=None, colorbar_orientation='vertical', colorbar_aspect=20,
        colorbar_labelsize=12.5, colorbar_extend='both',

        colorbar_ticks=None,
        colorbar_tick_labels=None,

        ticksize=12,
        xticks=None, yticks=None,
        minor_tick_length=4, major_tick_length=8,

        labelsize=15,
        xlabel=r'$x$',
        ylabel=r'$y$',

        title=None,
        title_fontsize=15,

        saveto=None,
        pad_inches=0,
        dpi=150,
        plot_type='contour',
        intermediate=False,
        magnitude=False,
        top_right_bounds=False,

        aspect=None,
        outline=None,
        outline_linewidth=1.5,
        outline_color='k',
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
    level_range
    num_levels
    linewidth
    linestyles:
        {None, 'solid', 'dashed', 'dashdot', 'dotted'}

    usetex
    colormap
    colors
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
    colorbar_ticks
    colorbar_tick_labels

    ticksize
    xticks
    yticks
    minor_tick_length
    major_tick_length

    labelsize
    xlabel
    ylabel
    title
    title_fontsize
    saveto
    plot_type:
        {'contour', 'contourf'}
    colorbar_only
    pad_inches
    dpi
    intermediate
    magnitude
    top_right_bounds

    aspect
    outline :
        The field outline.
    outline_linewidth :
    outline_color :

    Returns
    -------

    """

    # -- OUTLINE of the field ---------------------------------------------------
    if outline is None:
        outline_plot_type = ''
        outline_plot_info = {}
    else:
        from phyem.msehtt.static.mesh.partial.main import MseHttMeshPartial
        if isinstance(outline, MseHttMeshPartial):
            from phyem.msehtt.static.mesh.partial.elements.main import MseHttElementsPartialMesh
            # ----------- msehtt partial mesh --------------------------------------------
            composition = outline.composition
            if isinstance(composition, MseHttElementsPartialMesh):
                outline_plot_type = 'msehtt elements'
                outline_plot_info = composition._outline_plot_info()

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError(f"cannot parse the outline of the field from object: {outline}.")

    # ================================================================================================

    x = {0: x} if not isinstance(x, dict) else x
    y = {0: y} if not isinstance(y, dict) else y
    v = {0: v} if not isinstance(v, dict) else v

    if magnitude:
        for patch in v:
            _ = np.abs(v[patch])
            _[_ < 1e-16] = 1e-16
            _ = np.log10(_)
            v[patch] = _
    else:
        pass

    plt.rcParams.update({
        "text.usetex": usetex,
    })

    if usetex:
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

    if colormap is not None:
        plt.rcParams['image.cmap'] = colormap
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    # ------- label and  ticks -------------------------------------------------------
    if top_right_bounds:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)

    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    ax.tick_params(labelsize=ticksize)
    plt.tick_params(axis='both', which='minor', direction='out', length=minor_tick_length)
    plt.tick_params(axis='both', which='major', direction='out', length=major_tick_length)

    if levels is None:
        assert num_levels is not None, f"num_levels cannot be None when 'levels' is not provided."
        assert isinstance(num_levels, (int, float)) and num_levels > 0 and num_levels % 1 == 0, \
            f"num_levels = {num_levels} wrong. It must be a positive integer."
        if level_range is None:
            levels = ___set_contour_levels___(v, num_levels)
        else:
            assert len(level_range) == 2 and level_range[0] < level_range[1], \
                f"level range must be two increasing number"
            l_bound, u_bound = level_range
            levels = np.linspace(l_bound, u_bound, num_levels)
    else:
        pass
    # -------------- contour plot --------------------------------------------------------
    for patch in v:
        v_patch = v[patch]
        if plot_type == 'contour':
            if colors is None:
                plt.contour(
                    x[patch], y[patch], v_patch,
                    levels=levels, linewidths=linewidth,
                    linestyles=linestyles,
                )
            else:
                plt.contour(
                    x[patch], y[patch], v_patch,
                    levels=levels, linewidths=linewidth,
                    linestyles=linestyles,
                    colors=colors,
                )
        elif plot_type == 'contourf':
            VAL = v_patch
            VAL[VAL > levels[-1]] = levels[-1]
            VAL[VAL < levels[0]] = levels[0]
            plt.contourf(x[patch], y[patch], VAL, levels=levels)
        else:
            raise Exception(f"plot_type={plot_type} is wrong. Should be one of ('contour', 'contourf')")

    # ------ PLOT OUTLINE IF EXISTS ---------------------------------------------
    if outline is None:
        pass
    else:
        if outline_plot_type == 'msehtt elements':
            for face_info in outline_plot_info:
                # element_type, element_index, face_index = face_info
                data = outline_plot_info[face_info]
                if len(data) == 2 and np.ndim(data[0]) == 1 and np.ndim(data[1]) == 1 and len(data[0]) == len(data[1]):
                    x, y = data
                    plt.plot(x, y, c=outline_color, linewidth=outline_linewidth)
                else:
                    raise Exception()
        else:
            raise NotImplementedError()

    # --------------- title -----------------------------------------------------
    if title is None or title is False:
        pass
    elif isinstance(title, str):
        plt.title(title, fontsize=title_fontsize)
    else:
        raise Exception(f"title must be a string.")
    
    # -------------------------------- color bar ---------------------------------
    if show_colorbar:
        mappable = cm.ScalarMappable()
        mappable.set_array(np.array(levels))
        cb = plt.colorbar(mappable, ax=ax,
                          extend=colorbar_extend,
                          aspect=colorbar_aspect,
                          orientation=colorbar_orientation)

        if colorbar_ticks is None:
            pass
        else:
            if colorbar_tick_labels is None:
                cb.set_ticks(ticks=colorbar_ticks)
            else:
                # we can customize the colorbar ticks.
                cb.set_ticks(ticks=colorbar_ticks, labels=colorbar_tick_labels)

        if colorbar_label is not None:
            cb.set_label(colorbar_label, labelpad=10, size=15)

        cb.ax.tick_params(labelsize=colorbar_labelsize)

    if aspect is None:
        pass
    else:
        ax.set_aspect(aspect)

    # ---------------------- save to ---------------------------------------------
    if colorbar_only:
        ax.remove()
    else:
        pass
    if intermediate:
        return fig
    else:
        if saveto is None:
            from src.config import _setting
            matplotlib.use('TkAgg')
            plt.show(block=_setting['block'])
        else:
            plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)
            plt.close()

        plt.close()
        # --------------------------------------------------------------------------
        return fig
