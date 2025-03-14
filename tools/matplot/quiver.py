# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def quiver(
        X, Y, U, V,
        title=None,
        usetex=True, colormap='Spectral', xlim=None, ylim=None,
        show_colorbar=True,
        colorbar_label=None, colorbar_orientation='vertical', colorbar_aspect=20,
        colorbar_labelsize=15, colorbar_extend='both',
        colorbar_ticks=None,
        quiverkey='1<->1',
        ticksize=15,
        labelsize=15,
        saveto=None, dpi=200,
):
    """Could be very badly distributed arrows for non-uniform meshes. Try to use visualization
    of discrete forms.

    Parameters
    ----------
    X
    Y
    U
    V
    title
    usetex
    colormap
    xlim
    ylim
    show_colorbar
    colorbar_label
    colorbar_orientation
    colorbar_aspect
    colorbar_labelsize
    colorbar_extend
    colorbar_ticks
    ticksize

    quiverkey : str
        This defines the indicator: an arrow.
        For example:
            `quiverkey = '1 <-> text'`

            This gives a showcase arrow whose length is 1. Then we can compare it
            with the arrows to see what are they length.

            The text 'text' will be added beside the showcase arrow.

    labelsize
    saveto
    dpi:
        The dpi for pixel based figures.

    Returns
    -------

    """
    assert all([isinstance(_, np.ndarray) for _ in (X, Y, U, V)]), f"X, Y, U, V must all be ndarray."
    assert np.shape(X) == np.shape(Y) == np.shape(U) == np.shape(V), f"X, Y, U, V must be same shape."
    if np.ndim(X) > 1:
        X = X.ravel('F')
        Y = Y.ravel('F')
        U = U.ravel('F')
        V = V.ravel('F')
    else:
        pass

    M = np.hypot(U, V)

    # ---- check if zero field, if it is quiver will return warning, so we skip it ---------
    U_max, U_min = np.max(U), np.min(U)
    V_max, V_min = np.max(V), np.min(V)
    if U_max - U_min == 0 and V_max - V_min == 0:
        ZERO_FIELD = True
    else:
        ZERO_FIELD = False

    # -------------------------------------------------------------------------
    if saveto is not None:
        matplotlib.use('TkAgg')
    plt.rc('text', usetex=usetex)
    plt.rcParams.update({
        "font.family": "Times New Roman"
    })
    plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # noinspection PyUnresolvedReferences
    ax.spines['top'].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines['right'].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines['left'].set_visible(True)
    # noinspection PyUnresolvedReferences
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel(r"$x$", fontsize=labelsize)
    ax.set_ylabel(r"$y$", fontsize=labelsize)
    plt.tick_params(axis='both', which='both', labelsize=ticksize)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if ZERO_FIELD:
        pass
    else:
        if show_colorbar:
            # M = M / np.max(M) normalize to max == 1
            norm = matplotlib.colors.Normalize()
            if colorbar_ticks is None:
                pass
            else:
                tMin, tMax = min(colorbar_ticks), max(colorbar_ticks)
                assert tMax > tMin, f"colorbar_ticks={colorbar_ticks} wrong!"
                assert tMin >= 0, f"quiver tick can not be lower than 0!"
                LARGE = M > tMax
                M[LARGE] = tMax
                LOW = M < tMin
                M[LOW] = tMin
                M = np.concatenate((M, [tMin, tMax]))

            norm.autoscale(M)
            # cm = getattr(matplotlib.cm, colormap)
            cm = matplotlib.colormaps.get_cmap(colormap)
            sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
            sm.set_array([])
            ax.quiver(X, Y, U, V, color=cm(norm(M)))

            cbar = fig.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax,
                orientation=colorbar_orientation,
                extend=colorbar_extend,
                aspect=colorbar_aspect,
            )
            if colorbar_label is not None:
                cbar.set_label(colorbar_label, labelpad=10, size=colorbar_labelsize)
            if colorbar_ticks is not None:
                cbar.set_ticks(colorbar_ticks)           # set colorbar tick size
                cbar.ax.tick_params(labelsize=ticksize)  # set colorbar tick size

        else:
            if colormap is not None:
                plt.rcParams['image.cmap'] = 'Greys'
            Q = ax.quiver(X, Y, U, V, M, color='k')

            assert '<->' in quiverkey, " <Quiver> : quiverkey={} format wrong.".format(quiverkey)
            value, quivertext = quiverkey.split('<->')
            try:
                value = int(value)
            except ValueError:
                raise Exception(
                    " <Quiver>: quiverkey={} format wrong. "
                    "value (before '<->') is not int.".format(quiverkey))

            ax.quiverkey(Q, 0.8, 0.9, value, quivertext, labelpos='E', coordinates='figure')

    # =========== super title ==============================================================
    if title is None or title == '':
        pass
    else:
        plt.title(title)

    # ---------------------- save to --------------------------------------------------------
    if saveto is None or saveto is False:
        from src.config import _setting
        plt.show(block=_setting['block'])

    else:
        if saveto[-4:] == '.pdf':
            plt.savefig(saveto, bbox_inches='tight')
        else:
            plt.savefig(saveto, dpi=dpi, bbox_inches='tight')
        plt.close()

    return fig
