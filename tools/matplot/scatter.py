# -*- coding: utf-8 -*-
"""
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')
plt.rcParams.update({
    "font.family": "DejaVu sans",
})


def scatter(
        x, y, z,
        usetex=True, saveto=None, figsize=(8, 6), colormap='bwr', pad_inches=0.1, marker='s',
        title=None, xlabel=None, ylabel=None, label_size=13, color_range=None,
):
    """

    Parameters
    ----------
    x :
        For multiple patches, put different patches in a dict.
    y :
        For multiple patches, put different patches in a dict.
    z :
        For multiple patches, put different patches in a dict.
    usetex
    saveto
    figsize
    colormap
    pad_inches
    marker
    title
    xlabel
    ylabel
    label_size
    color_range

    Returns
    -------

    """
    if isinstance(x, dict):
        pass
    else:
        x = {0: x}
    if isinstance(y, dict):
        pass
    else:
        y = {0: y}
    if isinstance(z, dict):
        pass
    else:
        z = {0: z}

    # - config matplotlib -------------------------------------------------------------------------1
    if saveto is not None:
        matplotlib.use('Agg')

    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "DejaVu sans",
        # "font.serif": "Times New Roman",
    })

    if usetex:
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

    _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    X, Y, Z = [], [], []
    for patch_id in x:
        X.extend(x[patch_id].ravel())
        Y.extend(y[patch_id].ravel())
        Z.extend(z[patch_id].ravel())

    if color_range is None:
        colorbar_extend = 'neither'
    else:
        lower_bound, upper_bound = color_range
        Z = np.array(Z)
        if np.max(Z) > upper_bound:
            Z[Z > upper_bound] = upper_bound
            colorbar_extend_upper = True
        else:
            colorbar_extend_upper = False
        if np.min(Z) < lower_bound:
            Z[Z < lower_bound] = lower_bound
            colorbar_extend_lower = True
        else:
            colorbar_extend_lower = False
        if colorbar_extend_upper and colorbar_extend_lower:
            colorbar_extend = 'both'
        elif colorbar_extend_upper:
            colorbar_extend = 'max'
        elif colorbar_extend_lower:
            colorbar_extend = 'min'
        else:
            colorbar_extend = 'neither'

    plt.scatter(X, Y, c=Z, cmap=colormap, marker=marker)
    plt.colorbar(extend=colorbar_extend)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=label_size)
    # --------------- title -----------------------------------------------------
    if title is None or title is False:
        pass
    elif isinstance(title, str):
        plt.title(title)
    else:
        raise Exception(f"title must be a string.")

    # ---------------- save the figure ------------------------------------------------------------1
    plt.tight_layout()
    if saveto is not None and saveto != '':
        plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches, dpi=250)
        plt.close()
    else:
        from phyem.src.config import _setting
        plt.show(block=_setting['block'])
    # =============================================================================================1
    return 0
