# -*- coding: utf-8 -*-
"""
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams.update({
    "font.family": "DejaVu sans",
})


def scatter(
        x, y, z,
        usetex=True, saveto=None, figsize=(8, 6), colormap='bwr', pad_inches=0.1, marker='s',
        title=None,
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
    plt.scatter(X, Y, c=Z, cmap=colormap, marker=marker)
    plt.colorbar()
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
        plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches)
        plt.close()
    else:
        from src.config import _setting
        plt.show(block=_setting['block'])
    # =============================================================================================1
    return 0
