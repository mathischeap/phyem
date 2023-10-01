# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})


class MseHyPy2MeshFacesVisualize(Frozen):
    """"""

    def __init__(self, faces):
        """"""
        self._faces = faces
        self._freeze()

    def __call__(
            self,
            sampling_factor=1
    ):
        fig = self._faces.elements.visualize(
            sampling_factor=sampling_factor,
            color='lightgray',
            show_refining_strength_distribution=False,
            intermediate=True
        )
        from src.config import _setting, _pr_cache

        xi = np.linspace(-1, 1, 30)
        for i in self._faces:
            face = self._faces[i]
            ct = face.ct
            x, y = ct.mapping(xi)
            plt.plot(x, y, linewidth=1.25)

        sym_repr = self._faces.background.abstract._sym_repr
        plt.title(rf"${sym_repr}$")

        if _setting['pr_cache']:
            _pr_cache(fig, filename='msehy_py2_mesh')
        else:
            matplotlib.use('TkAgg')
            plt.tight_layout()
            plt.show(block=_setting['block'])

        return fig
