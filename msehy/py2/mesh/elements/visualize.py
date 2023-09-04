# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})


class MseHyPy2MeshElementsVisualize(Frozen):
    """"""

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._freeze()

    def _make_base_mesh(self, sampling_factor=1):
        """"""
        base_mesh_data_Lines = self._elements.background.visualize._generate_mesh_grid_data(
            sampling_factor=sampling_factor
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.xlabel(r"$x$", fontsize=15)
        plt.ylabel(r"$y$", fontsize=15)
        plt.tick_params(axis='both', which='both', labelsize=12)

        axis0 = [-1, 1]
        for i in base_mesh_data_Lines:  # region # i
            lines = base_mesh_data_Lines[i]
            for j, axis_lines in enumerate(lines):
                axis0, axis1 = axis_lines
                if j == 0:
                    plt.plot(axis0, axis1, linewidth=1, color='lightgray')
                elif j == 1:
                    plt.plot(axis0.T, axis1.T, linewidth=1, color='lightgray')
                else:
                    raise Exception
        density = len(axis0)
        return fig, density

    def __call__(self, sampling_factor=1, saveto=None):
        """"""
        fig, density = self._make_base_mesh(sampling_factor=sampling_factor)

        if density < 20:
            density = 20
        else:
            pass

        for level in self._elements._levels:
            fig = level._visualize(fig, density, color='k')

        # save -----------------------------------------------------------
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _setting, _pr_cache

            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehy_py2_mesh')
            else:
                matplotlib.use('TkAgg')
                plt.tight_layout()
                plt.show(block=_setting['block'])

        return fig
