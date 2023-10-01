# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
from tools.frozen import Frozen
from generic.py._2d_unstruct.mesh.main import GenericUnstructuredMesh2D
from generic.py._2d_unstruct.mesh.boundary_section.coordinate_transformation import Boundary_Section_CT


class BoundarySection(Frozen):
    """"""

    def __init__(self, base_mesh, element_edge_index):
        """"""
        assert base_mesh.__class__ is GenericUnstructuredMesh2D, \
            f"the base must be a {GenericUnstructuredMesh2D}."
        self._base = base_mesh
        self._indices = element_edge_index  # the indices of faces of this boundary section.
        self._ct = Boundary_Section_CT(self)
        self._involved_elements = None
        self._freeze()

    @property
    def base(self):
        """the base mesh; I am representing a boundary section of this base mesh."""
        return self._base

    def __iter__(self):
        """go through all indices of local faces."""
        for index in self._indices:
            yield index

    def __contains__(self, index):
        """If this index indicating a local face?"""
        return index in self._indices

    def __len__(self):
        """How many local faces?"""
        return len(self._indices)

    def __getitem__(self, index):
        """"""
        return self._base._boundary_faces(index)

    @property
    def ct(self):
        return self._ct

    def _find_involved_element(self):
        """return all mesh elements that are involved for this boundary section."""
        if self._involved_elements is None:
            self._involved_elements = list()
            for index in self:
                element_index = index[0]
                if element_index in self._involved_elements:
                    pass
                else:
                    self._involved_elements.append(element_index)
        return self._involved_elements

        # -------- methods ----------------------------------------------------------------------
    def visualize(
            self,
            density=10,
            top_right_bounds=False,
            saveto=None,
            dpi=200,
            title=None,
    ):
        """"""
        if density < 10:
            density = 10
        else:
            pass
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        if top_right_bounds:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.xlabel(r"$x$", fontsize=15)
        plt.ylabel(r"$y$", fontsize=15)
        plt.tick_params(axis='both', which='both', labelsize=12)
        for index in self.base:
            element = self.base[index]
            lines = element._plot_lines(density)
            for line in lines:
                plt.plot(*line, linewidth=0.5, color='lightgray')
        xi = np.linspace(-1, 1, 20)
        for index in self:
            face = self[index]
            x, y = face.ct.mapping(xi)
            plt.plot(x, y, linewidth=1)

        if title is None:
            pass
        else:
            plt.title(title)

        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight', dpi=dpi)
        else:
            from src.config import _setting, _pr_cache

            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehy_py2_boundary_section_mesh')
            else:
                matplotlib.use('TkAgg')
                plt.tight_layout()
                plt.show(block=_setting['block'])
        plt.close()
        return fig
