# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM, MPI
from _MPI.generic.py._2d_unstruct.mesh.elements.main import MPI_Py_2D_Unstructured_MeshElements
from _MPI.generic.py._2d_unstruct.mesh.boundary_section.coordinate_transformation import _MPI_PY_2d_BS_CT


class MPI_Py_2D_Unstructured_BoundarySection(Frozen):
    """"""

    def __init__(self, base_elements, including_element_faces):
        assert base_elements.__class__ is MPI_Py_2D_Unstructured_MeshElements, \
            f"must based on an elements-class."

        including_element_faces = COMM.gather(including_element_faces, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            _all = list()
            for _ in including_element_faces:
                _all.extend(_)
            including_element_faces = _all
        else:
            pass
        including_element_faces = COMM.bcast(including_element_faces, root=MASTER_RANK)
        local_element_faces = list()
        for ef in including_element_faces:
            element_index = ef[0]
            if element_index in base_elements:
                local_element_faces.append(ef)
        self._base = base_elements
        self._indices = local_element_faces  # the element-face-indices that involved in this rank.
        self._ct = _MPI_PY_2d_BS_CT(self)
        self._num_total_covered_faces = None
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return rf"<MPI-py-2d-BoundarySection @RANK:{RANK}" + super_repr

    @property
    def ct(self):
        """coordinate transformation."""
        return self._ct

    @property
    def base(self):
        """The base generic mpi version 2d mesh elements."""
        return self._base

    def __iter__(self):
        """go through all indices of local faces."""
        for index in self._indices:
            yield index

    def __contains__(self, index):
        """If this index indicating a local face?"""
        return index in self._indices

    def __len__(self):
        """How many local faces is boundary-section covers?"""
        return len(self._indices)

    @property
    def num_total_covered_faces(self):
        """The total amount of faces across all ranks this boundary section is covering."""
        if self._num_total_covered_faces is None:
            num_local_faces = len(self)
            self._num_total_covered_faces = COMM.allreduce(num_local_faces, op=MPI.SUM)
        return self._num_total_covered_faces

    def __getitem__(self, index):
        """"""
        return self._base._make_boundary_face(index)

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

        # base elements data -------------------------------------------
        base_lines = list()
        for index in self.base:
            element = self.base[index]
            lines = element._plot_lines(density)
            base_lines.append(lines)
        base_lines = COMM.gather(base_lines, root=MASTER_RANK)

        xi = np.linspace(-1, 1, density)
        edge_lines = list()
        for index in self:
            face = self[index]
            edge_line = face.ct.mapping(xi)
            edge_lines.append(edge_line)
        edge_lines = COMM.gather(edge_lines, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            return
        else:
            pass

        import matplotlib.pyplot as plt
        import matplotlib
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
        })

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

        for lines in base_lines:
            for Ls in lines:
                for line in Ls:
                    plt.plot(*line, linewidth=0.5, color='lightgray')
        for lines in edge_lines:
            for line in lines:
                plt.plot(*line, linewidth=1)

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
