# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.config import RANK, MASTER_RANK
from tools.frozen import Frozen
from tools._mpi import merge_dict
from generic.py.tools.vtk_ import BuildVtkUnStruct
from generic.py.gathering_matrix import PyGM


class MPI_PY_2D_FORM_VISUALIZE(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        self._mesh = f.mesh
        self._freeze()

    def __call__(
            self,
            *other_forms,
            t=None,
            saveto=None, sampling_factor=1,
            data_only=False, builder=True
    ):
        """

        Parameters
        ----------
        other_forms
        t
        saveto
        sampling_factor
        data_only
        builder

        Returns
        -------

        """
        if len(other_forms) == 0:  # save only this one form.
            if saveto is None:
                saveto = 'mpi_generic_py_2d_form'
            else:
                pass
            indicator = self._f._space.abstract.indicator

            if indicator in ('Lambda', ):
                k = self._f.space.abstract.k
                return getattr(self, f'_Lambda_k{k}')(
                    t,
                    saveto, sampling_factor,
                    data_only=data_only, builder=builder
                )
            else:
                raise NotImplementedError()

        else:   # we save a couple of forms together with this form.
            if saveto is None:
                saveto = 'mpi_generic_py_2d_form'
            else:
                pass

            vtk_builder, v = self.__call__(
                t=t,
                saveto=None, sampling_factor=sampling_factor,
                data_only=True, builder=True,
            )

            for of in other_forms:
                v_of = of.visualize(
                    t=t,
                    saveto=None, sampling_factor=sampling_factor,
                    data_only=True, builder=False,
                )
                if RANK == MASTER_RANK:
                    v.update(v_of)

            if RANK == MASTER_RANK:
                vtk_builder(saveto, point_data=v)

            return 0

    def _parse_p_nodes(self, sampling_factor, anti_top_corner_singularity=0):
        """"""
        p = self._f.space[self._f.degree].p
        p = int(p*sampling_factor)
        if p < 1:
            p = 1
        else:
            pass
        interval = 0.03
        _ = np.linspace(-1 + anti_top_corner_singularity*interval, 1, p+1)
        nodes = [_, _]
        return p, nodes

    def _Lambda_k0(
            self,
            t,
            saveto, sampling_factor,
            data_only=False, builder=True
    ):
        """"""
        p, nodes = self._parse_p_nodes(sampling_factor)

        xy, v = self._f[t].reconstruct(*nodes, ravel=False)
        x, y = xy
        v = v[0]
        x, y, v = merge_dict(x, y, v, root=MASTER_RANK)

        if RANK == MASTER_RANK:

            if data_only:
                if builder:
                    numbering = self._f.space.gathering_matrix.Lambda._k0_master(p)
                    numbering = PyGM(numbering)
                    vtk_builder = BuildVtkUnStruct(numbering, (x, y), cell_layout=p)
                    return vtk_builder, {self._f.name: v}
                else:
                    return {self._f.name: v}

            else:
                numbering = self._f.space.gathering_matrix.Lambda._k0_master(p)
                numbering = PyGM(numbering)
                vtk_builder = BuildVtkUnStruct(numbering, (x, y), cell_layout=p)
                vtk_builder(saveto, point_data={self._f.name: v})
                return 0

        else:
            if data_only:
                if builder:
                    return None, None
                else:
                    return None
            else:
                return None

    def _Lambda_k2(
            self,
            t,
            saveto, sampling_factor,
            data_only=False, builder=True
    ):
        """"""
        p, nodes = self._parse_p_nodes(sampling_factor, anti_top_corner_singularity=2)

        xy, v = self._f[t].reconstruct(*nodes, ravel=False)
        x, y = xy
        v = v[0]
        x, y, v = merge_dict(x, y, v, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            if data_only:
                if builder:
                    numbering = self._f.space.gathering_matrix.Lambda._k0_master(p)
                    numbering = PyGM(numbering)
                    vtk_builder = BuildVtkUnStruct(numbering, (x, y), cell_layout=p)
                    return vtk_builder, {self._f.name: v}
                else:
                    return {self._f.name: v}

            else:
                numbering = self._f.space.gathering_matrix.Lambda._k0_master(p)
                numbering = PyGM(numbering)
                vtk_builder = BuildVtkUnStruct(numbering, (x, y), cell_layout=p)
                vtk_builder(saveto, point_data={self._f.name: v})
                return 0

        else:
            if data_only:
                if builder:
                    return None, None
                else:
                    return None
            else:
                return None

    def _Lambda_k1(
            self,
            t,
            saveto, sampling_factor,
            data_only=False, builder=True
    ):
        """"""
        p, nodes = self._parse_p_nodes(sampling_factor, anti_top_corner_singularity=1)

        xy, v = self._f[t].reconstruct(*nodes, ravel=False)
        x, y = xy
        u, v = v
        x, y, u, v = merge_dict(x, y, u, v, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            if data_only:
                if builder:
                    numbering = self._f.space.gathering_matrix.Lambda._k0_master(p)
                    numbering = PyGM(numbering)
                    vtk_builder = BuildVtkUnStruct(numbering, (x, y), cell_layout=p)
                    return vtk_builder, {self._f.name: (u, v)}
                else:
                    return {self._f.name: (u, v)}

            else:
                numbering = self._f.space.gathering_matrix.Lambda._k0_master(p)
                numbering = PyGM(numbering)
                vtk_builder = BuildVtkUnStruct(numbering, (x, y), cell_layout=p)
                vtk_builder(saveto, point_data={self._f.name: (u, v)})
                return 0

        else:
            if data_only:
                if builder:
                    return None, None
                else:
                    return None
            else:
                return None
