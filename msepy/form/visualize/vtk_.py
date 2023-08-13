# -*- coding: utf-8 -*-
r"""
Export to corresponding vtk files.
"""
import numpy as np
from tools.frozen import Frozen
from msepy.tools.vtk_ import BuildVtkHexahedron


class MsePyRootFormVisualizeVTK(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._cache = {}
        self._freeze()

    def __call__(
            self,
            *other_forms,
            file_path=None, sampling_factor=1,
            data_only=False, builder=True,
    ):
        if len(other_forms) == 0:  # save only this one form.
            if file_path is None:
                file_path = self._f.name
            else:
                pass
            abs_sp = self._f.space.abstract
            m = abs_sp.m
            n = abs_sp.n
            k = abs_sp.k
            return getattr(self, f'_m{m}_n{n}_k{k}')(
                file_path, sampling_factor,
                data_only=data_only, builder=builder
            )
        else:   # we save a couple of forms together with this form.
            if file_path is None:
                file_path = 'msepy_forms_vtk'
            else:
                pass

            vtk_builder, v = self.__call__(
                file_path=None, sampling_factor=sampling_factor,
                data_only=True, builder=True,
            )

            for of in other_forms:
                v_of = of[self._f.visualize._t].visualize(
                    file_path=None, sampling_factor=sampling_factor,
                    data_only=True, builder=False,
                )
                v.update(v_of)

            vtk_builder(file_path, point_data=v)

            return 0

    def _m3_n3_k0(
            self, file_path, sampling_factor,
            data_only=False, builder=True
    ):
        """"""
        p = self._f.space[self._f.degree].p
        p = [int(i*sampling_factor*1.5) for i in p]
        for i, p_i in enumerate(p):
            if p_i < 1:
                p[i] = 1
            else:
                pass

        nodes = [np.linspace(-1, 1, p_i+1) for p_i in p]
        t = self._f.visualize._t
        xyz, v = self._f[t].reconstruct(*nodes, ravel=True)
        x, y, z = xyz
        v = v[0]

        if data_only:
            if builder:
                vtk_builder = BuildVtkHexahedron(x, y, z, cell_layout=p)
                return vtk_builder, {self._f.name: v}
            else:
                return {self._f.name: v}
        else:
            vtk_builder = BuildVtkHexahedron(x, y, z, cell_layout=p)
            vtk_builder(file_path, point_data={self._f.name: v})

            return 0

    def _m3_n3_k1(
            self, file_path, sampling_factor,
            data_only=False, builder=True
    ):
        """"""
        p = self._f.space[self._f.degree].p
        p = [int(i*sampling_factor*1.5) for i in p]
        for i, p_i in enumerate(p):
            if p_i < 1:
                p[i] = 1
            else:
                pass
        nodes = [np.linspace(-1, 1, p_i+1) for p_i in p]
        t = self._f.visualize._t
        xyz, v = self._f[t].reconstruct(*nodes, ravel=True)
        x, y, z = xyz

        if data_only:
            if builder:
                vtk_builder = BuildVtkHexahedron(x, y, z, cell_layout=p)
                return vtk_builder, {self._f.name: v}
            else:
                return {self._f.name: v}
        else:
            vtk_builder = BuildVtkHexahedron(x, y, z, cell_layout=p)
            vtk_builder(file_path, point_data={self._f.name: v})

            return 0

    def _m3_n3_k2(self, *args, **kwargs):
        """"""
        return self._m3_n3_k1(*args, **kwargs)

    def _m3_n3_k3(self, *args, **kwargs):
        """"""
        return self._m3_n3_k0(*args, **kwargs)
