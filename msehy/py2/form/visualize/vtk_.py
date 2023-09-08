# -*- coding: utf-8 -*-
r"""
Export to corresponding vtk files.
"""
import numpy as np
from tools.frozen import Frozen
from msehy.py2.tools.vtk_ import BuildVtkUnStruct


class MseHyPy2RootFormVisualizeVTK(Frozen):
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
            data_only=False, builder=True,   # cannot add **kwargs
    ):
        if len(other_forms) == 0:  # save only this one form.
            if file_path is None:
                file_path = self._f.name
            else:
                pass
            abs_sp = self._f.space.abstract
            k = abs_sp.k

            indicator = self._f._space.abstract.indicator

            if indicator in ('Lambda', ):
                return getattr(self, f'_Lambda_k{k}')(
                    file_path, sampling_factor,
                    data_only=data_only, builder=builder
                )
            else:
                raise NotImplementedError()

        else:   # we save a couple of forms together with this form.
            if file_path is None:
                file_path = 'msepy_forms_vtk'
            else:
                pass

            vtk_builder, v = self.__call__(
                file_path=None, sampling_factor=sampling_factor,
                data_only=True, builder=True,
            )

            from msepy.form.main import MsePyRootForm
            from msepy.form.static import MsePyRootFormStaticCopy

            for of in other_forms:
                if of.__class__ is MsePyRootFormStaticCopy:
                    v_of = of.visualize.vtk(
                        file_path=None, sampling_factor=sampling_factor,
                        data_only=True, builder=False,
                    )
                    v.update(v_of)
                elif of.__class__ is MsePyRootForm:
                    v_of = of[self._f.visualize._t].visualize.vtk(
                        file_path=None, sampling_factor=sampling_factor,
                        data_only=True, builder=False,
                    )
                    v.update(v_of)
                else:
                    raise NotImplementedError()

            vtk_builder(file_path, point_data=v)

            return 0

    def _Lambda_k0(
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
        xy, v = self._f[t].reconstruct(*nodes, ravel=True)
        x, y = xy
        v = v[0]

        if data_only:
            if builder:
                vtk_builder = BuildVtkUnStruct(x, y, cell_layout=p)
                return vtk_builder, {self._f.name: v}
            else:
                return {self._f.name: v}

        else:
            vtk_builder = BuildVtkUnStruct(x, y, cell_layout=p)
            vtk_builder(file_path, point_data={self._f.name: v})

            return 0

    def _Lambda_k1(
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
        xy, v = self._f[t].reconstruct(*nodes, ravel=True)
        x, y = xy

        if data_only:
            if builder:
                vtk_builder = BuildVtkUnStruct(x, y, cell_layout=p)
                return vtk_builder, {self._f.name: v, }
            else:
                return {self._f.name: v, }

        else:
            vtk_builder = BuildVtkUnStruct(x, y, cell_layout=p)
            vtk_builder(file_path, point_data={self._f.name: v, })

            return 0

    def _Lambda_k2(
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
        xy, v = self._f[t].reconstruct(*nodes, ravel=True)
        x, y = xy
        v = v[0]

        if data_only:
            if builder:
                vtk_builder = BuildVtkUnStruct(x, y, cell_layout=p)
                return vtk_builder, {self._f.name: v}
            else:
                return {self._f.name: v}

        else:
            vtk_builder = BuildVtkUnStruct(x, y, cell_layout=p)
            vtk_builder(file_path, point_data={self._f.name: v})

            return 0
