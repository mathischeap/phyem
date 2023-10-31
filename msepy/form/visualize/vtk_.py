# -*- coding: utf-8 -*-
r"""
Export to corresponding vtk files.
"""
import numpy as np
from tools.frozen import Frozen
from msepy.tools.vtk_ import BuildVtkHexahedron, BuildVtkQuad


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
            saveto=None, sampling_factor=1,
            data_only=False, builder=True,   # cannot add **kwargs
    ):
        if len(other_forms) == 0:  # save only this one form.
            if saveto is None:
                saveto = self._f.name
            else:
                pass
            abs_sp = self._f.space.abstract
            m = abs_sp.m
            n = abs_sp.n
            k = abs_sp.k

            indicator = self._f._space.abstract.indicator

            if indicator in ('Lambda', ):
                return getattr(self, f'_Lambda_m{m}_n{n}_k{k}')(
                    saveto, sampling_factor,
                    data_only=data_only, builder=builder
                )
            else:
                raise NotImplementedError()

        else:   # we save a couple of forms together with this form.
            if saveto is None:
                saveto = 'msepy_forms_vtk'
            else:
                pass

            vtk_builder, v = self.__call__(
                saveto=None, sampling_factor=sampling_factor,
                data_only=True, builder=True,
            )

            from msepy.form.main import MsePyRootForm
            from msepy.form.static import MsePyRootFormStaticCopy

            for of in other_forms:
                if of.__class__ is MsePyRootFormStaticCopy:
                    v_of = of.visualize.vtk(
                        saveto=None, sampling_factor=sampling_factor,
                        data_only=True, builder=False,
                    )
                    v.update(v_of)
                elif of.__class__ is MsePyRootForm:
                    v_of = of[self._f.visualize._t].visualize.vtk(
                        saveto=None, sampling_factor=sampling_factor,
                        data_only=True, builder=False,
                    )
                    v.update(v_of)
                else:
                    raise NotImplementedError()

            vtk_builder(saveto, point_data=v)

    def _Lambda_m3_n3_k0(
            self, saveto, sampling_factor,
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
            vtk_builder(saveto, point_data={self._f.name: v})

    def _Lambda_m3_n3_k1(
            self, saveto, sampling_factor,
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
                return vtk_builder, {self._f.name: v, }
            else:
                return {self._f.name: v, }

        else:
            vtk_builder = BuildVtkHexahedron(x, y, z, cell_layout=p)
            vtk_builder(saveto, point_data={self._f.name: v, })

    def _Lambda_m3_n3_k2(self, *args, **kwargs):
        """"""
        return self._Lambda_m3_n3_k1(*args, **kwargs)

    def _Lambda_m3_n3_k3(self, *args, **kwargs):
        """"""
        return self._Lambda_m3_n3_k0(*args, **kwargs)

    def _Lambda_m2_n2_k0(
              self, saveto, sampling_factor,
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
                vtk_builder = BuildVtkQuad(x, y, cell_layout=p)
                return vtk_builder, {self._f.name: v}
            else:
                return {self._f.name: v}

        else:
            vtk_builder = BuildVtkQuad(x, y, cell_layout=p)
            vtk_builder(saveto, point_data={self._f.name: v})

    def _Lambda_m2_n2_k1(
            self, saveto, sampling_factor,
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
                vtk_builder = BuildVtkQuad(x, y, cell_layout=p)
                return vtk_builder, {self._f.name: v, }
            else:
                return {self._f.name: v, }

        else:
            vtk_builder = BuildVtkQuad(x, y, cell_layout=p)
            vtk_builder(saveto, point_data={self._f.name: v, })

    def _Lambda_m2_n2_k2(self, *args, **kwargs):
        """"""
        return self._Lambda_m2_n2_k0(*args, **kwargs)
