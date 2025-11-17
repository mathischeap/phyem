# -*- coding: utf-8 -*-
"""
"""
from phyem.tools.frozen import Frozen
from importlib import import_module
from phyem.msehtt.static.mesh.partial.boundary_section.main import MseHttBoundarySectionPartialMesh


class MseHttStaticForm_Boundary_Integrate(Frozen):
    """"""
    def __init__(self, f):
        """"""
        self._f = f
        self._freeze()

    def with_vc_over_boundary_section(self, t, vc, boundary_section):
        """"""
        assert boundary_section.composition.__class__ is MseHttBoundarySectionPartialMesh, \
            f"The boundary section must be a {MseHttBoundarySectionPartialMesh}."
        assert boundary_section.tgm is self._f.tgm, f"the great mesh must match."
        boundary_section = boundary_section.composition
        assert vc._is_time_space_func(), f"provided vc is not a vector calculus object."

        space = self._f.space
        m = space.m
        n = space.n
        indicator = space.indicator

        if indicator == 'Lambda':
            k = space.abstract.k
            orientation = space.abstract.orientation
            indicator = f"m{m}n{n}k{k}"
            path = self.__repr__().split('main.')[0][1:] + f"vc.Lambda.bi_vc_{indicator}"
            module = import_module(path)
            if hasattr(module, 'bi_vc__' + indicator):
                return getattr(module, 'bi_vc__' + indicator)(
                    self._f, t, vc, boundary_section)
            else:
                return getattr(module, 'bi_vc__' + indicator + f"_{orientation}")(
                    self._f, t, vc, boundary_section)

        else:
            raise NotImplementedError(
                f"BI with_vc_over_boundary_section for {indicator} not implemented.")
