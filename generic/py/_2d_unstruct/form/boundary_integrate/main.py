# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from generic.py._2d_unstruct.mesh.boundary_section.main import BoundarySection

from generic.py._2d_unstruct.form.boundary_integrate.with_vc_over_boundary_section.Lambda import (
    Boundary_Integrate_VC_BS_Lambda)


class Boundary_Integrate(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._freeze()

    def with_vc_over_boundary_section(self, t, vc, boundary_section):
        """``vc`` will be evaluated at time ``t``.

        Parameters
        ----------
        t
        vc
        boundary_section

        Returns
        -------

        """
        assert boundary_section.__class__ is BoundarySection, f'need a boundary section'
        assert boundary_section.base is self._mesh, f'boundary section does not match the mesh.'

        space = self._f.space
        if space.abstract.indicator == 'Lambda':
            # scalar valued space boundary integrate
            return Boundary_Integrate_VC_BS_Lambda(self._f)(
                t, vc, boundary_section
            )

        else:
            raise NotImplementedError(f"Not implemented for space {space.abstract.__class__}.")
