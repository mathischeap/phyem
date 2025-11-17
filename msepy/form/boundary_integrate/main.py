# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh
from phyem.src.spaces.continuous.Lambda import ScalarValuedFormSpace

from phyem.msepy.form.boundary_integrate.with_vc_over_boundary_section.Lambda import BoundaryIntegrateVCBSLambda


class MsePyRootFormBoundaryIntegrate(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._freeze()

    def with_vc_over_boundary_section(self, t, vc, msepy_boundary_section):
        """``vc`` will be evaluated at time ``t``.

        Parameters
        ----------
        t
        vc
        msepy_boundary_section

        Returns
        -------

        """
        assert msepy_boundary_section.__class__ is MsePyBoundarySectionMesh and \
            msepy_boundary_section._base is self._mesh, \
            f"the provided msepy boundary section is not a part of the mesh boundary!"

        assert vc._is_time_space_func(), f"provided vc is not a vector calculus object."

        space = self._f.space
        if space.abstract.__class__ is ScalarValuedFormSpace:
            # scalar valued space boundary integrate

            return BoundaryIntegrateVCBSLambda(self._f)(
                t, vc, msepy_boundary_section
            )

        else:
            raise NotImplementedError(f"Not implemented for space {space.abstract.__class__}.")
