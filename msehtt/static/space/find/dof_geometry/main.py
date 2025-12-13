# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.find.dof_geometry.local.findLocalDof_Lambda import _find_geo_local_dof_m2n2k0_
from phyem.msehtt.static.space.find.dof_geometry.local.findLocalDof_Lambda import _find_geo_local_dof_m2n2k1_inner_
from phyem.msehtt.static.space.find.dof_geometry.local.findLocalDof_Lambda import _find_geo_local_dof_m2n2k1_outer_
from phyem.msehtt.static.space.find.dof_geometry.local.findLocalDof_Lambda import _find_geo_local_dof_m2n2k2_

from phyem.msehtt.static.space.find.dof_geometry.local.findLocalDof_Lambda import _find_geo_local_dof_m3n3k0_
from phyem.msehtt.static.space.find.dof_geometry.local.findLocalDof_Lambda import _find_geo_local_dof_m3n3k1_
from phyem.msehtt.static.space.find.dof_geometry.local.findLocalDof_Lambda import _find_geo_local_dof_m3n3k2_
from phyem.msehtt.static.space.find.dof_geometry.local.findLocalDof_Lambda import _find_geo_local_dof_m3n3k3_


class MseHttSpace_FindDofGeometry(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        self._freeze()

    def local_dof(self, degree, e, i):
        r"""For the space, a geometric object of the local dof #`i` in element indexed `e`
        for the form under `degree` will be found first.

        For example, in m2n2, for a dof of the scalar-valued 0-form space. i.e. Lambda-k0, we will return a
        m2n2-point, i.e. `Point2`.

        """
        indicator = self._space.str_indicator
        element = self._space.tpm.composition[e]
        if indicator == 'm2n2k0':
            return _find_geo_local_dof_m2n2k0_(degree, element, i)
        elif indicator == 'm2n2k1_inner':
            return _find_geo_local_dof_m2n2k1_inner_(degree, element, i)
        elif indicator == 'm2n2k1_outer':
            return _find_geo_local_dof_m2n2k1_outer_(degree, element, i)
        elif indicator == 'm2n2k2':
            return _find_geo_local_dof_m2n2k2_(degree, element, i)
        elif indicator == 'm3n3k0':
            return _find_geo_local_dof_m3n3k0_(degree, element, i)
        elif indicator == 'm3n3k1':
            return _find_geo_local_dof_m3n3k1_(degree, element, i)
        elif indicator == 'm3n3k2':
            return _find_geo_local_dof_m3n3k2_(degree, element, i)
        elif indicator == 'm3n3k3':
            return _find_geo_local_dof_m3n3k3_(degree, element, i)
        else:
            raise NotImplementedError(indicator)
