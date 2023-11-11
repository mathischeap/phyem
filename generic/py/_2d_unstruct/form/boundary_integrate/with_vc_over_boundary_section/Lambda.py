# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class Boundary_Integrate_VC_BS_Lambda(Frozen):
    """Basically, it does ``<vc | tr form>``, and ``self._f`` is the ``form`` and vc is provided."""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def __call__(self, t, vc, boundary_section):
        """``vc`` will be evaluated at ``t``.

        Parameters
        ----------
        t
        vc
        boundary_section

        Returns
        -------

        """
        f = self._f  # the form, i.e. ``f`` (instead of ``tr f``) of ``<vc | tr f>``
        k = f.space.abstract.k
        orientation = f.space.abstract.orientation

        if k == 1 and orientation == 'outer':

            return self._k1_outer(t, vc, boundary_section)

        else:
            raise NotImplementedError(
                f"BoundaryIntegrateVCBSLambda: not implemented for 'form (<vc | tr form>)' in: "
                f"k{k}, orientation: {orientation}"
            )

    def _k1_outer(self, t, vc, boundary_section):
        # <tr star 2-f | tr 1form>, self._f is the 1form (outer), and the vc is representing the 'tr star 2-f', a scalar
        assert vc.shape == (1, ) and vc.ndim == 2, f"Need a scalar in time + 2d-space."
        p = self._f.space[self._f._degree].p
        nodes, weights = Quadrature(p+2, category='Gauss').quad  # must use Lobatto
        outward_unit_normal_vector = boundary_section.ct.outward_unit_normal_vector(nodes)

        ones = np.array([1])
        _0_nodes = (nodes, -ones)   # y -
        _1_nodes = (ones, nodes)    # x +
        _2_nodes = (nodes, ones)    # y +
        _3_nodes = (-ones, nodes)   # x -

        bi_data = dict()
        num_local_dofs = self._f.space.num_local_dofs(self._f._degree)

        for element_index in self._f.mesh:
            bi_data[element_index] = np.zeros(
                num_local_dofs[self._f.mesh[element_index].type]
            )

        for face_index in outward_unit_normal_vector:  # go through all local element faces in this boundary section.
            # ``i`` is the local index, start from 0, end with num-1 while num is the amount of element faces.
            face = boundary_section[face_index]
            local_dofs = face.find_local_dofs_of(self._f)
            element_index = face._element_index
            edge_index = face._edge_index
            onv = outward_unit_normal_vector[face_index]
            xy = face.ct.mapping(nodes)

            if edge_index == 0:
                v = self._f.reconstruction_matrix(*_0_nodes, element_range=[element_index])
            elif edge_index == 1:
                v = self._f.reconstruction_matrix(*_1_nodes, element_range=[element_index])
            elif edge_index == 2:
                v = self._f.reconstruction_matrix(*_2_nodes, element_range=[element_index])
            elif edge_index == 3:
                v = self._f.reconstruction_matrix(*_3_nodes, element_range=[element_index])
            else:
                raise Exception()

            v = v[element_index]
            vx, vy = v
            vx = vx.T
            vy = vy.T
            vx = vx[local_dofs]
            vy = vy[local_dofs]
            nx, ny = onv
            # below, we evaluate vc on the boundary
            trStar_vc = vc(t, *xy)[0]  # this vector calculus object (a scalar) is for all regions; # <trStar_vc | ~>
            trace_1f = vx * nx + vy * ny  # <~ | trace-f>

            JM = face.ct.Jacobian_matrix(nodes)
            Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)
            boundary_integration = np.sum(trStar_vc * trace_1f * weights * Jacobian, axis=1)
            bi_data[element_index][local_dofs] = boundary_integration

        return bi_data
