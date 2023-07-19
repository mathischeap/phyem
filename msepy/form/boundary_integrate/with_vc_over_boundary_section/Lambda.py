# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:28 PM on 7/14/2023
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class BoundaryIntegrateVCBSLambda(Frozen):
    """Basically, it do ``<vc | tr form>``, and ``self._f`` is the ``form`` and vc is provided."""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def __call__(self, t, vc, msepy_boundary_section):
        """``vc`` will be evaluated at ``t``.

        Parameters
        ----------
        t
        vc
        msepy_boundary_section

        Returns
        -------

        """
        f = self._f  # the form, ``<vc | tr form>``
        m, n, k = f.space.m, f.space.n, f.space.abstract.k
        orientation = f.space.abstract.orientation

        assert msepy_boundary_section.n == n - 1, f"The dimensions of the boundary section must be 1 lower."

        if n == m == 2 and k == 1 and orientation == 'outer':

            return self._m2_n2_k1_outer(t, vc, msepy_boundary_section)

        else:
            raise NotImplementedError()

    def _m2_n2_k1_outer(self, t, vc, msepy_boundary_section):
        # <tr star 2-f | tr 1-f>, self._f is the 1form, and the vc is 'tr star 2-f', a scalar

        vc_ndim = vc.ndim
        vc_shape = vc.shape
        assert vc_shape == (1, ) and vc_ndim == 2, f"Need a scalar in time + 2d-space."

        quad_degree = [_ + 1 for _ in self._f.space[self._f._degree].p]

        quad_degree = max(quad_degree)

        nodes, weights = Quadrature(quad_degree, category='Lobatto').quad  # must use Lobatto

        outward_unit_normal_vector = msepy_boundary_section.ct.outward_unit_normal_vector(nodes)

        involved_mesh_elements = msepy_boundary_section.faces._elements_m_n[0, :]

        ones = np.array([1])
        N_nodes = (-ones, nodes)   # x -
        S_nodes = (ones, nodes)    # x +
        W_nodes = (nodes, -ones)   # y -
        E_nodes = (nodes, ones)    # y +

        N_RM = self._f.reconstruct_matrix(*N_nodes, element_range=involved_mesh_elements)
        S_RM = self._f.reconstruct_matrix(*S_nodes, element_range=involved_mesh_elements)
        W_RM = self._f.reconstruct_matrix(*W_nodes, element_range=involved_mesh_elements)
        E_RM = self._f.reconstruct_matrix(*E_nodes, element_range=involved_mesh_elements)

        bi_data = np.zeros(
            (
                msepy_boundary_section.base.elements._num,
                self._f.space.num_local_dofs(self._f._degree)
            )
        )

        for i in outward_unit_normal_vector:
            face = msepy_boundary_section.faces[i]
            local_dofs = face.find_corresponding_local_dofs_of(self._f)
            element, m, n = face._element, face._m, face._n
            onv = outward_unit_normal_vector[i]
            xy = face.ct.mapping(nodes)

            if m == 0 and n == 0:    # North
                v = N_RM[element]
            elif m == 0 and n == 1:  # South
                v = S_RM[element]
            elif m == 1 and n == 0:  # West
                v = W_RM[element]
            elif m == 1 and n == 1:  # East
                v = E_RM[element]
            else:
                raise Exception()

            vx, vy = v
            vx = vx.T
            vy = vy.T
            vx = vx[local_dofs, :]
            vy = vy[local_dofs, :]
            nx, ny = onv
            trStar_vc = vc(t, *xy)[0]

            if face.is_orthogonal():
                length = face.length
                Jacobian = length / 2
                trace_f = vx * nx + vy * ny
                boundary_integration = np.sum(trStar_vc * trace_f * weights * Jacobian, axis=1)
            else:
                raise NotImplementedError()

            bi_data[element, local_dofs] = boundary_integration

        return bi_data