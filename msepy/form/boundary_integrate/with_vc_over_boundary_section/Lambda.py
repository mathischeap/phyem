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
    """Basically, it does ``<vc | tr form>``, and ``self._f`` is the ``form`` and vc is provided."""

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
        f = self._f  # the form, i.e. ``f`` (instead of ``tr f``) of ``<vc | tr f>``
        m, n, k = f.space.m, f.space.n, f.space.abstract.k
        orientation = f.space.abstract.orientation

        assert msepy_boundary_section.n == n - 1, f"The dimensions of the boundary section must be 1 lower."

        if n == m == 2 and k == 1 and orientation == 'outer':

            return self._m2_n2_k1_outer(t, vc, msepy_boundary_section)

        elif n == m == 2 and k == 0 and orientation == 'inner':

            return self._m2_n2_k0_inner(t, vc, msepy_boundary_section)

        else:
            raise NotImplementedError(
                f"BoundaryIntegrateVCBSLambda: not implemented for `form (<vc | tr form>)` in : "
                f"m{m} n{n}, k{k}, orientation: {orientation}"
            )

    def _m2_n2_k0_inner(self, t, vc, msepy_boundary_section):
        # <tr star 1-f | tr 0form>, self._f is the 0form (inner), and the vc is representing the 'tr star 1-f', a scalar

        if vc.shape == (2, ) and vc.ndim == 2:  # we received a vector.
            # this means we do not pre-compute the flux of the vector (which then is a scalar)
            # over the boundary section

            quad_degree = [_ + 1 for _ in self._f.space[self._f._degree].p]
            quad_degree = max(quad_degree)
            nodes, weights = Quadrature(quad_degree, category='Lobatto').quad  # must use Lobatto
            outward_unit_normal_vector = msepy_boundary_section.ct.outward_unit_normal_vector(nodes)
            N_elements, S_elements, W_elements, E_elements = list(), list(), list(), list()
            for element, m, n in zip(*msepy_boundary_section.faces._elements_m_n):
                if m == 0:
                    if n == 0:
                        N_elements.append(element)
                    elif n == 1:
                        S_elements.append(element)
                    else:
                        raise Exception()
                elif m == 1:
                    if n == 0:
                        W_elements.append(element)
                    elif n == 1:
                        E_elements.append(element)
                    else:
                        raise Exception()
                else:
                    raise Exception()

            ones = np.array([1])
            N_nodes = (-ones, nodes)   # x -
            S_nodes = (ones, nodes)    # x +
            W_nodes = (nodes, -ones)   # y -
            E_nodes = (nodes, ones)    # y +

            N_RM = self._f.reconstruction_matrix(*N_nodes, element_range=N_elements)
            S_RM = self._f.reconstruction_matrix(*S_nodes, element_range=S_elements)
            W_RM = self._f.reconstruction_matrix(*W_nodes, element_range=W_elements)
            E_RM = self._f.reconstruction_matrix(*E_nodes, element_range=E_elements)

            bi_data = np.zeros(
                (
                    msepy_boundary_section.base.elements._num,
                    self._f.space.num_local_dofs(self._f._degree)
                )
            )

            for i in outward_unit_normal_vector:  # go through all local element faces in this boundary section.
                # ``i`` is the local index, start from 0, end with num-1 while num is the amount of element faces.

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

                v = v[0].T
                tr_0f = v[local_dofs]   # <~ | tr 0-f>

                nx, ny = onv
                vx, vy = vc(t, *xy)   # this vector calculus object is for all regions.

                trStar_1f = vx * nx + vy * ny   # <trStar 1-f | ~>

                # print(trStar_1f.shape, tr_0f.shape)
                if face.is_orthogonal():
                    length = face.length
                    Jacobian = length / 2
                else:
                    JM = face.ct.Jacobian_matrix(nodes)
                    Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)

                boundary_integration = np.sum(trStar_1f * tr_0f * weights * Jacobian, axis=1)
                bi_data[element, local_dofs] = boundary_integration

            return bi_data

        else:
            raise NotImplementedError()

    def _m2_n2_k1_outer(self, t, vc, msepy_boundary_section):
        # <tr star 2-f | tr 1form>, self._f is the 1form (outer), and the vc is representing the 'tr star 2-f', a scalar

        assert vc.shape == (1, ) and vc.ndim == 2, f"Need a scalar in time + 2d-space."
        quad_degree = [_ + 1 for _ in self._f.space[self._f._degree].p]
        quad_degree = max(quad_degree)
        nodes, weights = Quadrature(quad_degree, category='Lobatto').quad  # must use Lobatto
        outward_unit_normal_vector = msepy_boundary_section.ct.outward_unit_normal_vector(nodes)
        N_elements, S_elements, W_elements, E_elements = list(), list(), list(), list()
        for element, m, n in zip(*msepy_boundary_section.faces._elements_m_n):
            if m == 0:
                if n == 0:
                    N_elements.append(element)
                elif n == 1:
                    S_elements.append(element)
                else:
                    raise Exception()
            elif m == 1:
                if n == 0:
                    W_elements.append(element)
                elif n == 1:
                    E_elements.append(element)
                else:
                    raise Exception()
            else:
                raise Exception()

        ones = np.array([1])
        N_nodes = (-ones, nodes)   # x -
        S_nodes = (ones, nodes)    # x +
        W_nodes = (nodes, -ones)   # y -
        E_nodes = (nodes, ones)    # y +

        N_RM = self._f.reconstruction_matrix(*N_nodes, element_range=N_elements)
        S_RM = self._f.reconstruction_matrix(*S_nodes, element_range=S_elements)
        W_RM = self._f.reconstruction_matrix(*W_nodes, element_range=W_elements)
        E_RM = self._f.reconstruction_matrix(*E_nodes, element_range=E_elements)

        bi_data = np.zeros(
            (
                msepy_boundary_section.base.elements._num,
                self._f.space.num_local_dofs(self._f._degree)
            )
        )

        for i in outward_unit_normal_vector:  # go through all local element faces in this boundary section.
            # ``i`` is the local index, start from 0, end with num-1 while num is the amount of element faces.
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
            vx = vx[local_dofs]
            vy = vy[local_dofs]
            nx, ny = onv
            trStar_vc = vc(t, *xy)[0]   # this vector calculus object is for all regions; # <trStar_vc | ~>
            trace_f = vx * nx + vy * ny  # <~ | trace-f>
            if face.is_orthogonal():
                length = face.length
                Jacobian = length / 2
            else:
                JM = face.ct.Jacobian_matrix(nodes)
                Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)

            boundary_integration = np.sum(trStar_vc * trace_f * weights * Jacobian, axis=1)
            bi_data[element, local_dofs] = boundary_integration

        return bi_data
