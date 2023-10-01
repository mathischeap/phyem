# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class MsePyReconstructMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._e2c = space.mesh.elements._index_mapping._e2c
        self._freeze()

    def __call__(self, degree, *meshgrid_xi_et_sg, element_range=None):
        """"""

        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if element_range is None:
            element_range = range(self._mesh.elements._num)
        elif element_range.__class__.__name__ in ('float', 'int', 'int32', 'int64'):
            element_range = [element_range, ]
        else:
            pass

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(
                degree, element_range, *meshgrid_xi_et_sg
            )
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(
                degree, element_range, *meshgrid_xi_et_sg
            )

    def _m1_n1_k0(self, degree, element_range, *meshgrid_xi):
        """"""
        xi = meshgrid_xi[0]
        assert np.ndim(xi) == 1, f"I need 1d xi"
        _, bf = self._space.basis_functions[degree](*meshgrid_xi)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:

            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                x0 = bf[0].T
                rm_cache[cache_index] = (x0, )

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m1_n1_k1(self, degree, element_range, *meshgrid_xi):
        """"""
        xi = meshgrid_xi[0]
        assert np.ndim(xi) == 1, f"I need 1d xi"
        xi, bf = self._space.basis_functions[degree](*meshgrid_xi)
        iJ = self._mesh.elements.ct.inverse_Jacobian(*xi, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:
            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                ij = iJ[e]
                x0 = bf[0] * ij
                rm_cache[cache_index] = (x0.T, )

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m2_n2_k0(self, degree, element_range, *meshgrid_xi_et):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        _, bf = self._space.basis_functions[degree](*meshgrid_xi_et)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:

            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                x0 = bf[0].T
                rm_cache[cache_index] = (x0, )

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m2_n2_k1_outer(self, degree, element_range, *meshgrid_xi_et):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, bf = self._space.basis_functions[degree](*meshgrid_xi_et)
        u, v = bf
        iJ = self._mesh.elements.ct.inverse_Jacobian_matrix(*xi_et, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:

            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                assert e in iJ, f"element #{e} is out of range."
                iJ0, iJ1 = iJ[e]
                iJ00, iJ01 = iJ0
                iJ10, iJ11 = iJ1

                x0 = + u * iJ11
                x1 = - v * iJ01
                rm_e_x = np.vstack((x0, x1)).T

                y0 = - u * iJ10
                y1 = + v * iJ00
                rm_e_y = np.vstack((y0, y1)).T

                rm_cache[cache_index] = (rm_e_x, rm_e_y)

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m2_n2_k1_inner(self, degree, element_range, *meshgrid_xi_et):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, bf = self._space.basis_functions[degree](*meshgrid_xi_et)
        u, v = bf
        iJ = self._mesh.elements.ct.inverse_Jacobian_matrix(*xi_et, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:

            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                assert e in iJ, f"element #{e} is out of range."
                iJ0, iJ1 = iJ[e]
                iJ00, iJ01 = iJ0
                iJ10, iJ11 = iJ1

                x0 = u * iJ00
                x1 = v * iJ10
                rm_e_x = np.vstack((x0, x1)).T

                y0 = u * iJ01
                y1 = v * iJ11
                rm_e_y = np.vstack((y0, y1)).T

                rm_cache[cache_index] = (rm_e_x, rm_e_y)

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m2_n2_k2(self, degree, element_range, *meshgrid_xi_et):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, bf = self._space.basis_functions[degree](*meshgrid_xi_et)
        iJ = self._mesh.elements.ct.inverse_Jacobian(*xi_et, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:
            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                ij = iJ[e]
                x0 = bf[0] * ij
                rm_cache[cache_index] = (x0.T, )

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m3_n3_k0(self, degree, element_range, *meshgrid_xi_et_sg):
        """"""
        xi, et, sg = meshgrid_xi_et_sg
        assert np.ndim(xi) == np.ndim(et) == np.ndim(sg) == 1, f"I need 1d xi and et"
        _, bf = self._space.basis_functions[degree](*meshgrid_xi_et_sg)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:

            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                x0 = bf[0].T
                rm_cache[cache_index] = (x0, )

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m3_n3_k1(self, degree, element_range, *meshgrid_xi_et_sg):
        """"""
        xi, et, sg = meshgrid_xi_et_sg
        assert np.ndim(xi) == np.ndim(et) == np.ndim(sg) == 1, f"I need 1d xi, et and sg."
        xi_et_sg, bf = self._space.basis_functions[degree](*meshgrid_xi_et_sg)
        u, v, w = bf
        iJ = self._mesh.elements.ct.inverse_Jacobian_matrix(*xi_et_sg, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in rm_cache:
                pass

            else:
                assert e in iJ, f"element #{e} is out of range."
                iJ0, iJ1, iJ2 = iJ[e]
                iJ00, iJ01, iJ02 = iJ0
                iJ10, iJ11, iJ12 = iJ1
                iJ20, iJ21, iJ22 = iJ2
                x0 = u * iJ00
                x1 = v * iJ10
                x2 = w * iJ20
                rm_e_x = np.vstack((x0, x1, x2)).T

                y0 = u * iJ01
                y1 = v * iJ11
                y2 = w * iJ21
                rm_e_y = np.vstack((y0, y1, y2)).T

                z0 = u * iJ02
                z1 = v * iJ12
                z2 = w * iJ22
                rm_e_z = np.vstack((z0, z1, z2)).T

                rm_cache[cache_index] = (rm_e_x, rm_e_y, rm_e_z)

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m3_n3_k2(self, degree, element_range, *meshgrid_xi_et_sg):
        """"""
        xi, et, sg = meshgrid_xi_et_sg
        assert np.ndim(xi) == np.ndim(et) == np.ndim(sg) == 1, f"I need 1d xi, et and sg."
        xi_et_sg, bf = self._space.basis_functions[degree](*meshgrid_xi_et_sg)
        u, v, w = bf
        iJ = self._mesh.elements.ct.inverse_Jacobian_matrix(*xi_et_sg, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in rm_cache:
                pass

            else:
                assert e in iJ, f"element #{e} is out of range."
                ij = iJ[e]
                x0 = u * (ij[1][1]*ij[2][2] - ij[1][2]*ij[2][1])
                x1 = v * (ij[2][1]*ij[0][2] - ij[2][2]*ij[0][1])
                x2 = w * (ij[0][1]*ij[1][2] - ij[0][2]*ij[1][1])
                rm_e_x = np.vstack((x0, x1, x2)).T

                y0 = u * (ij[1][2]*ij[2][0] - ij[1][0]*ij[2][2])
                y1 = v * (ij[2][2]*ij[0][0] - ij[2][0]*ij[0][2])
                y2 = w * (ij[0][2]*ij[1][0] - ij[0][0]*ij[1][2])
                rm_e_y = np.vstack((y0, y1, y2)).T

                z0 = u * (ij[1][0]*ij[2][1] - ij[1][1]*ij[2][0])
                z1 = v * (ij[2][0]*ij[0][1] - ij[2][1]*ij[0][0])
                z2 = w * (ij[0][0]*ij[1][1] - ij[0][1]*ij[1][0])
                rm_e_z = np.vstack((z0, z1, z2)).T

                rm_cache[cache_index] = (rm_e_x, rm_e_y, rm_e_z)

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m3_n3_k3(self, degree, element_range, *meshgrid_xi_et_sg):
        """"""
        xi, et, sg = meshgrid_xi_et_sg
        assert np.ndim(xi) == np.ndim(et) == np.ndim(sg) == 1, f"I need 1d xi and et"
        xi_et_sg, bf = self._space.basis_functions[degree](*meshgrid_xi_et_sg)
        iJ = self._mesh.elements.ct.inverse_Jacobian(*xi_et_sg, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:
            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                ij = iJ[e]
                x0 = bf[0] * ij
                rm_cache[cache_index] = (x0.T, )

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict
