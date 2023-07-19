r"""

"""
import numpy as np
from tools.frozen import Frozen


class MsePyElement(Frozen):
    r""""""

    def __init__(self, elements, i):
        r"""The ith element in elements"""
        self._elements = elements
        self._i = i
        self._region, self._local_indices = \
            self._elements._find_region_and_local_indices_of_element(self._i)
        self._ct = MsePyElementCoordinateTransformation(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        return rf"<Element #{self._i} of {self._elements._mesh}>"

    @property
    def ct(self):
        """coordinate transformation."""
        return self._ct

    @property
    def metric_signature(self):
        """"""
        signature_dict = self._elements._element_mtype_dict
        if signature_dict is None:
            return None  # this is a unique element.
        else:
            for signature in signature_dict:
                elements = signature_dict[signature]

                if self._i in elements:
                    return signature
                else:
                    pass


class MsePyElementCoordinateTransformation(Frozen):
    """"""

    def __init__(self, element):
        """"""
        self._element = element
        self._faces = {}
        self._cache_index = self._element._elements._index_mapping._e2c[
            self._element._i
        ]
        origins = self._element._elements._origin[self._element._region]
        delta = self._element._elements._delta[self._element._region]

        self._origin = [0 for _ in range(len(origins))]
        self._delta = [0 for _ in range(len(delta))]
        for i, index in enumerate(self._element._local_indices):
            self._origin[i] = origins[i][index]
            self._delta[i] = delta[i][index]

        self._freeze()

    def __repr__(self):
        """repr"""
        return rf"<ct of {self._element}>"

    def face(self, m, n):
        """along m-axis, n(0:-, 1:+) side."""
        if (m, n) in self._faces:
            pass
        else:
            assert m % 1 == 0 and n % 1 == 0 and m >= 0 and 0 <= n <= 1, \
                f" (m, n) = {(m ,n)} is wrong."
            mesh_n = self._element._elements._mesh.n  # the mesh is n-dimensional
            assert m < mesh_n, f"m must be lower than {mesh_n} which is the dimensions of the mesh."
            self._faces[(m, n)] = _FaceCooTrans(self._element, m, n)
        return self._faces[(m, n)]

    def mapping(self, *xi_et_sg):
        """"""
        md_ref_coo = list()
        for j, _ in enumerate(xi_et_sg):
            _ = (_ + 1) * 0.5 * self._delta[j] + self._origin[j]
            md_ref_coo.append(_)

        return self._element._elements._mesh.manifold.ct.mapping(
            *md_ref_coo, regions=self._element._region
        )[self._element._region]

    def Jacobian_matrix(self, *xi_et_sg):
        """"""
        md_ref_coo = list()
        for j, _ in enumerate(xi_et_sg):
            _ = (_ + 1) * 0.5 * self._delta[j] + self._origin[j]
            md_ref_coo.append(_)

        jm = self._element._elements._mesh.manifold.ct.Jacobian_matrix(
            *md_ref_coo, regions=self._element._region
        )[self._element._region]

        s0 = len(jm)
        s1 = len(jm[0])

        JM = tuple([[0 for _ in range(s0)] for _ in range(s1)])
        for i in range(s0):
            for j in range(s1):
                jm_ij = jm[i][j]

                jm_ij *= self._delta[j] / 2

                JM[i][j] = jm_ij

        return JM

    def inverse_Jacobian_matrix(self, *xi_et_sg):
        """"""
        jm = self.Jacobian_matrix(*xi_et_sg)
        mesh = self._element._elements._mesh
        m, n = mesh.m, mesh.n

        if m == n == 1:

            iJM00 = 1 / jm[0][0]
            ijm = [[iJM00, ], ]

        elif m == n == 2:

            reciprocalJacobian = 1 / (jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0])
            iJ00 = + reciprocalJacobian * jm[1][1]
            iJ01 = - reciprocalJacobian * jm[0][1]
            iJ10 = - reciprocalJacobian * jm[1][0]
            iJ11 = + reciprocalJacobian * jm[0][0]
            ijm = \
                [
                    [iJ00, iJ01],
                    [iJ10, iJ11]
                ]

        elif m == n == 3:

            Jacobian = \
                + jm[0][0] * jm[1][1] * jm[2][2] + jm[0][1] * jm[1][2] * jm[2][0] \
                + jm[0][2] * jm[1][0] * jm[2][1] - jm[0][0] * jm[1][2] * jm[2][1] \
                - jm[0][1] * jm[1][0] * jm[2][2] - jm[0][2] * jm[1][1] * jm[2][0]

            reciprocalJacobian = 1 / Jacobian

            iJ00 = reciprocalJacobian * (jm[1][1] * jm[2][2] - jm[1][2] * jm[2][1])
            iJ01 = reciprocalJacobian * (jm[2][1] * jm[0][2] - jm[2][2] * jm[0][1])
            iJ02 = reciprocalJacobian * (jm[0][1] * jm[1][2] - jm[0][2] * jm[1][1])
            iJ10 = reciprocalJacobian * (jm[1][2] * jm[2][0] - jm[1][0] * jm[2][2])
            iJ11 = reciprocalJacobian * (jm[2][2] * jm[0][0] - jm[2][0] * jm[0][2])
            iJ12 = reciprocalJacobian * (jm[0][2] * jm[1][0] - jm[0][0] * jm[1][2])
            iJ20 = reciprocalJacobian * (jm[1][0] * jm[2][1] - jm[1][1] * jm[2][0])
            iJ21 = reciprocalJacobian * (jm[2][0] * jm[0][1] - jm[2][1] * jm[0][0])
            iJ22 = reciprocalJacobian * (jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0])

            ijm = [
                [iJ00, iJ01, iJ02],
                [iJ10, iJ11, iJ12],
                [iJ20, iJ21, iJ22]
            ]

        else:
            raise NotImplementedError()

        return ijm


class _FaceCooTrans(Frozen):
    """"""
    def __init__(self, element, m, n):
        """m-th axis, n(0:-, 1:+)-side"""
        self._element = element
        self._element_dimensions = self._element._elements._mesh.n
        self._m = m
        self._n = n
        self._freeze()

    def __repr__(self):
        """repr"""
        side = '+' if self._n == 1 else '-'
        self_repr = f"<FaceCT of {side}side along {self._m}axis of #{self._element}"
        return self_repr + '>'

    def mapping(self, *xi_et):
        """"""
        m, n = self._m, self._n
        assert len(xi_et) == self._element_dimensions - 1, f"xi_et wrong!"
        if self._element_dimensions == 2:
            t = xi_et[0]
            ones = np.ones(len(t))
            if m == 0:  # x-direction
                if n == 0:  # x-
                    return self._element.ct.mapping(-ones, t)
                elif n == 1:  # x+
                    return self._element.ct.mapping(ones, t)
                else:
                    raise Exception()
            elif m == 1:  # y-direction
                if n == 0:  # y-
                    return self._element.ct.mapping(t, -ones)
                elif n == 1:  # y+
                    return self._element.ct.mapping(t, ones)
                else:
                    raise Exception()
            else:
                raise Exception()

        else:
            raise NotImplementedError()

    def Jacobian_matrix(self, *xi_et):
        """"""
        m, n = self._m, self._n
        assert len(xi_et) == self._element_dimensions - 1, f"xi_et wrong!"
        if self._element_dimensions == 2:
            t = xi_et[0]
            ones = np.ones(len(t))
            if m == 0:  # x-direction
                if n == 0:  # x-
                    JM = self._element.ct.Jacobian_matrix(-ones, t)
                elif n == 1:  # x+
                    JM = self._element.ct.Jacobian_matrix(ones, t)
                else:
                    raise Exception()

                return JM[0][1], JM[1][1]

            elif m == 1:  # y-direction
                if n == 0:  # y-
                    JM = self._element.ct.Jacobian_matrix(t, -ones)
                elif n == 1:  # y+
                    JM = self._element.ct.Jacobian_matrix(t, ones)
                else:
                    raise Exception()

                return JM[0][0], JM[1][0]

            else:
                raise Exception()

        else:
            raise NotImplementedError()

    def outward_unit_normal_vector(self, *xi_et):
        """The outward unit norm vector."""

        assert len(xi_et) == self._element_dimensions - 1, f"xi_et wrong!"
        if self._element_dimensions == 2:  # 2-d mesh elements.

            JM = self.Jacobian_matrix(*xi_et)

            x, y = JM

            m = self._m
            n = self._n

            if m == 0 and n == 0:
                vx, vy = -y, x
            elif m == 1 and n == 1:
                vx, vy = -y, x
            else:
                vx, vy = y, -x

            magnitude = np.sqrt(vx**2 + vy**2)

            return vx / magnitude, vy / magnitude

        else:
            raise NotImplementedError(f"not implemented!")