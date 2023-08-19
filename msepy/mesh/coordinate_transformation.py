# -*- coding: utf-8 -*-
r"""
"""
import sys
if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
import numpy as np
from typing import Dict


class MsePyMeshCoordinateTransformation(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self.___cache_mapping_od___ = dict()  # cache for mapping.
        self.___cache_JM___ = None
        self._freeze()

    def mapping(self, *xi_et_sg, regions=None):
        """The mapping for elements in regions."""
        if regions is None:
            regions = range(0, len(self._mesh.manifold.regions))
        elif isinstance(regions, int):
            regions = [regions]
        else:
            raise Exception(f"pls compute mapping for one region or all regions!")

        elements = self._mesh.elements
        origin = elements._origin
        delta = elements._delta
        _xyz = dict()

        assert len(xi_et_sg) == self._mesh.ndim, f"I need {self._mesh.ndim} reference coordinates."

        for i in regions:
            key = elements._layout_cache_key[i]
            if key in self.___cache_mapping_od___:
                ori, dta, num_local_elements = self.___cache_mapping_od___[key]
            else:
                oi = origin[i]
                di = delta[i]
                length = [len(_) for _ in oi]

                if self._mesh.ndim == 1:   # 1-d mapping
                    ox = oi[0]
                    dx = di[0]
                    ori = [ox]
                    dta = [dx]

                elif self._mesh.ndim == 2:   # 2-d mapping
                    ox = np.tile(oi[0], length[1])
                    dx = np.tile(di[0], length[1])
                    oy = np.repeat(oi[1], length[0])
                    dy = np.repeat(di[1], length[0])
                    ori = [ox, oy]
                    dta = [dx, dy]

                elif self._mesh.ndim == 3:    # 3-d mapping
                    ox = np.tile(np.tile(oi[0], length[1]), length[2])
                    dx = np.tile(np.tile(di[0], length[1]), length[2])
                    oy = np.tile(np.repeat(oi[1], length[0]), length[2])
                    dy = np.tile(np.repeat(di[1], length[0]), length[2])
                    oz = np.repeat(np.repeat(oi[2], length[1]), length[0])
                    dz = np.repeat(np.repeat(di[2], length[1]), length[0])
                    ori = [ox, oy, oz]
                    dta = [dx, dy, dz]

                else:
                    raise NotImplementedError()

                num_local_elements = np.prod(length)
                self.___cache_mapping_od___[key] = ori, dta, num_local_elements

            md_ref_coo = list()
            for j, ref_coo in enumerate(xi_et_sg):
                _ = ref_coo[..., np.newaxis].repeat(num_local_elements, axis=-1)
                _ = (_ + 1) * 0.5 * dta[j] + ori[j]
                md_ref_coo.append(_)

            md_ref_coo = self._mesh.manifold.ct.mapping(*md_ref_coo, regions=i)[i]
            _xyz[i] = md_ref_coo

        if len(regions) == 1:
            return _xyz[regions[0]]
        else:
            xyz = [list() for _ in range(len(_xyz[0]))]
            for i in regions:
                for j, region_axis_value in enumerate(_xyz[i]):  # axis_value, elements
                    xyz[j].append(region_axis_value)
                del _xyz[i]
            for j, _ in enumerate(xyz):
                xyz[j] = np.concatenate(list(_), axis=-1)

            return xyz   # x, y, z, ... = xyz and a[..., i] (a in {x, y, z, ...})for element #i

    def Jacobian_matrix(self, *xi_et_sg):
        """The Jacobian matrix for each element.

        As it is computed through element index mapping, it will be computed for all elements.
        """
        if self.___cache_JM___ is None:
            eim = self._mesh.elements._index_mapping
            reference_elements = eim._reference_elements
            reference_delta = eim._reference_delta
            reference_origin = eim._reference_origin
            reference_regions = eim._reference_regions

            elements: Dict[int] = dict()  # Dict keys: region index
            origin: Dict[int] = dict()    # Dict keys: region index
            delta: Dict[int] = dict()     # Dict keys: region index
            for i, re in enumerate(reference_elements):
                region = reference_regions[i]
                if region not in origin:
                    elements[region] = list()
                    origin[region] = list()
                    delta[region] = list()
                else:
                    pass

                elements[region].append(re)
                origin[region].append(reference_origin[i])
                delta[region].append(reference_delta[i])

            for r in origin:
                origin[r] = np.array(origin[r]).T
                delta[r] = np.array(delta[r]).T

            self.___cache_JM___ = [elements, origin, delta]
        else:
            elements, origin, delta = self.___cache_JM___

        JM = dict()
        for r in elements:
            ele = elements[r]
            ori = origin[r]
            dta = delta[r]

            jm = list()
            for j, ref_coo in enumerate(xi_et_sg):
                _ = ref_coo[..., np.newaxis].repeat(len(ele), axis=-1)
                _ = (_ + 1) * 0.5 * dta[j] + ori[j]
                jm.append(_)

            jm = self._mesh.manifold.ct.Jacobian_matrix(*jm, regions=r)[r]

            ref_Jacobian = dta / 2
            s0 = len(jm)
            s1 = len(jm[0])
            for e in ele:
                assert e not in JM, f"trivial check, a reference element must appear once."
                JM[e] = tuple([[0 for _ in range(s0)] for _ in range(s1)])

            for i in range(s0):
                for j in range(s1):
                    jm_ij = jm[i][j]
                    if isinstance(jm_ij, int) and jm_ij == 0:
                        pass

                    else:
                        assert jm_ij.__class__.__name__ == 'ndarray', \
                            'Trivial check. Make sure we use ones_like.'
                        jm_ij *= ref_Jacobian[j]
                        for k, e in enumerate(ele):
                            JM[e][i][j] = jm_ij[..., k]

        return self._mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(JM)

    def Jacobian(self, *xi_et_sg, JM=None):
        """the Determinant of the Jacobian matrix. When Jacobian matrix is square, Jacobian = sqrt(g)."""
        if JM is None:
            JM = self.Jacobian_matrix(*xi_et_sg)
        else:
            pass

        m, n = self._mesh.m, self._mesh.n

        Jacobian_dict = {}
        for re in JM:
            jm = JM[re]

            if m == n == 1:
                Ji = jm[0][0]

            elif m == n == 2:
                Ji = jm[0][0]*jm[1][1] - jm[0][1]*jm[1][0]

            elif m == n == 3:
                Ji = \
                    + jm[0][0]*jm[1][1]*jm[2][2] + jm[0][1]*jm[1][2]*jm[2][0] \
                    + jm[0][2]*jm[1][0]*jm[2][1] - jm[0][0]*jm[1][2]*jm[2][1] \
                    - jm[0][1]*jm[1][0]*jm[2][2] - jm[0][2]*jm[1][1]*jm[2][0]
            else:
                raise Exception()

            Jacobian_dict[re] = Ji

        return self._mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(Jacobian_dict)

    def metric(self, *xi_et_sg, detJ=None):
        """ For square Jacobian matrix,
        the metric ``g:= det(G):=(det(J))**2``, where ``G`` is the metric matrix, or metric tensor.
        """
        m, n = self._mesh.m, self._mesh.n

        if detJ is None:
            detJ = self.Jacobian(*xi_et_sg)
        else:
            pass

        metric = dict()

        for re in detJ:
            det_j = detJ[re]

            if m == n:

                metric_re = det_j ** 2

            else:
                raise NotImplementedError()

            metric[re] = metric_re

        return self._mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(metric)

    def inverse_Jacobian_matrix(self, *xi_et_sg, JM=None):
        """The inverse Jacobian matrix. """
        m, n = self._mesh.m, self._mesh.n

        if JM is None:
            JM = self.Jacobian_matrix(*xi_et_sg)
        else:
            pass

        inverse_Jacobian_matrix_dict = {}

        for re in JM:

            jm = JM[re]

            if m == n == 1:

                iJM00 = 1 / jm[0][0]
                inverse_Jacobian_matrix_dict[re] = [[iJM00, ], ]

            elif m == n == 2:

                reciprocalJacobian = 1 / (jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0])
                iJ00 = + reciprocalJacobian * jm[1][1]
                iJ01 = - reciprocalJacobian * jm[0][1]
                iJ10 = - reciprocalJacobian * jm[1][0]
                iJ11 = + reciprocalJacobian * jm[0][0]
                inverse_Jacobian_matrix_dict[re] = \
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

                inverse_Jacobian_matrix_dict[re] = [
                    [iJ00, iJ01, iJ02],
                    [iJ10, iJ11, iJ12],
                    [iJ20, iJ21, iJ22],
                ]

            else:
                raise NotImplementedError()

        return self._mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(
            inverse_Jacobian_matrix_dict
        )

    def inverse_Jacobian(self, *xi_et_sg, iJM=None):
        """the Determinant of the inverse Jacobian matrix."""
        m, n = self._mesh.m, self._mesh.n

        if iJM is None:
            iJM = self.inverse_Jacobian_matrix(*xi_et_sg)
        else:
            pass

        inverse_Jacobian_dict = {}

        for re in iJM:
            ijm = iJM[re]

            if m == n == 1:
                iJ = ijm[0][0]

            elif m == n == 2:
                iJ = ijm[0][0]*ijm[1][1] - ijm[0][1]*ijm[1][0]

            elif m == n == 3:
                iJ = \
                    + ijm[0][0]*ijm[1][1]*ijm[2][2] + ijm[0][1]*ijm[1][2]*ijm[2][0] \
                    + ijm[0][2]*ijm[1][0]*ijm[2][1] - ijm[0][0]*ijm[1][2]*ijm[2][1] \
                    - ijm[0][1]*ijm[1][0]*ijm[2][2] - ijm[0][2]*ijm[1][1]*ijm[2][0]
            else:
                raise Exception()

            inverse_Jacobian_dict[re] = iJ

        return self._mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(
            inverse_Jacobian_dict
        )

    def metric_matrix(self, *xi_et_sg, JM=None):
        """
        Also called metric tensor. Let J be the Jacobian matrix. The ``metricMatrix`` is
        denoted by G, G := J^T.dot(J). And the metric is ``g := (det(J))**2 or g := det(G).``
        Which means for a square Jacobian matrix, the metric turns out to be the square of the
        determinant of the Jacobian matrix.

        The entries of G are normally denoted as g_{i,j}.
        """
        if JM is None:
            JM = self.Jacobian_matrix(*xi_et_sg)
        else:
            pass

        m, n = self._mesh.m, self._mesh.n

        metric_matrix_dict = {}

        for re in JM:
            jm = JM[re]

            G = [[None for _ in range(n)] for __ in range(n)]

            for i in range(n):
                for j in range(i, n):
                    G[i][j] = jm[0][i] * jm[0][j]
                    for L in range(1, m):
                        G[i][j] += jm[L][i] * jm[L][j]
                    if i != j:
                        G[j][i] = G[i][j]

            metric_matrix_dict[re] = G

        return self._mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(
            metric_matrix_dict
        )

    def inverse_metric_matrix(self, *xi_et_sg, iJM=None):
        """
        The ``inverseMetricMatrix`` is the metric matrix of the inverse Jacobian matrix
        or the metric of the inverse mapping. It is usually denoted as G^{-1}.

        The entries of G^{-1} is normally denoted as g^{i,j}.
        """
        if iJM is None:
            iJM = self.inverse_Jacobian_matrix(*xi_et_sg)
        else:
            pass

        m, n = self._mesh.m, self._mesh.n

        inverse_metric_matrix_dict = {}

        for re in iJM:
            ijm = iJM[re]

            iG = [[None for _ in range(m)] for __ in range(m)]
            for i in range(m):
                for j in range(i, m):
                    # noinspection PyTypeChecker
                    iG[i][j] = ijm[i][0] * ijm[j][0]
                    for L in range(1, n):
                        iG[i][j] += ijm[i][L] * ijm[j][L]
                    if i != j:
                        iG[j][i] = iG[i][j]

            inverse_metric_matrix_dict[re] = iG

        return self._mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(
            inverse_metric_matrix_dict
        )


if __name__ == '__main__':
    # python msepy/mesh/coordinate_transformation.py
    import __init__ as ph

    space_dim = 3
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)

    msepy, obj = ph.fem.apply('msepy', locals())

    mnf = obj['manifold']
    msh = obj['mesh']

    msepy.config(mnf)('crazy', c=0., periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    # msepy.config(mnf)('backward_step')
    msepy.config(msh)([5 for _ in range(space_dim)])
    # msepy.config(msh)(([1, 2, 1], [2, 3], [1, 2, 2, 4]))
    # msepy.config(msh)(([1, 2, 2], [2, 3]))

    # xi_et_sg = [np.array([-0.5, 0, 0.25, 0.5]) for _ in range(space_dim)]
    # xi_et_sg = [np.linspace(-1, 1, 4) for _ in range(space_dim)]
    xi_et_sg = [np.random.rand(7, 11) for _ in range(space_dim)]

    xyz = msh.ct.mapping(*xi_et_sg)
    # jm = msh.ct.inverse_metric_matrix(*xi_et_sg)
    jm = msh.ct.inverse_metric_matrix(*xi_et_sg)
    msh.visualize(sampling_factor=1)
