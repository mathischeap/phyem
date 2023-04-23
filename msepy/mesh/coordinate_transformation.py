# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
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
        self.___cache_mapping_od___ = dict()  # cache #1 for mapping.
        self.___cache_JM_od___ = dict()  # cache #1 for mapping.
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
            return _xyz[0]
        else:
            xyz = [list() for _ in range(len(_xyz[0]))]
            for i in regions:
                for j, region_axis_value in enumerate(_xyz[i]):  # axis_value, elements
                    xyz[j].append(region_axis_value)
                del _xyz[i]
            for j, _ in enumerate(xyz):
                xyz[j] = np.concatenate(list(_), axis=-1)

            return xyz

    def Jacobian_matrix(self, *xi_et_sg):
        """The Jacobian matrix for each element.

        As it is computed through element index mapping, it will be computed for all elements.
        """
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
                        assert jm_ij.__class__.__name__ == 'ndarray', 'Trivial check. Make sure we use ones_like.'
                        jm_ij *= ref_Jacobian[j]
                        for k, e in enumerate(ele):
                            JM[e][i][j] = jm_ij[..., k]

        return self._mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(JM)


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

    msepy.config(mnf)('crazy', c=0.3, periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    # msepy.config(mnf)('backward_step')
    msepy.config(msh)([5 for _ in range(space_dim)])
    # msepy.config(msh)(([1, 2, 1], [2, 3], [1, 2, 2, 4]))
    # msepy.config(msh)(([1, 2, 2], [2, 3]))

    # xi_et_sg = [np.array([-0.5, 0, 0.25, 0.5]) for _ in range(space_dim)]
    # xi_et_sg = [np.linspace(-1, 1, 4) for _ in range(space_dim)]
    xi_et_sg = [np.random.rand(7, 11) for _ in range(space_dim)]

    xyz = msh.ct.mapping(*xi_et_sg)
    jm = msh.ct.Jacobian_matrix(*xi_et_sg)
    print(jm(2))

    # msh.visualize(refining_factor=1)
