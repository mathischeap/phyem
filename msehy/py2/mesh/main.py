# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.main import base as msepy_base


class MseHyPy2Mesh(Frozen):
    """"""

    def __init__(self, abstract_mesh):
        self._abstract = abstract_mesh
        self._freeze()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    @property
    def abstract(self):
        return self._abstract

    @property
    def background(self):
        """We return it in realtime."""
        return msepy_base['meshes'][self._abstract._sym_repr]


if __name__ == '__main__':
    # python msehy/py2/mesh/main.py
    import __init__ as ph

    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim, is_periodic=False)
    mesh = ph.mesh(manifold)
    mesh.boundary_partition(r"\Gamma_\perp", r"\Gamma_P")

    msehy, obj = ph.fem.apply('msehy', locals())

    manifold = msehy.base['manifolds'][r'\mathcal{M}']
    mesh = msehy.base['meshes'][r'\mathfrak{M}']

    msehy.config(manifold)('cylinder_channel')

    # manifold.background.visualize()

    Gamma_perp = msehy.base['manifolds'][r"\Gamma_\perp"]

    msehy.config(Gamma_perp)(
        manifold, {
            0: [1, 0, 1, 0],
            1: [0, 0, 1, 1],
            2: [0, 0, 1, 0],
            3: [1, 1, 0, 0],
            4: [1, 0, 0, 0],
            5: [1, 0, 0, 1],
            6: [0, 0, 1, 1],
            7: [0, 0, 0, 1],
        }
    )
    msehy.config(mesh)(5)

    for msh in msehy.base['meshes']:
        msh = msehy.base['meshes'][msh]
        msh.background.visualize()
