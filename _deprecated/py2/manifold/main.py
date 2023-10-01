# -*- coding: utf-8 -*-
r"""
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.main import base as msepy_base


class MseHyPy2Manifold(Frozen):
    """"""

    def __init__(self, abstract_manifold):
        self._abstract = abstract_manifold
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
        return msepy_base['manifolds'][self._abstract._sym_repr]


if __name__ == '__main__':
    # python msehy/py2/manifold/main.py
    import __init__ as ph

    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim, is_periodic=False)

    msehy, obj = ph.fem.apply('msehy', locals())

    manifold = msehy.base['manifolds'][r'\mathcal{M}']

    print(manifold.background)
