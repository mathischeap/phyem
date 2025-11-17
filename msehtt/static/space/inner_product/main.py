# -*- coding: utf-8 -*-
"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.inner_product.Lambda_Lambda.main import MseHttSpace_Lambda_ip_Lambda


class MseHttSpace_InnerProduct(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda_ip_Lambda_ = None
        self._freeze()

    def __call__(
            self,        self_degree,  self_cochain,
            other_space, other_degree, other_cochain,
            inner_type='L2'
    ):
        """"""
        assert self._space.tpm is other_space.tpm, f"works only for a same grid."
        self_space = self._space
        self_indicator = self_space.indicator
        other_indicator = other_space.indicator
        if self_indicator == other_indicator == 'Lambda':
            if self._Lambda_ip_Lambda_ is None:
                self._Lambda_ip_Lambda_ = MseHttSpace_Lambda_ip_Lambda()
            else:
                pass
            return self._Lambda_ip_Lambda_(
                self_space, self_degree, self_cochain,
                other_space, other_degree, other_cochain,
                inner_type=inner_type
            )
        else:
            raise NotImplementedError()
