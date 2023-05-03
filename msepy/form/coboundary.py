# -*- coding: utf-8 -*-
"""
phyem@RAM-EEMCS-UT
Yi Zhang
"""

import sys

import numpy as np

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from random import random
from time import time


class MsePyRootFormCoboundary(Frozen):
    """"""

    def __init__(self, rf):
        self._f = rf
        self._freeze()

    def __getitem__(self, t):
        """"""
        return MsePyRootFormCoboundaryTimeInstant(self._f, t)

    @property
    def incidence_matrix(self):
        """E."""
        return self._f.space.incidence_matrix(self._f.degree)  # cached in space.


class MsePyRootFormCoboundaryTimeInstant(Frozen):
    """"""

    def __init__(self, rf, t):
        self._f = rf
        self._t = t
        assert self._t in self._f.cochain, \
            f"{self._f} has no cochain at time={self._t}, cannot perform coboundary."
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split(' object')[1]
        f_repr = self._f.__repr__()
        return f"<Coboundary of " + f_repr + f' @ t = {self._t}' + super_repr

    def __call__(self):
        """We will make a new root-form, no connection to `self._f` will be stored. Also, we will not
        update the abstract root global cache.
        """
        from msepy.main import new
        ab_space = self._f.abstract.space
        d_ab_space = ab_space.d()
        d_msepy_space = new(d_ab_space)  # make msepy space, must using this function.
        sym_repr = str(hash(random() + time()))      # random sym_repr  <-- important, do not try to print its repr
        lin_repr = str(hash(random() + time() + 2))  # random lin_repr  <-- important, do not try to print its repr
        # The below abstract root-form is not recorded.
        ab_df = self._f.abstract.__class__(d_ab_space, sym_repr, lin_repr, True, update_cache=False)
        d_ab_space.finite.specify_form(ab_df, self._f.degree)
        df = self._f.__class__(ab_df)
        assert df.space is d_msepy_space, f"must be!"
        incidence_matrix = self._f.coboundary.incidence_matrix
        cochain_at_t = self._f.cochain[self._t].local
        d_cochain_at_t = np.einsum(
            'ij, kj -> ki',
            incidence_matrix.toarray(), cochain_at_t,
            optimize='optimal',
        )
        df.cochain._set(self._t, d_cochain_at_t)
        df.cochain._locker = True  # lock the cochain, this df is only for `self._t`.
        cf = self._f.cf.field

        if cf is None:
            pass
        else:
            vc_operator = self._corresponding_vc_operators
            new_cf = dict()
            for i in cf:
                new_cf[i] = getattr(cf[i], vc_operator)
            df.cf = new_cf

        return df
    
    @property
    def _corresponding_vc_operators(self):
        """"""
        space = self._f.space.abstract
        space_indicator = space.indicator
        if space_indicator == 'Lambda':  # scalar valued form spaces.
            m, n, k = space.m, space.n, space.k
            if m == n == 1 and k == 0:  # 0-form on 1d manifold in 1d space.
                return 'derivative'
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    # python msepy/form/coboundary.py
    pass
