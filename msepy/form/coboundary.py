# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from random import random
from time import time
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix


class MsePyRootFormCoboundary(Frozen):
    """"""

    def __init__(self, rf):
        self._f = rf
        self._E = None
        self._freeze()

    def __getitem__(self, t):
        """"""
        return self._f.d()[t]

    @property
    def incidence_matrix(self):
        """E."""
        gm0 = self._f.space.gathering_matrix._next(self._f.degree)
        gm1 = self._f.cochain.gathering_matrix
        E = MsePyStaticLocalMatrix(  # every time we make new instance, do not cache it.
            self._f.space.incidence_matrix(self._f.degree),  # constant sparse matrix
            gm0,
            gm1,
        )
        return E

    def _make_df(self):
        """Make a copy of df of empty cochain; do not specify cochain."""
        ab_space = self._f.abstract.space
        d_ab_space = ab_space.d()
        sym_repr = str(hash(random() + time()))      # random sym_repr <-- important, do not try to print its repr
        lin_repr = str(hash(random() + time() + 2))  # random lin_repr <-- important, do not try to print its repr
        # The below abstract root-form is not recorded.
        ab_df = self._f.abstract.__class__(d_ab_space, sym_repr, lin_repr, True, update_cache=False)
        d_ab_space.finite.specify_form(ab_df, self._f.degree)
        df = self._f.__class__(ab_df)
        return df
