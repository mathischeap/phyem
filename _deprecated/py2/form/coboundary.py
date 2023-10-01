# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from random import random
from time import time


class MseHyPy2RootFormCoboundary(Frozen):
    """"""

    def __init__(self, rf):
        self._f = rf
        self._freeze()

    def __getitem__(self, t_g):
        """"""
        return self._f.d()[t_g]

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
        # note that in order to make this work, we should have the df msehy-py space ready at the background.
        # otherwise, the _objective will be empty.
        return df
