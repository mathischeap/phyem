# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.src.config import _parse_lin_repr
from phyem.src.form.parameters import constant_scalar

_cs1 = constant_scalar(1)

_global_nop_arrays = dict()  # using pure_lin_repr as cache keys


class AbstractNonlinearOperator(Frozen):
    """"""

    def __init__(self, sym_repr, pure_lin_repr):
        """"""
        assert isinstance(sym_repr, str) and isinstance(pure_lin_repr, str), f"use string sym and lin repr."
        assert pure_lin_repr not in _global_nop_arrays, f"pure_lin_repr = {pure_lin_repr} for md array is taken."
        lin_repr, pure_lin_repr = _parse_lin_repr('multidimensional_array', pure_lin_repr)

        for _existing_mda_plr in _global_nop_arrays:
            existing_mda = _global_nop_arrays[_existing_mda_plr]
            assert existing_mda._sym_repr != sym_repr, \
                f"sym_repr={sym_repr} for md array is taken."
        self._sym_repr = sym_repr
        self._lin_repr = lin_repr
        self._pure_lin_repr = pure_lin_repr
        self._factor = _cs1
        self._freeze()

    def __rmul__(self, other):
        """"""
        if other.__class__ is _cs1.__class__:
            if self._factor == _cs1:
                self._factor = other
                return self
            else:
                raise NotImplementedError()
        else:
            pass
