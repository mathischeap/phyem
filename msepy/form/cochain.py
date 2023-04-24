# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

from tools.frozen import Frozen


class MsePyRootFormCochain(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        if rf._is_base():
            self._tcd = dict()  # time-cochain-dict
        else:
            pass
        self._freeze()

    @staticmethod
    def _parse_t(t):
        """To make time safer."""
        return round(t, 8)  # to make it safer.

    def _set(self, t, cochain):
        """add to time-cochain-dict."""
        rf = self._f
        if rf._is_base():
            t = self._parse_t(t)

            _cochain_at_time = _CochainAtOneTime(self._f, t)

            _cochain_at_time._receive(cochain)

            self._tcd[t] = _cochain_at_time
        else:
            rf._base.cochain._set(t, cochain)

    def __getitem__(self, t):
        """Return the cochain at time `t`."""
        rf = self._f
        if rf._is_base():
            t = self._parse_t(t)
            assert t in self._tcd, f"t={t} is not in cochain."
            return self._tcd[t]
        else:
            return rf._base.cochain[t]


class _CochainAtOneTime(Frozen):
    """"""

    def __init__(self, rf, t):
        """"""
        assert rf._is_base, f"rf must be a base root-form."
        self._f = rf
        self._t = t
        self._local_cochain = None
        self._freeze()

    def __repr__(self):
        """"""
        rf_repr = self._f.__repr__()
        my_repr = rf"<Cochain at time={self._t} of "
        super_repr = super().__repr__().split(' object')[1]
        return my_repr + rf_repr + super_repr

    def _receive(self, cochain):
        """"""
        # TODO parse cochain to make it `local` type.
        self._local_cochain = cochain

    @property
    def local(self):
        """"""
        return self._local_cochain

    @local.setter
    def local(self, local_cochain):
        """"""
        self._local_cochain = local_cochain
