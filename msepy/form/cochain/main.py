# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

from tools.frozen import Frozen
from msepy.form.cochain.time_instant import _CochainAtOneTime


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

    def __contains__(self, t):
        """if rf has cochain at time`t`?"""
        t = self._parse_t(t)
        return t in self._tcd

    @property
    def gathering_matrix(self):
        return None

    @property
    def local_numbering(self):
        return None
