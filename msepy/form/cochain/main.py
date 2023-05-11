# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

from tools.frozen import Frozen
from msepy.form.cochain.time_instant import _CochainAtOneTime
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector


class MsePyLockCochainError(Exception):
    """"""


class MsePyRootFormCochain(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        if rf._is_base():
            self._tcd = dict()  # time-cochain-dict
        else:
            pass
        self._locker = False  # if locker, cannot set new cochain.
        self._freeze()

    @staticmethod
    def _parse_t(t):
        """To make time safer."""
        return round(t, 8)  # to make it safer.

    def _set(self, t, cochain):
        """add to cochain at `t` to be cochain."""
        if self._locker:  # cochain locked, cannot set new cochain.
            raise MsePyLockCochainError(f"Cochain of {self._f} is locked!")

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
            assert t in self._tcd, f"t={t} is not in cochain of form {self._f}."
            return self._tcd[t]
        else:
            return rf._base.cochain[t]

    def __contains__(self, t):
        """if rf has cochain at time`t`?"""
        t = self._parse_t(t)
        rf = self._f
        if rf._is_base():
            return t in self._tcd
        else:
            return t in rf._base.cochain._tcd

    @property
    def gathering_matrix(self):
        """"""
        return self._f.space.gathering_matrix(self._f.degree)

    @property
    def local_numbering(self):
        return self._f.space.local_numbering(self._f.degree)

    def static_vec(self, t):
        """"""
        if t in self:
            return MsePyRootFormStaticCochainVector(self._f, t, self[t], self.gathering_matrix)
        else:
            # this one is usually used to receive a cochain.
            return MsePyRootFormStaticCochainVector(self._f, t, None, self.gathering_matrix)
