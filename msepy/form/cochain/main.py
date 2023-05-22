# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

from tools.frozen import Frozen
from msepy.form.cochain.time_instant import _CochainAtOneTime
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector
from msepy.form.cochain.vector.dynamic import MsePyRootFormDynamicCochainVector


class MsePyLockCochainError(Exception):
    """"""


class MsePyRootFormCochain(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        if rf._is_base():
            self._newest_t = None
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
            self._newest_t = t

        else:
            rf._base.cochain._set(t, cochain)

    @property
    def newest(self):
        """newest time instant added to the cochain."""
        rf = self._f

        if rf._is_base():
            return self._newest_t
        else:
            return rf._base.cochain._newest_t

    def clean(self, what=None):
        """Clean instances for particular time instants in cochain."""
        rf = self._f

        if rf._is_base():
            new_tcd = {}
            if what is None:  # clear all t except the newest t
                newest_t = self._newest_t
                if new_tcd is None:
                    pass
                else:
                    new_tcd[newest_t] = self._tcd[newest_t]
            elif what == 'all':
                pass
            else:
                raise NotImplementedError(f"cannot clean {what}!")
        else:
            rf._base.cochain.clean(what=what)

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
        assert isinstance(t, (int, float)), f"t={t} is wrong."
        if t in self:
            return MsePyRootFormStaticCochainVector(self._f, t, self[t].local, self.gathering_matrix)
        else:
            # this one is usually used to receive a cochain.
            return MsePyRootFormStaticCochainVector(self._f, t, None, self.gathering_matrix)

    @property
    def dynamic_vec(self):
        """"""
        return MsePyRootFormDynamicCochainVector(self._f, self._callable_cochain)

    def _callable_cochain(self, *args, **kwargs):
        """"""
        if self._f._is_base():
            t = args[0]
            assert isinstance(t, (int, float)), f"for general root-form, I receive a real number!"
            return self.static_vec(t)
        else:
            ati = self._f._pAti_form['ati']
            t = ati(**kwargs)()
            return self.static_vec(t)
