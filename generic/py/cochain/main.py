# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from generic.py.cochain.time_instant_cochain import ParticularCochainAtTimeInstant
from generic.py.vector.localize.static import Localize_Static_Vector_Cochain
from generic.py.vector.localize.dynamic import Localize_Dynamic_Vector_Cochain


class CochainLockedError(Exception):
    """"""


class Cochain(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        if f._is_base():
            self._newest_t = None
            self._tcd = dict()  # time-cochain-dict
        else:
            pass
        self._locker = False  # if locker, cannot set new cochain.
        self._freeze()

    @staticmethod
    def _parse_t(t):
        """To make time safer."""
        assert t is not None, f"time is None!"
        return round(t, 8)  # to make it safer.

    def _set(self, t, cochain):
        """add to cochain at `t` to be cochain."""
        if self._locker:  # cochain locked, cannot set new cochain.
            raise CochainLockedError(f"Cochain of {self._f} is locked!")
        else:
            pass

        rf = self._f

        if rf._is_base():
            t = self._parse_t(t)

            _cochain_at_time = ParticularCochainAtTimeInstant(self._f, t)

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
            if what is None:  # clear all `t` except the `newest t`
                newest_t = self._newest_t
                if new_tcd is None:
                    pass
                else:
                    new_tcd[newest_t] = self._tcd[newest_t]
            else:
                raise NotImplementedError()

            self._tcd = new_tcd

        else:
            rf._base.cochain.clean(what=what)

    def __getitem__(self, t):
        """Return the cochain at time `t`."""
        if t is None:
            t = self.newest
        else:
            pass
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

    def __len__(self):
        """How many valid time instants save in self._tcd."""
        if self._f._is_base():
            return len(self._tcd)
        else:
            return len(self._f._base.cochain._tcd)

    @property
    def gathering_matrix(self):
        """"""
        return self._f.space.gathering_matrix(self._f.degree)

    @property
    def local_numbering(self):
        """"""
        return self._f.space.local_numbering(self._f.degree)

    def static_vec(self, t):
        """"""
        assert isinstance(t, (int, float)), f"t={t} is wrong."
        if t in self:
            return Localize_Static_Vector_Cochain(self._f, t, self[t].local, self.gathering_matrix)
        else:
            # this one is usually used to receive a cochain.
            return Localize_Static_Vector_Cochain(self._f, t, None, self.gathering_matrix)
            # the data is None (empty)

    @property
    def dynamic_vec(self):
        """"""
        return Localize_Dynamic_Vector_Cochain(self._dynamic_cochain_caller)

    def _dynamic_cochain_caller(self, *args, **kwargs):
        """"""
        t = self._ati_time_caller(*args, **kwargs)
        return self.static_vec(t)

    def _ati_time_caller(self, *args, **kwargs):
        """"""
        if self._f._is_base():
            t = args[0]
            assert isinstance(t, (int, float)), f"for general root-form, I receive a real number!"
        else:
            ati = self._f._pAti_form['ati']
            t = ati(**kwargs)()
        return t
