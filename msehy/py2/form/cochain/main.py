# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.form.cochain.time_instant import _IrregularCochainAtOneTime


class MseHyPy2CochainError(Exception):
    """"""


class MseHyPy2Cochain(Frozen):
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
        assert t is not None, f"time is None!"
        return round(t, 8)  # to make it safer.

    def _set(self, t, cochain, generation=None):
        """add to cochain at `t` to be cochain."""
        if self._locker:  # cochain locked, cannot set new cochain.
            raise MseHyPy2CochainError(f"Cochain of {self._f} is locked!")

        rf = self._f

        if rf._is_base():

            t = self._parse_t(t)

            generation = self._f._pg(generation)
            _cochain_at_time = _IrregularCochainAtOneTime(self._f, t, generation)

            _cochain_at_time._receive(cochain)

            self._tcd[t] = _cochain_at_time
            self._newest_t = t

        else:
            rf._base.cochain._set(t, cochain, generation=generation)

    @property
    def newest(self):
        """newest time instant added to the cochain."""
        rf = self._f

        if rf._is_base():
            return self._newest_t
        else:
            return rf._base.cochain._newest_t

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

    def clean(self, what=None):
        """Clean instances for particular time instants in cochain."""
        rf = self._f

        if rf._is_base():
            new_tcd = {}
            if what is None:  # clear all instant cochains that live on old generations of the mesh.
                # different from the msepy cochain clean method.
                for t in self._tcd:
                    cochain = self._tcd[t]
                    if cochain.generation == self._f.mesh.current_generation:
                        new_tcd[t] = cochain
                    else:
                        pass

            elif isinstance(what, (int, float)):

                what = int(what)
                if what < 0:  # clean all cochain except the last newest `what` ones.

                    leave_amount = -what
                    keys = list(self._tcd.keys())
                    keys.sort()

                    if len(keys) <= leave_amount:
                        new_tcd = self._tcd
                    else:
                        keys = keys[-leave_amount:]
                        for key in keys:
                            new_tcd[key] = self._tcd[key]

                else:
                    raise NotImplementedError(f"cannot clean {what}!")

            else:
                raise NotImplementedError(f"cannot clean {what}!")

            self._tcd = new_tcd

        else:
            rf._base.cochain.clean(what=what)

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

    def gathering_matrix(self, generation=None):
        """"""
        return self._f.space.gathering_matrix(self._f.degree, generation)

    @property
    def local_numbering(self):
        """"""
        return self._f.space.local_numbering(self._f.degree)
