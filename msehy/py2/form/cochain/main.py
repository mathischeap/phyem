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
            self._newest_g = None
            self._tg_cd = dict()  # time-cochain-dict
        else:
            pass
        self._locker = False  # if locker, cannot set new cochain.
        self._freeze()

    @staticmethod
    def _parse_t(t):
        """To make time safer."""
        assert t is not None, f"time is None!"
        return round(t, 8)  # to make it safer.

    def _set(self, t, generation, cochain):
        """add to cochain at `t` to be cochain."""
        if self._locker:  # cochain locked, cannot set new cochain.
            raise MseHyPy2CochainError(f"Cochain of {self._f} is locked!")

        rf = self._f

        if rf._is_base():

            t = self._parse_t(t)
            generation = self._f._pg(generation)
            _cochain_at_time = _IrregularCochainAtOneTime(self._f, t, generation)
            _cochain_at_time._receive(cochain)

            self._tg_cd[(t, generation)] = _cochain_at_time
            self._newest_t = t
            self._newest_g = generation

        else:
            rf._base.cochain._set(t, cochain, generation=generation)

    @property
    def newest(self):
        """newest time instant and generation added to the cochain."""
        rf = self._f

        if rf._is_base():
            return self._newest_t, self._newest_g
        else:
            return rf._base.cochain._newest_t

    def __getitem__(self, t_g):
        """Return the cochain at time `t` on generation `g`; `t_g = (t, g)`."""
        if t_g is None:
            t_g = self.newest
        else:
            pass
        rf = self._f
        if rf._is_base():
            t, g = t_g
            t = self._parse_t(t)
            g = self._f._pg(g)
            assert (t, g) in self._tg_cd, f"t, g =({t}, {g}) is not in cochain of form {self._f}."
            return self._tg_cd[(t, g)]
        else:
            return rf._base.cochain[t_g]

    def clean(self, what=None):
        """Clean instances for particular time instants in cochain."""
        rf = self._f

        if rf._is_base():
            new_tg_cd = {}
            if what is None:  # clear all instant cochains that live on not cached generations of the mesh.
                # different from the msepy cochain clean method.
                for t_g in self._tg_cd:
                    cochain = self._tg_cd[t_g]
                    if cochain.generation in self._f.mesh.generations:
                        new_tg_cd[t_g] = cochain
                    else:
                        pass

            else:
                raise NotImplementedError(f"cannot clean {what}!")

            self._tg_cd = new_tg_cd

        else:
            rf._base.cochain.clean(what=what)

    def __contains__(self, t_g):
        """if rf has cochain at time`t`?"""
        t, g = t_g
        t = self._parse_t(t)
        g = self._f._pg(g)
        rf = self._f
        if rf._is_base():
            return (t, g) in self._tg_cd
        else:
            return (t, g) in rf._base.cochain._tg_cd

    def __len__(self):
        """How many valid time instants save in self._tg_cd."""
        if self._f._is_base():
            return len(self._tg_cd)
        else:
            return len(self._f._base.cochain._tg_cd)

    def gathering_matrix(self, generation):
        """"""
        generation = self._f._pg(generation)  # do it also here for safety.
        return self._f.space.gathering_matrix(self._f.degree, generation)

    @property
    def local_numbering(self):
        """"""
        return self._f.space.local_numbering(self._f.degree)
