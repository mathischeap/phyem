# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msehy.tools.time_instant import _IrregularCochainAtOneTime
from msehy.tools.vector.static.local import IrregularStaticCochainVector
from msehy.tools.vector.dynamic import IrregularDynamicCochainVector


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

    def visualize_difference(self, t_g_1, t_g_2, density=100, magnitude=True):
        """
        reconstruction of t_g_1 minus reconstruction of t_g_2.

        t_g_1 - t_g_2.

        Parameters
        ----------
        t_g_1
        t_g_2
        density
        magnitude

        Returns
        -------

        """
        t1, g1 = t_g_1
        t2, g2 = t_g_2
        t1 = self._parse_t(t1)
        t2 = self._parse_t(t2)
        g1 = self._f._pg(g1)
        g2 = self._f._pg(g2)
        assert (t1, g1) != (t2, g2), f"{(t1, g1)} == {(t2, g2)}, no difference!"
        r = np.linspace(0, 1, density)
        s = np.linspace(0, 1, density)
        r, s = np.meshgrid(r, s, indexing='ij')
        dds1 = self._f[(t1, g1)].numeric.region_wise_reconstruct(r, s)
        dds2 = self._f[(t2, g2)].numeric.region_wise_reconstruct(r, s)
        dds = dds1 - dds2   # 1 - 2
        dds.visualize(magnitude=magnitude)

    @staticmethod
    def _parse_t(t):
        """To make time safer."""
        assert t is not None, f"time is None!"
        return round(t, 8)  # to make it safer.

    def _set(self, t, generation, cochain):
        """add to cochain at `t` to be cochain."""
        if self._locker:  # cochain locked, cannot set new cochain.
            raise MseHyPy2CochainError(f"Cochain of {self._f} is locked!")

        self.clean()  # whenever set a new cochain, we clean the cochain that are out-dated.

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

    def gathering_matrix(self, g):
        """"""
        g = self._f._pg(g)  # do it also here for safety.
        return self._f.space.gathering_matrix(self._f.degree, g)

    @property
    def local_numbering(self):
        """"""
        return self._f.space.local_numbering(self._f.degree)

    def static_vec(self, t, g):
        """"""
        assert isinstance(t, (int, float)), f"t={t} is wrong."
        if (t, g) in self:
            return IrregularStaticCochainVector(
                self._f, t, g, self[(t, g)].local, self.gathering_matrix(g)
            )
        else:
            # this one is usually used to receive a cochain.
            return IrregularStaticCochainVector(
                self._f, t, g, None, self.gathering_matrix(g)
            )
            # the data is None (empty)

    @property
    def dynamic_vec(self):
        """"""
        return IrregularDynamicCochainVector(self._f, self._callable_cochain)

    def _callable_cochain(self, *args, g=None, **kwargs):
        """"""
        t = self._ati_time_caller(*args, g=g, **kwargs)
        g = self._generation_caller(*args, g=g, **kwargs)
        return self.static_vec(t, g)

    def _ati_time_caller(self, *args, g=None, **kwargs):
        """"""
        _ = g
        if self._f._is_base():
            t = args[0]
            assert isinstance(t, (int, float)), f"for general root-form, I receive a real number!"
        else:
            ati = self._f._pAti_form['ati']
            t = ati(**kwargs)()
        return t

    def _generation_caller(self, *args, g=None, **kwargs):
        """"""
        _ = args, kwargs
        return self._f._pg(g)
