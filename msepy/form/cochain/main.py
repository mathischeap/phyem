# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen

from src.config import _setting

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
        assert t is not None, f"time is None!"
        return round(t, 8)  # to make it safer.

    def _set(self, t, cochain):
        """add to cochain at `t` to be cochain."""
        if self._locker:  # cochain locked, cannot set new cochain.
            raise MsePyLockCochainError(f"Cochain of {self._f} is locked!")

        # -- auto_cleaning: memory saving ------------------------------------------------
        auto_cleaning = _setting['auto_cleaning']
        if auto_cleaning is False:
            pass
        else:
            if auto_cleaning is True:
                left_cochain_amount = 10
            elif isinstance(auto_cleaning, int):
                assert auto_cleaning >= 2, f"auto_cleaning wrong for cochain."
                left_cochain_amount = auto_cleaning
            else:
                raise Exception

            if len(self) > 2 * left_cochain_amount:
                self.clean(- left_cochain_amount)
            else:
                pass
        # =================================================================================

        rf = self._f

        if rf._is_base():
            t = self._parse_t(t)

            _cochain_at_time = _CochainAtOneTime(self._f, t)

            _cochain_at_time._receive(cochain)

            self._tcd[t] = _cochain_at_time

            if self._newest_t is None:
                self._newest_t = t
            else:
                self._newest_t = max([self._newest_t, t])  # the largest, not the last t set to cochain.

        else:
            rf._base.cochain._set(t, cochain)

    @property
    def newest(self):
        """the largest (not the last) time instant added to the cochain."""
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

            elif isinstance(what, (int, float)):

                what = int(what)
                if what < 0:  # clean all cochain except the largest `what` ones.

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

            elif what == 'all':
                raise NotImplementedError(f"cannot clean {what}!")

            else:
                raise NotImplementedError(f"cannot clean {what}!")

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

    def __call__(self, t):
        """use linear interpolation to retrieve cochain at any intermediate time."""
        if t is None:
            t = self.newest
        else:
            pass
        rf = self._f
        if rf._is_base():
            t = self._parse_t(t)
            if t in self._tcd:
                return self._tcd[t]
            else:
                amount_cochains = len(self._tcd)
                if amount_cochains < 2:
                    raise Exception(f"not enough cochain instances for interpolation. "
                                    f"Need at least two valid cochain instances. Now "
                                    f"I have {amount_cochains}.")
                else:
                    cochain_times = list(self._tcd.keys())
                    cochain_times.sort()
                    lower_bound = cochain_times[0]
                    upper_bound = cochain_times[-1]
                    if t < lower_bound:
                        raise Exception(
                            f"t={t} is lower than the lower bound of "
                            f"valid cochain time range [{lower_bound}, {upper_bound}].")
                    elif t > upper_bound:
                        raise Exception(
                            f"t={t} is larger than the upper bound of "
                            f"valid cochain time range [{lower_bound}, {upper_bound}].")
                    else:
                        segment_lb = None
                        segment_ub = None
                        for i, segment_lb in enumerate(cochain_times):
                            if t >= segment_lb:
                                segment_ub = cochain_times[i+1]
                                break
                            else:
                                pass
                        assert segment_lb is not None and segment_ub is not None, \
                            f"something is wrong."
                        cochain_lb = self[segment_lb]
                        cochain_ub = self[segment_ub]
                        ratio = (t - segment_lb) / (segment_ub - segment_lb)
                        cochain_t = cochain_lb + ratio * (cochain_ub - cochain_lb)
                        return cochain_t
        else:
            return rf._base.cochain(t)

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
        return self._f.space.local_numbering(self._f.degree)

    def static_vec(self, t):
        """"""
        assert isinstance(t, (int, float)), f"t={t} is wrong."
        if t in self:
            return MsePyRootFormStaticCochainVector(self._f, t, self[t].local, self.gathering_matrix)
        else:
            # this one is usually used to receive a cochain.
            return MsePyRootFormStaticCochainVector(self._f, t, None, self.gathering_matrix)
            # the data is None (empty)

    @property
    def dynamic_vec(self):
        """"""
        return MsePyRootFormDynamicCochainVector(self._f, self._callable_cochain)

    def _callable_cochain(self, *args, **kwargs):
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
