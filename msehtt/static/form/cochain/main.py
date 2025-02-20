# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import _setting
from msehtt.static.form.cochain.instant import MseHttTimeInstantCochain
from msehtt.static.form.cochain.vector.static import MseHttStaticCochainVector
from msehtt.static.form.cochain.vector.dynamic import MseHttDynamicCochainVector


class MseHttCochain(Frozen):
    r""""""

    def __init__(self, f):
        """"""
        self._f = f
        if f._is_base():
            self._newest_t = None
            self._tcd = dict()  # time-cochain-dict
            self._gm = None
        else:
            pass
        self._freeze()

    @staticmethod
    def _parse_t(t):
        """To make time safer."""
        assert t is not None, f"time is None!"
        assert isinstance(t, (int, float)), f"time must be int or float."
        return round(t, 8)  # to make it safer.

    @property
    def newest(self):
        """the largest (not the last) time instant added to the cochain."""
        rf = self._f

        if rf._is_base():
            return self._newest_t
        else:
            return rf._base.cochain._newest_t

    def _set(self, t, cochain):
        """add to cochain at `t` to be cochain."""
        rf = self._f

        if rf._is_base():
            # -- auto_cleaning: memory saving ------------------------------------------------
            auto_cleaning = _setting['auto_cleaning']
            if auto_cleaning is False:
                pass
            else:
                if auto_cleaning is True:
                    left_cochain_amount = 3
                elif isinstance(auto_cleaning, (int, float)):
                    left_cochain_amount = auto_cleaning
                else:
                    raise Exception()
                left_cochain_amount = int(left_cochain_amount)
                assert left_cochain_amount >= 2, f"auto_cleaning must left more than one cochains."

                if len(self) > 2 * left_cochain_amount:
                    self.clean(- left_cochain_amount)
                else:
                    pass

                assert len(self) <= 2 * left_cochain_amount, f'must be!'
            # =================================================================================
            t = self._parse_t(t)
            _cochain_at_time = MseHttTimeInstantCochain(self._f, t)
            _cochain_at_time._receive(cochain)
            self._tcd[t] = _cochain_at_time
            if self._newest_t is None:
                self._newest_t = t
            else:
                self._newest_t = max([self._newest_t, t])  # the largest, not the last t set to cochain.

        else:
            rf._base.cochain._set(t, cochain)

    def clean(self, what=None):
        """Clean instances for particular time instants in cochain."""
        rf = self._f

        if rf._is_base():
            new_tcd = {}
            if what is None:  # clear all t except the newest t
                newest_t = self._newest_t
                if newest_t is None:
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
                        keys = keys[what:]
                        for key in keys:
                            new_tcd[key] = self._tcd[key]
                else:
                    raise NotImplementedError(f"cannot clean {what}! Use a negative integer.")

            elif what == 'all':
                new_tcd = {}

            else:
                raise NotImplementedError(f"cannot clean {what}!")

            self._tcd = new_tcd

            if len(new_tcd) == 0:
                self._newest_t = None
            else:
                self._newest_t = max(list(self._tcd.keys()))

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

    def __iter__(self):
        """iteration over all time instants in tcd."""
        if self._f._is_base():
            for t in self._tcd:
                yield t
        else:
            for t in self._f._base.cochain._tcd:
                yield t

    @property
    def num_global_dofs(self):
        r"""Return the number of dofs across all elements in all ranks."""
        return self.gathering_matrix.num_global_dofs

    @property
    def gathering_matrix(self):
        """"""
        rf = self._f
        if rf._is_base():
            if self._gm is None:
                gm = self._f.space.gathering_matrix(self._f.degree)
                assert len(gm) == len(self._f.tgm.elements), \
                    (f"len(gm) = {len(gm)}, len(self._f.tgm.elements)={len(self._f.tgm.elements)}. "
                     f"They are different! So, wrong! Knowing that "
                     f"even if a form has no business with some great elements, the indices of those"
                     f"great elements still should be in the gm with the values to be None or an empty array.")
                self._gm = gm
                assert self._gm.num_rank_elements == len(self._f.tgm.elements), f'Must be!'
                for i in self._f.tgm.elements:
                    assert i in self._gm, f"must be!"
            return self._gm
        else:
            return rf._base.cochain.gathering_matrix

    def static_vec(self, t):
        """"""
        assert isinstance(t, (int, float)), f"t={t} is wrong."
        if t in self:
            return MseHttStaticCochainVector(self._f, t, self[t].___cochain_caller___, self.gathering_matrix)
        else:
            # This one is usually used to receive a cochain late on. Thus, the data is None (empty)
            return MseHttStaticCochainVector(self._f, t, None, self.gathering_matrix)

    @property
    def dynamic_vec(self):
        """"""
        return MseHttDynamicCochainVector(self._f, self._callable_cochain)

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
