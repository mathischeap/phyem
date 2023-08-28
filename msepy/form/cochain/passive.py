# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.form.cochain.time_instant import _CochainAtOneTime


class MsePyRootFormCochainPassive(Frozen):
    """We can only retrieve a cochain at a particular time from a passive cochain. """

    def __init__(self, rf, reference_form_cochain):
        """"""
        self._f = rf
        if isinstance(reference_form_cochain, (list, tuple)):
            self._crc = _ChainReferenceCochains(*reference_form_cochain)
        else:
            self._crc = _ChainReferenceCochains(reference_form_cochain)
        self._realtime_local_cochain_caller = None
        self._freeze()

    @property
    def gathering_matrix(self):
        """"""
        return self._f.space.gathering_matrix(self._f.degree)

    def _parse_t(self, t):
        """To make time safer."""
        return self._crc._parse_t(t)

    @property
    def newest(self):
        return self._crc.newest

    def __getitem__(self, t):
        """Return the cochain at time `t`."""
        if t is None:
            t = self.newest
        else:
            pass
        t = self._parse_t(t)
        cochain = _CochainAtOneTime(self._f, t)
        cochain._receive(self._realtime_local_cochain_caller)
        return cochain


class _ChainReferenceCochains(Frozen):
    """"""
    def __init__(self, *cochains):
        """"""
        assert len(cochains) > 0, f"I receive no cochain object."
        self._cochains = cochains

    def _parse_t(self, t):
        """"""
        assert t is not None, "t is None."
        ts = list()
        for ch in self._cochains:
            ts.append(ch._parse_t(t))
        assert all([_ == ts[0] for _ in ts]), f"get wrong times, {ts}"
        return ts[0]

    @property
    def newest(self):
        new_list = list()
        for ch in self._cochains:
            new_list.append(ch.newest)
        if None in new_list:
            return None
        else:
            if all([_ == new_list[0] for _ in new_list]):
                return new_list[0]
            else:
                return None
