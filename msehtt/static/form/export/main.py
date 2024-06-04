# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.form.export.rws import MseHtt_Static_Form_Export_RWS


class MseHtt_Static_Form_Export(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._t = t
        self._f = f
        self._rws = None
        self._freeze()

    @property
    def rws(self):
        if self._rws is None:
            self._rws = MseHtt_Static_Form_Export_RWS(self._f, self._t)
        return self._rws
