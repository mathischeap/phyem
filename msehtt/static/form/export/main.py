# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHtt_Static_Form_Export(Frozen):
    r""""""

    def __init__(self, f, t):
        r""""""
        self._t = t
        self._f = f
        self._rws = None
        self._freeze()

    def rws(self, filename, ddf=1):
        r""""""
        dds = self._f.numeric.rws(self._t, ddf=ddf)
        if dds is None:
            pass
        else:
            dds.saveto(filename)

    def vtk(self, filename, ddf=1):
        r""""""
        from tools.vtk_.msehtt_form_static_copy import ___ph_vtk_msehtt_static_copy___
        ___ph_vtk_msehtt_static_copy___(filename, self._f[self._t], ddf=ddf)
