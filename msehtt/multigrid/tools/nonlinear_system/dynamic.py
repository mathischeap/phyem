# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class MseHtt_MultiGrid_DynamicNonLinearSystem(Frozen):
    r""""""

    def __init__(self, wf_mp_nls, base):
        """"""
        from phyem.src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem
        assert wf_mp_nls.__class__ is MatrixProxyNoneLinearSystem, \
            f"I need a {MatrixProxyNoneLinearSystem}!"
        self._mp_nls = wf_mp_nls
        self._base = base
        self._freeze()
