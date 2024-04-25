# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen


class MseHttDynamicNonLinearSystem(Frozen):
    """"""

    def __init__(self, wf_mp_nls, base):
        """"""
        from src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem
        assert wf_mp_nls.__class__ is MatrixProxyNoneLinearSystem, f"I need a {MatrixProxyNoneLinearSystem}!"
        self._mp_nls = wf_mp_nls
        self._nls = wf_mp_nls._nls
        self._mp = wf_mp_nls._mp
        self._freeze()
