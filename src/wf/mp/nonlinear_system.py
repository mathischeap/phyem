# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')
from tools.frozen import Frozen


class MatrixProxyNoneLinearSystem(Frozen):
    """"""

    def __init__(self, mp, mp_ls, nls):
        """"""
        self._mp = mp   # this nonlinear system is from ``mp``
        self._mp_ls = mp_ls   # the ``matrix proxy`` wrapper of ``ls``, mainly to parse the bc, i.e., ``mp_ls._bc``.
        self._nls = nls  # the nonlinear system I am wrapping over.
        self._freeze()

    def pr(self, figsize=(12, 6)):
        """"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass

        seek_text = self._mp._mp_seek_text()
        symbolic = r"$" + self._mp._pr_text() + r"$"
        if self._mp_ls.bc is None or len(self._mp_ls.bc) == 0:
            bc_text = ''
        else:
            bc_text = self._mp_ls._bc._bc_text()
        fig = plt.figure(figsize=figsize)
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.5, seek_text + symbolic + bc_text, ha='left', va='center', size=15)

        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='matrixProxyNonLinearSystem')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])
        return fig

    def _pr_temporal_advancing(self, *args, **kwargs):
        """"""
        return self._mp._pr_temporal_advancing(*args, **kwargs)
