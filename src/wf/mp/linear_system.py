# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')

from src.wf.mp.linear_system_bc import MatrixProxyLinearSystemBoundaryConditions


class MatrixProxyLinearSystem(Frozen):
    """"""

    def __init__(self, mp, ls, mp_bc):
        """"""
        self._mp = mp
        self._ls = ls
        self._parse_bc(mp_bc)
        self._freeze()

    def _parse_bc(self, mp_bc):
        """"""
        if mp_bc is None or len(mp_bc) == 0:
            self._bc = None
        else:
            self._bc = MatrixProxyLinearSystemBoundaryConditions(self, mp_bc)

    @property
    def bc(self):
        """boundary condition."""
        return self._bc

    def pr(self, figsize=(12, 6)):
        """"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass
        seek_text = self._mp._mp_seek_text()
        linear_system_text = self._ls._pr_text()
        symbolic = r"$" + linear_system_text + r"$"
        if self.bc is None or len(self.bc) == 0:
            bc_text = ''
        else:
            bc_text = self.bc._bc_text()
        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, seek_text + symbolic + bc_text, ha='left', va='center', size=15)

        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='matrixProxyLinearSystem')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])
        return fig

    def _pr_temporal_advancing(self, *args, **kwargs):
        """"""
        return self._mp._pr_temporal_advancing(*args, **kwargs)
