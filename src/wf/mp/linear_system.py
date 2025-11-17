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

from phyem.tools.frozen import Frozen
from phyem.src.wf.mp.linear_system_bc import MatrixProxyLinearSystemBoundaryConditions
from phyem.tools.miscellaneous.latex_bmatrix_to_array import bmatrix_to_array


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
        """"""
        return self._bc

    def pr(self, figsize=(12, 6)):
        """"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return None
        else:
            pass
        seek_text = self._mp._mp_seek_text()
        linear_system_text = self._ls._pr_text()
        symbolic = r"$" + linear_system_text + r"$"

        # ----- for big matrix, bmatrix environment does not work, use array instead -----------------------
        num_unknowns = len(self._mp._wf.unknowns)
        if num_unknowns > 5:
            # matrix_begin_latex = r"\left[\begin{array}{" + r"c" * num_unknowns + r"}"
            # matrix_end_latex = r"\end{array}\right]"
            # if r"\begin{bmatrix}" in symbolic:
            #     symbolic = symbolic.replace(r"\begin{bmatrix}", matrix_begin_latex)
            # if r"\end{bmatrix}" in symbolic:
            #     symbolic = symbolic.replace(r"\end{bmatrix}", matrix_end_latex)

            symbolic = bmatrix_to_array(symbolic, num_unknowns)

        else:
            pass
        # ===================================================================================================

        if self.bc is None or len(self.bc) == 0:
            bc_text = ''
        else:
            bc_text = self.bc._bc_text()
        fig = plt.figure(figsize=figsize)
        plt.axis((0, 1, 0, 1))
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
