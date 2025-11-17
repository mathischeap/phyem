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
from phyem.tools.miscellaneous.latex_bmatrix_to_array import bmatrix_to_array


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
            return None
        else:
            pass

        seek_text = self._mp._mp_seek_text()

        symbolic = r"$" + self._mp._pr_text() + r"$"

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

        if self._mp_ls.bc is None or len(self._mp_ls.bc) == 0:
            bc_text = ''
        else:
            bc_text = self._mp_ls._bc._bc_text()
        fig = plt.figure(figsize=figsize)
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.5, seek_text + symbolic + bc_text, ha='left', va='center', size=15)
        # plt.text(0.05, 0.5, symbolic, ha='left', va='center', size=15)
        # print(symbolic)

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
