# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from scipy.sparse import linalg as spspalinalg


class MseHtt_Local_LinearSystem_Solve(Frozen):
    r"""Use this property of a msehtt-local-linear-system, we can try to solve the local
    systems element-wise.

    Thus, we do not assemble the local systems. This sometimes is useful, for example, when the problem is
    set up locally at the first please. In this case, assembling it and solving it will cause error.

    """

    def __init__(self, lls):
        r"""

        Parameters
        ----------
        lls :
            The Local Linear System.
        """
        self._lls = lls
        self._freeze()

    def __call__(self, scheme, **kwargs):
        """"""
        if scheme == 'spsolve':
            # use scipy sparse linalg sparse solve
            x = dict()
            for i in self._lls:
                A = self._lls.A[i]
                b = self._lls.b[i]
                x[i] = spspalinalg.spsolve(A, b)
            message = 'spsolve'
            info = 0
            return x, message, info
        else:
            raise NotImplementedError(scheme)
