# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:41 PM on 5/17/2023
"""
from scipy.sparse import linalg as spspalinalg
from tools.frozen import Frozen
from time import time
from tools.miscellaneous.timer import MyTimer


class MsePyStaticLinearSystemAssembledSolve(Frozen):
    """"""
    def __init__(self, als):
        """"""
        self._als = als
        self._A = als.A._M
        self._b = als.b._v
        self._system_info = f'<Linear system> <shape: {self._A.shape}> '
        self._last_solver_message = ''
        self._freeze()

    def __call__(self, update_x=True):
        """direct solver."""
        t_start = time()
        x = self._direct()
        t_cost = time() - t_start
        if update_x:
            self._als._static.x.update(x)
        else:
            pass
        t_cost = MyTimer.seconds2dhms(t_cost)
        self._last_solver_message = \
            self._system_info + f"<direct solver costs: {t_cost}> "
        return x

    @property
    def message(self):
        """return the _last_solver_message."""
        return self._last_solver_message

    def _direct(self):
        x = spspalinalg.spsolve(self._A, self._b)
        return x

    def gmres(self, update_x=True, **kwargs):
        """"""
        t_start = time()
        x, info = self._gmres(**kwargs)
        t_cost = time() - t_start
        if update_x:
            self._als._static.x.update(x)
        else:
            pass
        t_cost = MyTimer.seconds2dhms(t_cost)
        info_kwargs_exclusive = ['x0', 'M', 'callback']
        info_kwargs = {}
        for key in kwargs:
            if key not in info_kwargs_exclusive:
                info_kwargs[key] = kwargs[key]
        self._last_solver_message = \
            self._system_info + f"<gmres costs: {t_cost}> " \
                                f"<info: {info}> <inputs: {info_kwargs}>"
        return x

    def _gmres(self, **kwargs):
        """"""
        x, info = spspalinalg.gmres(
            self._A, self._b, **kwargs
        )
        return x, info
