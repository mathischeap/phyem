# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen


class Interval(Frozen):
    r"""For example,

    i0 = Interval((0, 1))  # open interval (0, 1)
    i2 = Interval([1, 3])  # closed interval [1, 3]

    """
    def __init__(self, interval):
        r""""""
        if isinstance(interval, tuple) and len(interval) == 2:  # open interval

            i0, i1 = interval
            assert isinstance(i0, (int, float)) and isinstance(i1, (int, float)) and i0 < i1, \
                f"An open interval = {interval} is illegal."

            self._type = 'open'
            self._lb, self._ub = interval

        elif isinstance(interval, list) and len(interval) == 2:  # closed interval

            i0, i1 = interval
            assert isinstance(i0, (int, float)) and isinstance(i1, (int, float)) and i0 < i1, \
                f"An open interval = {interval} is illegal."

            self._type = 'closed'
            self._lb, self._ub = interval

        else:
            raise Exception()

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('at')[1]
        if self._type == 'open':
            STR = f"({self._lb}, {self._ub})"
        elif self._type == 'closed':
            STR = f"({self._lb}, {self._ub})"
        else:
            raise Exception()
        return rf"{self._type} interval {STR} at " + super_repr

    def linspace(self, num_of_nodes):
        r""""""
        return np.linspace(self._lb, self._ub, num_of_nodes)

    def __contains__(self, item):
        r""""""
        if isinstance(item, (int, float)) or (isinstance(item, np.ndarray) and item.ndim == 0):
            if self._type == 'open':
                return self._lb < item < self._ub
            elif self._type == 'closed':
                return self._lb <= item <= self._ub
            else:
                raise Exception

        elif isinstance(item, np.ndarray):

            if item.ndim == 1:
                if self._type == 'open':
                    A = item > self._lb
                    B = item < self._ub
                    C = np.logical_and(A, B)
                    return all(C)

                elif self._type == 'closed':
                    A = item >= self._lb
                    B = item <= self._ub
                    C = np.logical_and(A, B)
                    return all(C)

                else:
                    raise Exception

            else:
                raise NotImplementedError(item, item.__class__, item.ndim)

        else:
            raise NotImplementedError(f"Cannot check whether {item} is in an interval.")
