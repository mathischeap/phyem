# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from importlib import import_module


class MseHttFormVisualizeQuick(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._freeze()

    def __call__(self, ddf=1, **kwargs):
        """"""
        space = self._f.space
        m = space.m
        n = space.n
        indicator = space.indicator
        path = self.__repr__().split('main.')[0][1:] + f"{indicator}"
        module = import_module(path)
        title = "$t = %.5f$" % self._t
        if indicator == 'Lambda':
            k = space.abstract.k
            if hasattr(module, f'quick_visualizer_m{m}n{n}k{k}'):
                getattr(module, f'quick_visualizer_m{m}n{n}k{k}')(self._f, self._t, ddf=ddf, title=title, **kwargs)
            else:
                raise NotImplementedError(f"No quick visualizing scheme found for Lambda-form-m{m}n{n}k{k}.")
        else:
            raise NotImplementedError()
