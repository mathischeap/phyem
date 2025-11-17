# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.main import MseHttSpace


class MseHtt_Adaptive_TopSpace(Frozen):
    """"""

    def __init__(self, abstract_space, ___MAX_GENERATIONS____):
        """"""
        self._abstract_space = abstract_space
        self.___MAX_GENERATIONS____ = ___MAX_GENERATIONS____
        self._generations_ = list()
        self._total_generation_ = 0
        self.___renew_stamp___ = ''
        self._freeze()

    @property
    def ith_generation(self):
        r"""1st generation is the one initialized. And after each renew, this property +=1."""
        return self._total_generation_

    @property
    def abstract(self):
        return self._abstract_space

    @property
    def current(self):
        return self._generations_[-1]

    def ___renew___(self):
        r""""""
        if len(self._generations_) >= self.___MAX_GENERATIONS____:
            self._generations_ = self._generations_[(-self.___MAX_GENERATIONS____+1):]
        else:
            pass
        new_space = MseHttSpace(self._abstract_space)
        self._generations_.append(new_space)
        self._total_generation_ += 1
