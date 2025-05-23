# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.mesh.partial.main import MseHttMeshPartial
from msehtt.static.mesh.great.main import MseHttGreatMesh


class MseHtt_Adaptive_TopMesh(Frozen):
    """"""

    def __init__(self, abstract_mesh, ___MAX_GENERATIONS____):
        """"""
        self._abstract_mesh = abstract_mesh
        self._TGM_ = None
        self._including_ = None
        self.___MAX_GENERATIONS____ = ___MAX_GENERATIONS____
        self._generations_ = list()
        self._total_generation_ = 0
        self.___renew_stamp___ = ''
        self._freeze()

    @property
    def ith_generation(self):
        r"""1st generation is the one initialized. And after each renew, this property +=1."""
        return self._total_generation_

    def _config(self, tgm, including):
        r""""""
        assert isinstance(tgm, MseHttGreatMesh)
        self._TGM_ = tgm  # this is the base tgm (the tgm of initialization) all refinement will be based on it.
        if isinstance(including, self.__class__):
            self._including_ = ('REAL-TIME', including, 'current')
        elif isinstance(including, dict):
            INCLUDE = {}
            for key in including:
                value = including[key]
                if isinstance(value, self.__class__):
                    INCLUDE[key] = ('REAL-TIME', value, 'current')
                else:
                    INCLUDE[key] = value
            self._including_ = INCLUDE
        else:
            self._including_ = including

    @property
    def abstract(self):
        return self._abstract_mesh

    @property
    def current(self):
        return self._generations_[-1]

    def ___renew___(self, new_tgm):
        r"""Must use this method to add a new generation to the list."""
        if len(self._generations_) >= self.___MAX_GENERATIONS____:
            self._generations_ = self._generations_[(-self.___MAX_GENERATIONS____+1):]
        else:
            pass
        new_mesh = MseHttMeshPartial(self._abstract_mesh)

        if isinstance(self._including_, tuple) and self._including_[0] == 'REAL-TIME':
            including = getattr(self._including_[1], self._including_[2])
        elif isinstance(self._including_, dict):
            including = {}
            for key in self._including_:
                value = self._including_[key]
                if isinstance(value, tuple) and value[0] == 'REAL-TIME':
                    including[key] = getattr(value[1], value[2])
                else:
                    including[key] = value
        else:
            including = self._including_

        new_mesh._config(new_tgm, including)
        self._generations_.append(new_mesh)
        self._total_generation_ += 1

    @property
    def visualize(self):
        r""""""
        return self.current.visualize
