r"""

"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.main import MseHttSpace


class MseHtt_MultiGrid_Space(Frozen):
    r""""""
    def __init__(self, abstract_space):
        """"""
        self._abstract = abstract_space
        self._tpm = None
        self._levels = dict()
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    @property
    def tpm(self):
        if self._tpm is None:
            raise Exception(f"first set tpm: I am {self}!")
        return self._tpm

    @property
    def tgm(self):
        return self.tpm.tgm

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        return '<MseHtt-MultiGrid-Space: ' + ab_space_repr + super().__repr__().split('object')[1]

    def get_level(self, lvl=None):
        r""""""
        if lvl is None:
            lvl = self.tgm.max_level
        else:
            pass
        if lvl in self._levels:
            return self._levels[lvl]
        else:
            lvl_tpm = self.tpm.get_level(lvl)
            lvl_space = MseHttSpace(self.abstract)
            lvl_space._tpm = lvl_tpm
            self._levels[lvl] = lvl_space
            return lvl_space
