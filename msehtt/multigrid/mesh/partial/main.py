r"""

"""
from phyem.src.mesh import Mesh
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.mesh.partial.main import MseHttMeshPartial


class Empty_MultiGrid_CompositionError(Exception):
    """"""


class MseHtt_MultiGrid_MeshPartial(Frozen):
    """"""

    def __init__(self, abstract_mesh):
        """"""
        assert abstract_mesh.__class__ is Mesh, f"I need an abstract mesh."
        self._abstract = abstract_mesh
        self._tgm = None
        self.___config_including___ = None
        self._levels = dict()
        self._freeze()

    @property
    def ___is_msehtt_multigrid_partial_mesh___(self):
        return True

    @property
    def abstract(self):
        """return the abstract mesh instance."""
        return self._abstract

    @property
    def tgm(self):
        """Raise Error if it is not set yet!"""
        if self._tgm is None:
            raise Exception(f"{self}'s tgm is empty! Config me first.")
        return self._tgm

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    def info(self):
        r"""informing self."""
        print(f"msehtt-multigrid-partial-mesh: {self.abstract._sym_repr}: ")
        lvl_range = self.tgm.level_range
        for lvl in lvl_range:
            lvl_pm = self.get_level(lvl)
            lvl_pm.info(additional_info=f'\tlvl#{lvl}: ')

    def _config(self, tgm, including):
        r"""We set this partial mesh to be the great mesh, or a part of the great mesh."""
        assert tgm.__class__.__name__ == "MseHtt_MultiGrid_GreatMesh"
        assert self._tgm is None, f"tgm must not be set."
        assert self.___config_including___ is None, f"Must not be configured yet."
        self._tgm = tgm
        self.___config_including___ = including  # basically, we just need to save the args and later use them to
        # the particular partial mesh

    def get_level(self, lvl=None):
        r""""""
        if lvl is None:
            lvl = self.tgm.max_level
        else:
            pass
        if lvl in self._levels:
            return self._levels[lvl]
        else:
            lvl_tgm = self.tgm.get_level(lvl)  # the msehtt great mesh at this level
            lvl_tpm = MseHttMeshPartial(self.abstract)
            lvl_config_including =  self._parse_lvl_config_including_(
                lvl, self.___config_including___
            )
            lvl_tpm._config(lvl_tgm, lvl_config_including)
            self._levels[lvl] = lvl_tpm
            return lvl_tpm

    def _parse_lvl_config_including_(self, lvl, top_config_including):
        r""""""
        if top_config_including == 'all':
            return top_config_including
        if top_config_including.__class__ is self.__class__:
            lvl_config_including = top_config_including.get_level(lvl)
            return lvl_config_including
        elif isinstance(top_config_including, dict):
            lvl_config_including = {}
            for key in top_config_including:
                val = top_config_including[key]
                if val.__class__ is self.__class__:
                    lvl_config_including[key] = val.get_level(lvl)
                else:
                    lvl_config_including[key] = val
            return lvl_config_including
        else:
            raise NotImplementedError(top_config_including.__class__)

    @property
    def visualize(self):
        r"""Visualize the max-level mesh."""
        return self.get_level().visualize
