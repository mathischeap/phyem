# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from src.mesh import Mesh
from msehtt_ncf.static.mesh.partial.elements.main import MseHtt_NCF_Elements_PartialMesh


class EmptyCompositionError(Exception):
    """"""


class MseHtt_NCF_Static_PartialMesh(Frozen):
    r""""""

    def __init__(self, abstract_mesh):
        r""""""
        assert abstract_mesh.__class__ is Mesh, f"I need an abstract mesh."
        self._abstract = abstract_mesh
        self._tgm = None
        self._composition = None
        self._freeze()

    def info(self):
        r"""info self."""
        try:
            composition = self.composition
        except EmptyCompositionError:
            print(f"Mesh not-configured: {self.abstract._sym_repr}.")
        else:
            composition.info()

    @property
    def ___is_msehtt_ncf_partial_mesh___(self):
        r""""""
        return True

    @property
    def abstract(self):
        r"""return the abstract mesh instance."""
        return self._abstract

    @property
    def tgm(self):
        r"""Raise Error if it is not set yet!"""
        if self._tgm is None:
            raise Exception('tgm is empty!')
        return self._tgm

    @property
    def composition(self):
        r"""The composition; the main body of this partial mesh."""
        if self._composition is None:
            raise EmptyCompositionError(f"msehtt-ncf partial mesh of {self.abstract._sym_repr} has no composition yet")
        else:
            return self._composition

    @property
    def visualize(self):
        """Call the visualization scheme of the composition."""
        return self.composition.visualize

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    def _config(self, tgm, including):
        r""""""
        assert self._tgm is None, f"tgm must be set."
        assert self._composition is None, f"components are not set!"
        self._tgm = tgm

        if including == 'all':
            # CONFIGURATION 1: -----------------------------------------------------------
            # this partial mesh includes all elements of the great mesh.
            including = {
                'type': 'local great elements',
                'range': self._tgm.elements._elements_dict.keys()
            }

        else:
            raise NotImplementedError()

        self._perform_configuration(including)

    def _perform_configuration(self, including):
        r"""Really do the configuration."""
        _type = including['type']
        if _type == 'local great elements':
            # CONFIGURATION 1 ===========================================================================
            # this partial mesh consists of local (rank) elements of the great mesh.
            rank_great_element_range = including['range']  # the range of local great elements.
            self._composition = MseHtt_NCF_Elements_PartialMesh(self, self._tgm, rank_great_element_range)

        else:
            raise NotImplementedError()
