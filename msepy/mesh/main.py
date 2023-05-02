# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
"""
import sys
import numpy as np

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.manifold.main import MsePyManifold
from msepy.mesh.elements import MsePyMeshElements
from msepy.mesh.coordinate_transformation import MsePyMeshCoordinateTransformation
from msepy.mesh.visualize.main import MsePyMeshVisualize
from msepy.mesh.topology.main import MsePyMeshTopology


def config(mesh, manifold, element_layout):
    """"""
    assert manifold.__class__ is MsePyManifold and \
        mesh.abstract.manifold is manifold.abstract, \
        "mesh and manifold are not compatible."
    assert manifold._regions, f"manifold is not configured yet."
    assert mesh._elements is None, f"elements are configured already."
    mesh._manifold = manifold
    mesh._elements = MsePyMeshElements(mesh)  # initialize the mesh elements.
    mesh._parse_elements_from_element_layout(element_layout)
    mesh._config_dependent_meshes()  # config all mesh on boundary or partition of the manifold.
    assert mesh.elements._index_mapping is not None, \
        f"we should have set elements._index_mapping"
    assert mesh.elements._map is not None, \
        f"we should have set elements._map"


class MsePyMesh(Frozen):
    """"""
    def __init__(self, abstract_mesh):
        self._abstract = abstract_mesh
        abstract_mesh._objective = self  # link this mesh to the abstract one.
        self._manifold = None
        self._elements = None
        self._ct = MsePyMeshCoordinateTransformation(self)
        self._visualize = None
        self._topology = None
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    @property
    def manifold(self):
        """The mse-manifold of this mesh."""
        assert self._manifold is not None, f"`mesh is not configured yet."
        return self._manifold

    @property
    def regions(self):
        """regions"""
        return self.manifold.regions

    def _regionwsie_stack(self, *ndas):
        """We use this method to stack a ndarray regions-wise. This function is very useful
        in plotting reconstruction data. Since in a region, the date are structured mesh element wise,
        so we can only plot element by element. But if we group data from elements of the same
        region, then we can plot region by region. This very much increases the plotting speed.

        Parameters
        ----------
        ndas :
            nd-arrays.

        """
        if self.n == 2:
            _SD = tuple()
            for nda in ndas:
                if isinstance(nda, dict):
                    for _ in nda:
                        assert np.ndim(nda[_]) == 2
                elif nda.__class__ is np.ndarray:
                    assert np.ndim(nda) == 2 + 1
                else:
                    raise NotImplementedError(nda.__class__)

                assert len(nda) == self._elements._num
                _sd = {}
                if isinstance(nda, dict):
                    ij = np.shape(nda[0])
                else:
                    ij = np.shape(nda)[1:]
                I, J = ij
                EGN = self.elements._numbering
                for Rn in EGN:
                    layout = self.elements._distribution[Rn]
                    region_data_shape = [ij[i] * layout[i] for i in range(2)]
                    _sd[Rn] = np.zeros(region_data_shape)
                    for j in range(layout[1]):
                        for i in range(layout[0]):
                            _sd[Rn][i * I:(i + 1) * I, j * J:(j + 1) * J] = \
                                nda[EGN[Rn][i, j]]
                _SD += (_sd,)

            _SD = _SD[0] if len(ndas) == 1 else _SD
            return _SD

        else:
            raise NotImplementedError()

    @property
    def ndim(self):
        """n"""
        return self.manifold.ndim

    @property
    def esd(self):
        """m"""
        return self.manifold.esd

    @property
    def m(self):
        return self.esd

    @property
    def n(self):
        return self.ndim

    def _parse_elements_from_element_layout(self, element_layout):
        """"""
        if isinstance(element_layout, (int, float)):
            element_layout = [element_layout for _ in range(self.ndim)]
        else:
            pass
        if not isinstance(element_layout, dict):
            _temp = dict()
            for i in self.manifold.regions:
                _temp[i] = element_layout
            element_layout = _temp
        else:
            pass

        layout = dict()
        for i in element_layout:  # element layout for i# region.
            layout_i = element_layout[i]
            assert len(layout_i) == self.ndim, \
                f"element_layout for region #{i} = {layout_i} is illegal"

            _temp = list()
            for j, layout_ij in enumerate(layout_i):
                if isinstance(layout_ij, (int, float)):
                    assert layout_ij % 1 == 0 and layout_ij >= 1, \
                        f"element_layout of region #{i} = {layout_i} is illegal."
                    layout_ij = np.array([1/layout_ij for i in range(int(layout_ij))])

                else:
                    assert np.ndim(layout_ij) == 1, f"element_layout of region #{i} = {layout_i} is illegal."
                    for _ in layout_ij:
                        assert isinstance(_, (int, float)) and _ > 0, \
                            f"element_layout of region #{i} = {layout_i} is illegal."
                    layout_ij = np.array(layout_ij) * 1.
                    layout_ij /= np.sum(layout_ij)

                assert np.round(np.sum(layout_ij), 10) == 1, \
                    f"scale layout array into 1 pls, now it is {np.sum(layout_ij)}."
                _temp.append(layout_ij)

            layout[i] = _temp

        self.elements._generate_elements_from_layout(layout)

    def _config_dependent_meshes(self):
        """"""
        # Cannot repeat `_generate_elements_from_layout`, must code a method like
        # `_generate_elements_from_region_map`.

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    @property
    def elements(self):
        """elements"""
        assert self._elements is not None, f"elements is not configured yet!"
        return self._elements

    @property
    def ct(self):
        return self._ct

    @property
    def visualize(self):
        if self._visualize is None:
            self._visualize = MsePyMeshVisualize(self)
        return self._visualize

    @property
    def topology(self):
        if self._topology is None:
            self._topology = MsePyMeshTopology(self)
        return self._topology


if __name__ == '__main__':
    # python msepy/mesh/main.py
    import __init__ as ph
    space_dim = 3
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)

    msepy, obj = ph.fem.apply('msepy', locals())

    mnf = obj['manifold']
    msh = obj['mesh']

    msepy.config(mnf)('crazy', c=0.3, periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    # msepy.config(mnf)('backward_step')
    msepy.config(msh)([3 for _ in range(space_dim)])

    msh.visualize()
    mnf.visualize()
    # print(msh.elements._layout_cache_key)
    # msh.visualize()
