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
from msepy.mesh.elements.main import MsePyMeshElements
from msepy.mesh.coordinate_transformation import MsePyMeshCoordinateTransformation
from msepy.mesh.visualize.main import MsePyMeshVisualize
from msepy.mesh.topology.main import MsePyMeshTopology
from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh


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
    mesh._config_dependent_boundary_section_meshes()  # config all mesh on boundary or partition of the manifold.
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
        self._face_dict = dict()
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

    def _regionwsie_stack(self, *ndas, axis=0):
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

                if isinstance(nda, dict):
                    assert len(nda) == self._elements._num
                else:
                    assert nda.shape[axis] == self._elements._num, \
                        f"along {axis}-axis, it must have same layer as num elements"

                _sd = {}
                if isinstance(nda, dict):
                    ij = np.shape(nda[0])
                else:
                    sp = np.shape(nda)
                    if axis == 0:
                        ij = sp[1:]
                    elif axis == -1:
                        ij = sp[:-1]
                    else:
                        raise NotImplementedError('axis must be 0 or -1!')

                I, J = ij
                EGN = self.elements._numbering
                for Rn in EGN:
                    layout = self.elements._distribution[Rn]
                    region_data_shape = [ij[i] * layout[i] for i in range(2)]
                    _sd[Rn] = np.zeros(region_data_shape)
                    if axis == 0:
                        for j in range(layout[1]):
                            for i in range(layout[0]):
                                _sd[Rn][i * I:(i + 1) * I, j * J:(j + 1) * J] = \
                                    nda[EGN[Rn][i, j]]
                    elif axis == -1:
                        for j in range(layout[1]):
                            for i in range(layout[0]):
                                _sd[Rn][i * I:(i + 1) * I, j * J:(j + 1) * J] = \
                                    nda[:, :, EGN[Rn][i, j]]
                    else:
                        raise Exception()

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

    def _config_dependent_boundary_section_meshes(self):
        """"""

        from msepy.main import base

        all_msepy_manifolds = base['manifolds']

        for sym in all_msepy_manifolds:

            msepy_manifold = all_msepy_manifolds[sym]

            regions = msepy_manifold.regions

            if regions._map_type == 1:   # boundary section manifold

                if regions._base_regions is self.manifold.regions:

                    # this msepy_manifold should be built boundary section mesh over

                    abstract_manifold = msepy_manifold.abstract

                    meshes = base['meshes']

                    the_abstract_mesh = None
                    for _sym in meshes:
                        _mesh = meshes[_sym]
                        if _mesh.abstract._manifold is abstract_manifold:
                            the_abstract_mesh = _mesh.abstract

                    assert the_abstract_mesh is not None, \
                        f"must find the abstract mesh for abstract manifold {sym}."

                    boundary_section_mesh = MsePyBoundarySectionMesh(
                        self,
                        msepy_manifold,
                        the_abstract_mesh,
                    )

                    # now, we renew base['meshes']
                    for _sym in meshes:
                        _mesh = meshes[_sym]

                        if _mesh.abstract._manifold is abstract_manifold:
                            meshes[_sym] = boundary_section_mesh
                            break

                else:
                    pass
            else:
                pass

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
    space_dim = 1
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    print(manifold._sym_repr)
    mesh = ph.mesh(manifold)

    msepy, obj = ph.fem.apply('msepy', locals())

    mnf = obj['manifold']
    msh = obj['mesh']

    msepy.config(mnf)('crazy_multi', c=0.3, periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    msepy.config(msh)([3 for _ in range(space_dim)])

    # ph.config.set_embedding_space_dim(3)
    # manifold = ph.manifold(3)
    #
    # print(manifold._sym_repr)

    # mesh = ph.mesh(manifold)
    #
    # msepy, obj = ph.fem.apply('msepy', locals())
    #
    # mnf = obj['manifold']
    # msh = obj['mesh']
    #
    # msepy.config(mnf)('crazy', c=0.3, periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    # msepy.config(msh)([3 for _ in range(space_dim)])
