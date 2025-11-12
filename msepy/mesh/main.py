# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen
from tools.quadrature import Quadrature
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

    def info(self):
        """info self."""
        print(f"-{self.abstract._sym_repr}: {self.elements._num} elements.")

    @property
    def abstract(self):
        return self._abstract

    @property
    def manifold(self):
        """The mse-manifold of this mesh."""
        assert self._manifold is not None, f"mesh: {self} is not configured yet."
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
            if self.manifold._default_element_layout_maker is None:
                element_layout = [element_layout for _ in range(self.ndim)]
            else:
                element_layout = self.manifold._default_element_layout_maker(element_layout)
                assert isinstance(element_layout, dict), \
                    f"default_element_layout_maker must return a dict indicating the element layouts in all regions."
                assert len(element_layout) == len(self.manifold.regions), \
                    f"default_element_layout_maker must return a dict indicating the element layouts in all regions."
                for i in element_layout:
                    assert i in self.manifold.regions, \
                            (f"default_element_layout_maker must return a dict indicating the element layouts "
                             f"in all regions.")
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
        assert isinstance(element_layout, dict), f"element_layout must eventually be parsed as a dict!"
        for i in element_layout:  # element layout for #i region.
            layout_i = element_layout[i]
            assert layout_i is not None and len(layout_i) == self.ndim, \
                f"element_layout for region #{i} = {layout_i} is illegal"

            _temp = list()
            for j, layout_ij in enumerate(layout_i):
                if isinstance(layout_ij, (int, float)):
                    assert layout_ij % 1 == 0 and layout_ij >= 1, \
                        f"element_layout of region #{i} = {layout_i} is illegal."
                    layout_ij = np.array([1/layout_ij for i in range(int(layout_ij))])

                elif isinstance(layout_ij, str):

                    layout_ij = self._parse_str_element_layout(layout_ij)

                else:
                    assert np.ndim(layout_ij) == 1, \
                        f"element_layout of region #{i} = {layout_i} is illegal."
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
        return layout

    def _parse_str_element_layout(self, str_layout_indicator):
        """"""
        assert '-' in str_layout_indicator, \
            (f"when use a string to indicate an element layout, we must have '-' in this string, and "
             f"before '-', we have the distribution indicator like 'Lobatto' distribution and after "
             f"'-', we have a int to represent its degree. Now `str_layout_indicator`={str_layout_indicator} "
             f"of type <{str_layout_indicator.__class__.__name__}>, which is not legal.")
        indicator, degree = str_layout_indicator.split('-')
        degree = int(degree)
        assert degree > 0, f'degree = {degree} for indicator {str_layout_indicator} is wrong.'
        if indicator == 'Lobatto':
            quad = Quadrature(degree, category=indicator)
            nodes = quad.quad_nodes
            layout = np.diff(nodes)
            layout /= np.sum(layout)

        else:
            raise NotImplementedError(
                f"cannot understand indicator {str_layout_indicator}."
            )
        return layout

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
