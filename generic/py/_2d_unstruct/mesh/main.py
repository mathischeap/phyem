# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
from tools.frozen import Frozen
from generic.py._2d_unstruct.mesh.elements.distributor import distributor
from generic.py._2d_unstruct.mesh.coordinate_transformation import Py2CoordinateTransformation
from generic.py._2d_unstruct.mesh.boundary_section.face import Face


class GenericUnstructuredMesh2D(Frozen):
    """"""

    def __init__(self, type_dict, vertex_dict, vertex_coordinates, same_vertices_dict):
        """"""
        assert isinstance(type_dict, dict), f"type_dict must be a dict."
        assert isinstance(vertex_dict, dict), f"vertex_dict must be a dict."
        assert isinstance(vertex_coordinates, dict), f"vertex_coordinates must be a dict."
        assert isinstance(same_vertices_dict, dict), f"same_vertices_dict must be a dict."

        self._elements_dict = self._make_elements(type_dict, vertex_dict, vertex_coordinates)
        self._map = self._make_element_map(vertex_dict, same_vertices_dict)

        # properties ------------------------------------------------------------------------
        self._domain_area = None
        self._all_edges = None
        self._opposite_outer_orientation_pairs = None
        self._opposite_inner_orientation_pairs = None
        self._ct = Py2CoordinateTransformation(self)
        self._boundary_face_cache = {}
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Py2-mesh of {len(self)}-elements" + super_repr + '>'

    @staticmethod
    def _make_elements(type_dict, vertex_dict, vertex_coordinates):
        """"""
        element_dict = dict()
        for index in type_dict:
            ele_typ = type_dict[index]
            ele_vertices = vertex_dict[index]
            element_coordinates = list()
            for vertex in ele_vertices:
                element_coordinates.append(vertex_coordinates[vertex])

            if ele_typ == 'rq':
                pass
            elif ele_typ == 'rt':
                sequence = (0, 1, 2)
                vertex_dict[index] = tuple([ele_vertices[seq] for seq in sequence])
                element_coordinates = [element_coordinates[seq] for seq in sequence]
            else:
                raise NotImplementedError(f"cannot make a {ele_typ} element.")

            element_dict[index] = distributor(ele_typ)(element_coordinates)

        return element_dict

    @staticmethod
    def _make_element_map(vertex_dict, same_vertices_dict):
        """Note that the numbering of vertices may jump. So it could be 0, 1, 2, 3, 5, 7, ..., since the original
        numbered 4 vertex may be renumbered into something else because of the periodic boundary condition.

        Parameters
        ----------
        vertex_dict
        same_vertices_dict

        Returns
        -------
        real_vertex_dict :
            {
                index: (i, j, k, ....),  # the vertices of element #`index` are `i, j, k, ...`.
                ...
            }

        """
        if same_vertices_dict == {}:
            real_vertex_dict = vertex_dict
        else:
            real_vertex_dict = dict()
            for index in vertex_dict:
                local_vertices = list(vertex_dict[index])
                for j, vertex in enumerate(local_vertices):
                    if vertex in same_vertices_dict:
                        local_vertices[j] = same_vertices_dict[vertex]
                real_vertex_dict[index] = local_vertices
        return real_vertex_dict

    @property
    def map(self):
        return self._map

    def __iter__(self):
        for index in self._elements_dict:
            yield index

    def __getitem__(self, index):
        return self._elements_dict[index]

    def __len__(self):
        return len(self._elements_dict)

    def __contains__(self, index):
        return index in self._elements_dict

    @property
    def n(self):
        """the dimensions of the mesh"""
        return 2

    def _boundary_faces(self, index):
        """"""
        if index in self._boundary_face_cache:
            return self._boundary_face_cache[index]
        else:
            element_index, edge_index = index
            assert element_index in self, f"element #{element_index} is illegal."
            face = Face(self, element_index, edge_index)
            self._boundary_face_cache[index] = face
            return face

    # -------- properties ------------------------------------------------------------------
    @property
    def domain_area(self):
        """the total area of the domain."""
        if self._domain_area is None:
            self._domain_area = 0
            for index in self._elements_dict:
                self._domain_area += self._elements_dict[index].area
        return self._domain_area

    @property
    def all_edges(self):
        """

        Returns
        -------
        _all_edges = {
            (0, 1): ([index, j], [...]),
                # the edge from `vertex 0` to `vertex 1` is the `j`th edge of element #`index` and  ...
            ....
        }

        """
        if self._all_edges is None:
            self._all_edges = dict()
            for index in self:
                vertices = self.map[index]
                num_edges = len(vertices)
                sequence = list(vertices) + [vertices[0]]
                for j in range(num_edges):
                    edge_position = [index, j]
                    edge_indicator = (sequence[j], sequence[j+1])
                    if edge_indicator not in self._all_edges:
                        self._all_edges[edge_indicator] = tuple()
                    else:
                        pass
                    self._all_edges[edge_indicator] += (edge_position, )
        return self._all_edges

    @property
    def opposite_outer_orientation_pairs(self):
        """"""
        if self._opposite_outer_orientation_pairs is None:
            pairs = {}

            all_edges = self.all_edges
            for edge in all_edges:

                pos_positions = all_edges[edge]

                vertex0, vertex1 = edge
                if (vertex1, vertex0) in all_edges:
                    neg_positions = all_edges[(vertex1, vertex0)]
                else:
                    neg_positions = list()

                num_found_positions = len(pos_positions) + len(neg_positions)
                if num_found_positions == 1:   # must be on boundary
                    assert len(neg_positions) == 0   # must have no neg_position
                elif num_found_positions == 2:
                    positions = list()
                    signs = list()
                    for position in pos_positions:
                        index, edge_index = position
                        element = self[index]
                        sign = element.outer_orientations(edge_index)
                        positions.append(position)
                        signs.append(sign)
                    for position in neg_positions:
                        index, edge_index = position
                        element = self[index]
                        sign = element.outer_orientations(edge_index)
                        positions.append(position)
                        signs.append(sign)

                    sign0, sign1 = signs
                    pos0, pos1 = positions
                    if sign0 != sign1:
                        pass
                    else:
                        pos0 = tuple(pos0)
                        pos1 = tuple(pos1)

                        if pos0 in pairs:
                            assert pos1 not in pairs and pairs[pos0] == pos1
                        elif pos1 in pairs:
                            assert pos0 not in pairs and pairs[pos1] == pos0
                        else:
                            pos0 = tuple(pos0)
                            pos1 = tuple(pos1)

                            if pos0 in pairs:
                                assert pos1 not in pairs and pairs[pos0] == pos1
                            elif pos1 in pairs:
                                assert pos0 not in pairs and pairs[pos1] == pos0
                            else:
                                index0 = pos0[0]
                                index1 = pos1[0]
                                if isinstance(index0, str) and not isinstance(index1, str):
                                    pairs[pos0] = pos1
                                elif isinstance(index1, str) and not isinstance(index0, str):
                                    pairs[pos1] = pos0
                                elif isinstance(index0, str) and isinstance(index1, str):
                                    if len(index0) >= len(index1):
                                        pairs[pos0] = pos1
                                    else:
                                        pairs[pos1] = pos0
                                else:
                                    pairs[pos0] = pos1

                else:
                    raise Exception()

            self._opposite_outer_orientation_pairs = pairs
        return self._opposite_outer_orientation_pairs

    @property
    def opposite_inner_orientation_pairs(self):
        """"""
        if self._opposite_inner_orientation_pairs is None:
            pairs = {}

            all_edges = self.all_edges
            for edge in all_edges:

                pos_positions = all_edges[edge]

                vertex0, vertex1 = edge
                if (vertex1, vertex0) in all_edges:
                    neg_positions = all_edges[(vertex1, vertex0)]
                else:
                    neg_positions = list()

                num_found_positions = len(pos_positions) + len(neg_positions)
                if num_found_positions == 1:   # must be on boundary
                    assert len(neg_positions) == 0   # must have no neg_position
                elif num_found_positions == 2:
                    positions = list()
                    signs = list()
                    for position in pos_positions:
                        index, edge_index = position
                        element = self[index]
                        sign = element.inner_orientations(edge_index)
                        positions.append(position)
                        signs.append(sign)
                    for position in neg_positions:
                        index, edge_index = position
                        element = self[index]
                        sign = element.inner_orientations(edge_index)
                        if sign == '+':
                            sign = '-'
                        elif sign == '-':
                            sign = '+'
                        else:
                            raise Exception()
                        positions.append(position)
                        signs.append(sign)

                    sign0, sign1 = signs
                    pos0, pos1 = positions
                    if sign0 == sign1:
                        pass
                    else:
                        pos0 = tuple(pos0)
                        pos1 = tuple(pos1)

                        if pos0 in pairs:
                            assert pos1 not in pairs and pairs[pos0] == pos1
                        elif pos1 in pairs:
                            assert pos0 not in pairs and pairs[pos1] == pos0
                        else:
                            index0 = pos0[0]
                            index1 = pos1[0]
                            if isinstance(index0, str) and not isinstance(index1, str):
                                pairs[pos0] = pos1
                            elif isinstance(index1, str) and not isinstance(index0, str):
                                pairs[pos1] = pos0
                            elif isinstance(index0, str) and isinstance(index1, str):
                                if len(index0) >= len(index1):
                                    pairs[pos0] = pos1
                                else:
                                    pairs[pos1] = pos0
                            else:
                                pairs[pos0] = pos1

                else:
                    raise Exception()

            self._opposite_inner_orientation_pairs = pairs
        return self._opposite_inner_orientation_pairs

    @property
    def num_elements(self):
        return len(self)

    @property
    def ct(self):
        """coordinate transformation."""
        return self._ct

    # -------- methods ----------------------------------------------------------------------
    def visualize(
            self,
            density=10,
            top_right_bounds=False,
            saveto=None,
            dpi=200,
    ):
        """"""
        if density < 10:
            density = 10
        else:
            pass
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        if top_right_bounds:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.xlabel(r"$x$", fontsize=15)
        plt.ylabel(r"$y$", fontsize=15)
        plt.tick_params(axis='both', which='both', labelsize=12)
        for index in self:
            element = self[index]
            lines = element._plot_lines(density)
            for line in lines:
                plt.plot(*line, linewidth=0.5, color='gray')

        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight', dpi=dpi)
        else:
            from src.config import _setting, _pr_cache

            if _setting['pr_cache']:
                _pr_cache(fig, filename='mpi_msehy_py2_mesh')
            else:
                matplotlib.use('TkAgg')
                plt.tight_layout()
                plt.show(block=_setting['block'])
        plt.close()
        return fig
