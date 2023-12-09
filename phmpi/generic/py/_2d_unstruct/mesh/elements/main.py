# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM, SIZE

from legacy.generic.py._2d_unstruct.mesh.elements.distributor import distributor, distributor_with_cache

from phmpi.generic.py._2d_unstruct.mesh.elements.coordinate_transformation import CT
from phmpi.generic.py._2d_unstruct.mesh.boundary_section.face import _MPI_PY_2d_Face


class MPI_Py_2D_Unstructured_MeshElements(Frozen):
    """"""

    def __init__(
            self,
            type_dict,
            vertex_dict,
            vertex_coordinates,
            same_vertices_dict,
            element_distribution=None
    ):
        """

        Parameters
        ----------
        type_dict
        vertex_dict
        vertex_coordinates
        same_vertices_dict
        element_distribution :
            When provide `element_distribution` (in the master core only), we will use this distribution, otherwise,
            the program distributes the elements automatically.

        """
        assert isinstance(type_dict, dict), f"type_dict must be a dict."
        assert isinstance(vertex_dict, dict), f"vertex_dict must be a dict."
        assert isinstance(vertex_coordinates, dict), f"vertex_coordinates must be a dict."
        assert isinstance(same_vertices_dict, dict), f"same_vertices_dict must be a dict."

        type_dict = COMM.gather(type_dict, root=MASTER_RANK)
        vertex_dict = COMM.gather(vertex_dict, root=MASTER_RANK)
        vertex_coordinates = COMM.gather(vertex_coordinates, root=MASTER_RANK)
        same_vertices_dict = COMM.gather(same_vertices_dict, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            _ = {}
            for d in type_dict:
                _.update(d)
            type_dict = _
            _ = {}
            for d in vertex_dict:
                _.update(d)
            vertex_dict = _
            _ = {}
            for d in vertex_coordinates:
                _.update(d)
            vertex_coordinates = _
            _ = {}
            for d in same_vertices_dict:
                _.update(d)
            same_vertices_dict = _

            element_map = self._make_element_map(vertex_dict, same_vertices_dict)
            if element_distribution is None:
                element_distribution = self._distribute_elements_to_ranks(element_map)
            else:
                pass

            # ---- these global properties only stored in the master core -----------------
            self._total_map = element_map
            self._element_distribution = element_distribution
            self._total_type = type_dict
            # ---- below, we distribute data for initializing elements in each core -------
            TYPE_DICT = list()
            VERTEX_DICT = list()
            COORDINATES_DICT = list()
            MAP = list()
            for indices in element_distribution:
                local_type = dict()
                local_vertex = dict()
                local_coordinates = dict()
                local_map = dict()
                for index in indices:
                    local_type[index] = type_dict[index]
                    local_vertex[index] = vertex_dict[index]
                    for vertex in vertex_dict[index]:
                        local_coordinates[vertex] = vertex_coordinates[vertex]
                    local_map[index] = self._total_map[index]
                TYPE_DICT.append(local_type)
                VERTEX_DICT.append(local_vertex)
                COORDINATES_DICT.append(local_coordinates)
                MAP.append(local_map)
        else:
            self._total_map = None
            self._element_distribution = None
            self._total_type = None
            TYPE_DICT = None
            VERTEX_DICT = None
            COORDINATES_DICT = None
            MAP = None

        type_dict = COMM.scatter(TYPE_DICT, root=MASTER_RANK)
        vertex_dict = COMM.scatter(VERTEX_DICT, root=MASTER_RANK)
        vertex_coordinates = COMM.scatter(COORDINATES_DICT, root=MASTER_RANK)
        self._elements_dict, self._element_type_indices_dict = self._make_elements(
            type_dict, vertex_dict, vertex_coordinates)
        self._map = COMM.scatter(MAP, root=MASTER_RANK)

        # =================================================================================
        self._domain_area = None
        self._all_edges = None
        self._opposite_outer_orientation_pairs = None
        self._opposite_inner_orientation_pairs = None
        self._ct = CT(self)
        self._boundary_face_cache = {}
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return rf"<MPI-generic-py-un-struct-2d-mesh@RANK{RANK}" + super_repr

    @staticmethod
    def _make_elements(type_dict, vertex_dict, vertex_coordinates):
        """

        Parameters
        ----------
        type_dict
        vertex_dict
        vertex_coordinates

        Returns
        -------
        element_dict : dict
            {
                element index: element body,
                ...,
            }
        element_type_indices_dict : dict
            {
                'rq': list of element indices of this element type,
                'rt': ...,
                ...,
            }

        """
        element_dict = dict()
        element_type_indices_dict = {
            'rq': list(),
            'rt': list(),
        }
        for index in type_dict:
            ele_typ = type_dict[index]
            ele_vertices = vertex_dict[index]
            element_coordinates = list()

            for vertex in ele_vertices:
                element_coordinates.append(vertex_coordinates[vertex])

            if ele_typ == 'rq':   # regular quadrilateral
                element_type_indices_dict['rq'].append(index)
            elif ele_typ == 'rt':   # regular triangle
                element_type_indices_dict['rt'].append(index)
            else:
                raise NotImplementedError(f"cannot make a {ele_typ} element.")

            element_dict[index] = distributor_with_cache(ele_typ, element_coordinates)
        return element_dict, element_type_indices_dict

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
                ...,
            }

        """
        assert RANK == MASTER_RANK, f"make element map only in the master core."
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

    def _distribute_elements_to_ranks(self, element_map, method='naive'):
        """"""
        assert RANK == MASTER_RANK, f"make element map only in the master core."
        element_indices = list(element_map.keys())
        total_num_elements = len(element_indices)
        amount_distribution = self._element_amount_in_ranks(total_num_elements)
        if method == 'naive':
            current = 0
            indices_distribution = list()
            for num in amount_distribution:
                indices_distribution.append(
                    element_indices[current:current+num]
                )
                current += num
            assert current == total_num_elements, f"must have distributed all elements."
            return indices_distribution
        else:
            raise NotImplementedError()

    @staticmethod
    def _element_amount_in_ranks(total_num_elements, master_loading_factor=0.75):
        """

        Parameters
        ----------
        total_num_elements
        master_loading_factor:
            It must be in [0, 1], when it is lower, the master core loading is lower.

        Returns
        -------

        """
        if SIZE == 1:
            amount_distribution = [total_num_elements, ]
        elif SIZE <= 2:
            amount_distribution = \
                [total_num_elements // SIZE + (1 if x < total_num_elements % SIZE else 0) for x in range(SIZE)]
            amount_distribution = amount_distribution[::-1]
        else:
            amount_distribution = \
                [total_num_elements // SIZE + (1 if x < total_num_elements % SIZE else 0) for x in range(SIZE)]
            master_amount = amount_distribution[-1]
            slaves_amount = amount_distribution[:-1]
            if master_amount > 10:
                final_master_amount = int(master_amount * master_loading_factor)
                for_slaves = master_amount - final_master_amount
            else:
                final_master_amount = master_amount
                for_slaves = 0

            if for_slaves > 0:
                slave_num = SIZE - 1
                for_slaves = [
                    for_slaves // slave_num + (1 if x < for_slaves % slave_num else 0) for x in range(slave_num)
                ]
                final_slave_amount = list()
                for i, num in enumerate(slaves_amount):
                    final_slave_amount.append(
                        num + for_slaves[i]
                    )
            else:
                final_slave_amount = slaves_amount
            amount_distribution = [final_master_amount, ] + final_slave_amount
        return amount_distribution

    # ---------- properties ----------------------------------------------------------------------------
    @property
    def map(self):
        return self._map

    @property
    def n(self):
        """the dimensions of the mesh"""
        return 2

    @property
    def ct(self):
        """coordinate transformation."""
        return self._ct

    @property
    def domain_area(self):
        """the total area of the domain."""
        if self._domain_area is None:
            domain_area = 0
            for index in self._elements_dict:
                domain_area += self._elements_dict[index].area
            domain_area = COMM.gather(domain_area, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                domain_area = sum(domain_area)
            else:
                pass
            domain_area = COMM.bcast(domain_area, root=MASTER_RANK)
            self._domain_area = domain_area
        return self._domain_area

    # ==================================================================================================

    def __iter__(self):
        """iteration over local elements by indices."""
        for index in self._elements_dict:
            yield index

    def __getitem__(self, index):
        """get a local element by index."""
        return self._elements_dict[index]

    def __len__(self):
        """How many local elements?"""
        return len(self._elements_dict)

    def __contains__(self, index):
        """In element #index is a valid local element?"""
        return index in self._elements_dict

    def _make_boundary_face(self, element_face_index):
        """"""
        if element_face_index in self._boundary_face_cache:
            return self._boundary_face_cache[element_face_index]
        else:
            element_index, face_index = element_face_index
            assert element_index in self, f"element #{element_index} is illegal (or not local)."
            face = _MPI_PY_2d_Face(self, element_index, face_index)
            self._boundary_face_cache[element_face_index] = face
            return face

    # ------------- methods ----------------------------------------------------------------------------
    def visualize(
            self,
            density=10,
            top_right_bounds=False,
            saveto=None,
            dpi=200,
            title=None,
    ):
        if density < 10:
            density = 10
        else:
            pass
        local_lines = list()
        for index in self:
            element = self[index]
            lines = element._plot_lines(density)
            local_lines.append(lines)

        local_lines = COMM.gather(local_lines, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            return
        else:
            pass

        import matplotlib.pyplot as plt
        import matplotlib
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
        })

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

        for rank_lines in local_lines:
            for element_lines in rank_lines:
                for line in element_lines:
                    plt.plot(*line, linewidth=0.5, color='gray')

        if title is None:
            pass
        else:
            plt.title(title)

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

    def find_rank_of_element(self, index):
        """Return None if the element #index is nowhere."""
        if RANK == MASTER_RANK:
            found_rank = None
            for rank, indices in enumerate(self._element_distribution):
                if index in indices:
                    found_rank = rank
                    break
                else:
                    pass
        else:
            found_rank = None
        found_rank = COMM.bcast(found_rank, root=MASTER_RANK)
        if found_rank == RANK:
            assert index in self, f'must be, a double check!'
        else:
            assert index not in self, f"double check, an element is only at one place."
        return found_rank

    # ------------------------------------------------------------------------------------------------
    @property
    def all_edges(self):
        """

        Returns
        -------
        _all_edges = {
            (0, 1): ([index, j], [...]),
                # the edge from `vertex 0` to `vertex 1` is the `j`th edge of element #`index` and  ...
            ....,
        }

        """
        if RANK != MASTER_RANK:
            return None
        else:
            if self._all_edges is None:
                self._all_edges = dict()
                for index in self._total_map:
                    vertices = self._total_map[index]
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
            if RANK == MASTER_RANK:
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
                            sign = distributor(self._total_type[index]).outer_orientations(edge_index)
                            positions.append(position)
                            signs.append(sign)
                        for position in neg_positions:
                            index, edge_index = position
                            sign = distributor(self._total_type[index]).outer_orientations(edge_index)
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
            else:
                pairs = None

            if RANK == MASTER_RANK:
                pairs_distribution = list()
                for rank in range(SIZE):
                    pairs_distribution.append(dict())

                for location in pairs:
                    element_index = location[0]
                    for rank, indices in enumerate(self._element_distribution):
                        if element_index in indices:
                            pairs_distribution[rank][location] = pairs[location]
                            break
                        else:
                            pass
            else:
                pairs_distribution = None

            pairs = COMM.scatter(pairs_distribution, root=MASTER_RANK)

            self._opposite_outer_orientation_pairs = pairs

        return self._opposite_outer_orientation_pairs

    @property
    def opposite_inner_orientation_pairs(self):
        """"""
        if self._opposite_inner_orientation_pairs is None:

            if RANK == MASTER_RANK:
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
                            sign = distributor(self._total_type[index]).inner_orientations(edge_index)
                            positions.append(position)
                            signs.append(sign)

                        for position in neg_positions:
                            index, edge_index = position
                            sign = distributor(self._total_type[index]).inner_orientations(edge_index)
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
            else:
                pairs = None

            if RANK == MASTER_RANK:
                pairs_distribution = list()
                for rank in range(SIZE):
                    pairs_distribution.append(dict())

                for location in pairs:
                    element_index = location[0]
                    for rank, indices in enumerate(self._element_distribution):
                        if element_index in indices:
                            pairs_distribution[rank][location] = pairs[location]
                            break
                        else:
                            pass
            else:
                pairs_distribution = None

            pairs = COMM.scatter(pairs_distribution, root=MASTER_RANK)

            self._opposite_inner_orientation_pairs = pairs

        return self._opposite_inner_orientation_pairs
