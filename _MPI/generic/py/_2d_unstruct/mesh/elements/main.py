# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
from tools.frozen import Frozen
from generic.py._2d_unstruct.mesh.elements.distributor import distributor
from src.config import RANK, MASTER_RANK, COMM, SIZE


class MPI_Py_2D_Unstructured_MeshElements(Frozen):
    """"""

    def __init__(self, type_dict, vertex_dict, vertex_coordinates, same_vertices_dict):
        """

        Parameters
        ----------
        type_dict
        vertex_dict
        vertex_coordinates
        same_vertices_dict
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
            element_distribution = self._distribute_elements_to_ranks(element_map)

            # ---- these two global properties only stored in the master core -------------
            self._total_map = element_map
            self._element_distribution = element_distribution
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
            TYPE_DICT = None
            VERTEX_DICT = None
            COORDINATES_DICT = None
            MAP = None

        type_dict = COMM.scatter(TYPE_DICT, root=MASTER_RANK)
        vertex_dict = COMM.scatter(VERTEX_DICT, root=MASTER_RANK)
        vertex_coordinates = COMM.scatter(COORDINATES_DICT, root=MASTER_RANK)
        self._elements_dict = self._make_elements(type_dict, vertex_dict, vertex_coordinates)
        self._map = COMM.scatter(MAP, root=MASTER_RANK)
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return rf"<MPI-generic-py-unstructured-2d-mesh @ RANK {RANK}" + super_repr

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

            if ele_typ == 'regular quadrilateral':
                pass
            elif ele_typ == 'regular triangle':
                sequence = (0, 1, 2)
                vertex_dict[index] = tuple([ele_vertices[seq] for seq in sequence])
                element_coordinates = [element_coordinates[seq] for seq in sequence]
            else:
                raise NotImplementedError(f"cannot make a {ele_typ} element.")

            element_dict[index] = distributor(ele_typ, element_coordinates)

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
            return indices_distribution
        else:
            raise NotImplementedError()

    @staticmethod
    def _element_amount_in_ranks(total_num_elements, master_loading_factor=0.8):
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
    # ==================================================================================================

    def __iter__(self):
        """iteration over local elements by indices"""
        for index in self._elements_dict:
            yield index

    def __getitem__(self, index):
        """get a local element by index"""
        return self._elements_dict[index]

    def __len__(self):
        """How many local elements?"""
        return len(self._elements_dict)

    def __contains__(self, index):
        """In element #index is a valid local element?"""
        return index in self._elements_dict

    # ------------- methods ----------------------------------------------------------------------------
    def visualize(
            self,
            density=10,
            top_right_bounds=False,
            saveto=None,
            dpi=200,
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
