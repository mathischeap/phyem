# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, SIZE, COMM


class MseHttVtuInterface(Frozen):
    r""""""

    def __init__(self, coo, connections, cell_types, periodic_setting=None, distribution_method='Naive'):
        r"""

        Parameters
        ----------
        coo
        connections
        cell_types
        periodic_setting
        distribution_method

        """
        local_cell_indices = connections.keys()
        assert local_cell_indices == cell_types.keys(), f"local cells must match."

        _global_element_type_dict = COMM.gather(cell_types, root=MASTER_RANK)

        if RANK == MASTER_RANK:

            cell_distribution = {}

            for rank, cells in enumerate(_global_element_type_dict):
                for index in cells:
                    if index not in cell_distribution:
                        cell_distribution[index] = [rank, ]
                    else:
                        cell_distribution[index].append(rank)

            shared_cells = {}
            for index in cell_distribution:
                if len(cell_distribution[index]) == 1:
                    pass
                else:
                    shared_cells[index] = cell_distribution[index]
            del cell_distribution

            _ = {}
            for __ in _global_element_type_dict:
                _.update(__)
            _global_element_type_dict = _
        else:
            pass

        _global_element_map_dict = COMM.gather(connections, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            _ = {}
            self._element_distribution = {}
            for rank, __ in enumerate(_global_element_map_dict):
                _.update(__)
                self._element_distribution[rank] = list(__.keys())
            _global_element_map_dict = _
        else:
            pass

        if RANK == MASTER_RANK:
            self._global_element_type_dict = _global_element_type_dict
            self._global_element_map_dict = _global_element_map_dict

            # noinspection PyUnboundLocalVariable
            num_shared = len(shared_cells)
            if num_shared > 0:
                redistribute = True
            else:
                redistribute = False

        else:
            redistribute = None

        redistribute = COMM.bcast(redistribute, root=MASTER_RANK)

        if redistribute:

            all_coo = COMM.gather(coo, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                _ = {}
                for __ in all_coo:
                    _.update(__)
                all_coo = _

                num_total_elements = len(self._global_element_type_dict)

                element_distribution = {}

                if distribution_method == 'Naive':
                    num_piles = 3 * (SIZE - 1) + 1
                    num_elements_each_pile = num_total_elements / num_piles  # OK to be decimal

                    elements_indices = list(self._global_element_type_dict.keys())
                    start = 0
                    for rank in range(SIZE):
                        take_num_piles = 1 if rank == MASTER_RANK else 3
                        end = int(start + take_num_piles * num_elements_each_pile) + 1
                        element_distribution[rank] = elements_indices[start:end]
                        start = end
                else:
                    raise NotImplementedError(f'vif distribution method = {distribution_method}')

                self._element_distribution = element_distribution

                coo = []
                connections = []
                cell_types = []

                for rank in range(SIZE):
                    indices = element_distribution[rank]

                    rank_connection = {}
                    rank_cell_types = {}
                    rank_coo = {}

                    for i in indices:
                        rank_connection[i] = self._global_element_map_dict[i]
                        rank_cell_types[i] = self._global_element_type_dict[i]

                        for node in rank_connection[i]:
                            if node in rank_coo:
                                pass
                            else:
                                rank_coo[node] = all_coo[node]

                    coo.append(rank_coo)
                    connections.append(rank_connection)
                    cell_types.append(rank_cell_types)

            else:
                coo = None
                connections = None
                cell_types = None

            coo = COMM.scatter(coo, root=MASTER_RANK)
            connections = COMM.scatter(connections, root=MASTER_RANK)
            cell_types = COMM.scatter(cell_types, root=MASTER_RANK)

        else:
            pass

        # --------- PERIODIC SETTING ---------------------------------------------
        if periodic_setting is None:
            pass
        else:
            raise NotImplementedError()
        # ========================================================================

        self._coo = coo
        self._connections = connections
        self._cell_types = cell_types

        self._freeze()


class MseHttVtuConfig(Frozen):
    r""""""

    def __init__(self, tgm, vtu_interface):
        r"""

        Parameters
        ----------
        tgm
        vtu_interface
        """
        self._tgm = tgm
        self._vif = vtu_interface
        self._freeze()

    def __call__(self):
        r""""""
        element_type_dict = self._vif._cell_types
        element_map_dict = self._vif._connections

        element_parameter_dict = {}
        for i in element_type_dict:
            element_parameter_dict[i] = list()
            for node in element_map_dict[i]:
                coo = self._vif._coo[node]
                element_parameter_dict[i].append(coo)
        return element_type_dict, element_parameter_dict, element_map_dict
