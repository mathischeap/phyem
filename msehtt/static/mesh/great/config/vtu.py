# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, SIZE, COMM


class MseHttVtuInterface(Frozen):
    r""""""

    def __init__(
            self, coo, connections, cell_types, periodic_setting=None, distribution_method='Naive',
            redistribute=None,
    ):
        r"""

        Parameters
        ----------
        coo : dict
        connections : dict
        cell_types : dict
        periodic_setting
        distribution_method : str
        redistribute :

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

        if redistribute is None:
            if RANK == MASTER_RANK:
                # noinspection PyUnboundLocalVariable
                num_shared = len(shared_cells)
                if num_shared > 0:
                    redistribute = True
                else:
                    redistribute = False

            else:
                redistribute = None

            redistribute = COMM.bcast(redistribute, root=MASTER_RANK)
        else:
            pass

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
                    if num_total_elements < SIZE:
                        raise r"number of elements is lower than SIZE, reduce SIZE."

                    elif SIZE == 1:
                        element_distribution[0] = list(self._global_element_type_dict.keys())

                    elif SIZE == num_total_elements:
                        for rank, e in enumerate(self._global_element_map_dict):
                            element_distribution[rank] = [e]

                    else:
                        element_distribution = ___master_rank_distribute___(
                            all_coo, self._global_element_map_dict
                        )

                    # elif num_total_elements < 3 * SIZE:
                    #     rank_indices = np.array_split(range(num_total_elements), SIZE)
                    #     elements_indices = list(self._global_element_type_dict.keys())
                    #     for rank, indices in enumerate(rank_indices):
                    #         low, upper = min(indices), max(indices) + 1
                    #         element_distribution[rank] = elements_indices[low:upper]
                    #
                    # else:
                    #     num_piles = 3 * (SIZE - 1) + 1
                    #     num_elements_each_pile = num_total_elements / num_piles  # OK to be decimal
                    #
                    #     elements_indices = list(self._global_element_type_dict.keys())
                    #     start = 0
                    #     for rank in range(SIZE):
                    #         take_num_piles = 1 if rank == MASTER_RANK else 3
                    #         end = int(start + take_num_piles * num_elements_each_pile) + 1
                    #         element_distribution[rank] = elements_indices[start:end]
                    #         start = end

                else:
                    raise NotImplementedError(f'vif distribution method = {distribution_method}')

                for rank in range(SIZE):
                    if len(element_distribution[rank]) == 0:
                        raise Exception(f"RANK{rank} is distributed no element. Decrease the SIZE or refine the mesh.")
                    else:
                        pass

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
            raise NotImplementedError(f"periodic setting={periodic_setting} not implemented!")
        # ========================================================================
        self._coo = coo
        self._connections = connections
        self._cell_types = cell_types
        self._freeze()


def ___master_rank_distribute___(coo_dict, map_dict):
    r"""This function will only be called in the master rank."""
    num_total_elements = len(map_dict)
    if num_total_elements < 4 * SIZE:
        rank_element_numbers = ___split___(num_total_elements, SIZE)
    else:
        rank_element_numbers = [0 for _ in range(SIZE)]
        rank_numbers = ___split___(num_total_elements, SIZE)
        master_rank_element_number = rank_numbers[MASTER_RANK] // 2
        rank_element_numbers[MASTER_RANK] = master_rank_element_number
        to_be_sent_to_slaves = rank_numbers[MASTER_RANK] - master_rank_element_number
        to_be_sent_to_slaves = ___split___(to_be_sent_to_slaves, SIZE - 1)
        i = 0
        for rank in range(SIZE):
            if rank != MASTER_RANK:
                rank_element_numbers[rank] = rank_numbers[rank] + to_be_sent_to_slaves[i]
                i += 1
            else:
                pass

    assert sum(rank_element_numbers) == num_total_elements, f"must be!"

    distributed_elements = []
    remaining_elements = list(map_dict.keys())

    element_centers = dict()
    for e in map_dict:
        _map = map_dict[e]
        coordinates = []
        for m in _map:
            coordinates.append(coo_dict[m])
        coordinates = np.array(coordinates)
        element_centers[e] = np.sum(coordinates, axis=0) / len(coordinates)

    element_distribution = {}

    for rank in range(SIZE):
        if len(remaining_elements) == 0:
            element_distribution[rank] = []
        else:
            referencing_element_index = remaining_elements[0]
            reference_center = element_centers[referencing_element_index]
            distance_dict = dict()
            all_distances = list()
            for e in remaining_elements:
                distance = np.sqrt(np.sum((element_centers[e] - reference_center) ** 2))
                distance_dict[e] = distance
                all_distances.append(distance)
            all_distances.sort()
            if rank_element_numbers[rank] >= len(all_distances):
                reference_distance = all_distances[-1] + 1
            else:
                reference_distance = all_distances[rank_element_numbers[rank]]
            rank_elements = []
            for e in remaining_elements:
                if distance_dict[e] <= reference_distance:
                    rank_elements.append(e)
                else:
                    pass
            element_distribution[rank] = rank_elements
            distributed_elements.extend(rank_elements)
            for e in rank_elements:
                remaining_elements.remove(e)

    assert len(remaining_elements) == 0 and len(distributed_elements) == num_total_elements, f"must be!"
    return element_distribution


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


def ___split___(x, n):
    r""""""
    # If we cannot split the number into exactly 'N' parts
    res = []
    if x < n:
        res = [0 for _ in range(n)]
        for i in range(x):
            res[i] = 1
        res = res[::-1]
    # If x % n == 0 then the minimum difference is 0 and all numbers are x / n
    elif x % n == 0:
        for i in range(n):
            res.append(x // n)
    else:
        # upto n-(x % n) the values will be x / n after that the values will be x / n + 1
        zp = n - (x % n)
        pp = x // n
        for i in range(n):
            if i >= zp:
                res.append(pp + 1)
            else:
                res.append(pp)

    return res


if __name__ == '__main__':
    res = ___split___(5, 10)
    print(res)
