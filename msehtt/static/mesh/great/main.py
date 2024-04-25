# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, SIZE, COMM
from msepy.manifold.predefined.distributor import PredefinedMsePyManifoldDistributor
from msehtt.static.mesh.great.config.msepy_ import MseHttMsePyConfig
from msehtt.static.mesh.great.elements.main import MseHttGreatMeshElements
from msehtt.static.mesh.great.visualize.main import MseHttGreatMeshVisualize
from msehtt.static.mesh.great.elements.types.distributor import MseHttGreatMeshElementDistributor


class MseHttGreatMesh(Frozen):
    """"""

    def __init__(self):
        """"""
        self._msepy_manifold = None  # only for configuring msepy elements.
        self._elements = None
        self._visualize = None
        if RANK == MASTER_RANK:
            self._global_element_type_dict = None
            self._global_element_map_dict = None
            self._element_distribution = None
        else:
            pass

        self._freeze()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__}" + super_repr

    @property
    def elements(self):
        """Return all great elements instance if it exists."""
        if self._elements is None:
            raise Exception('No great elements found!')
        else:
            return self._elements

    @property
    def visualize(self):
        if self._visualize is None:
            self._visualize = MseHttGreatMeshVisualize(self)
        return self._visualize

    def _make_elements_(self, rank_elements_type, rank_elements_parameter, rank_elements_map):
        """"""
        assert self._elements is None, f"elements exist, do not renew them!"
        self._check_elements(rank_elements_type, rank_elements_parameter, rank_elements_map)
        element_distributor = MseHttGreatMeshElementDistributor()
        rank_element_dict = {}
        if self._msepy_manifold is not None:   # we are configuring from a msepy mesh (manifold).
            for i in rank_elements_type:
                rank_element_dict[i] = element_distributor(
                    i, rank_elements_type[i], rank_elements_parameter[i], rank_elements_map[i],
                    msepy_manifold=self._msepy_manifold,
                )
        else:
            raise NotImplementedError()

        self._elements = MseHttGreatMeshElements(self, rank_element_dict)

    def _config(self, indicator, element_layout, **kwargs):
        """"""
        if RANK != MASTER_RANK:
            if isinstance(indicator, str) and indicator in PredefinedMsePyManifoldDistributor._predefined_manifolds():
                # We config the great mesh through a predefined msepy mesh, and we need to save the msepy manifold
                self._msepy_manifold = COMM.bcast(None, root=MASTER_RANK)
            else:
                raise NotImplementedError()

            element_type_dict, element_parameter_dict, element_map_dict = (
                self._distribute_elements_to_ranks(
                    None, None, None
                )
            )

        else:
            if isinstance(indicator, str) and indicator in PredefinedMsePyManifoldDistributor._predefined_manifolds():
                # We config the great mesh through a predefined msepy mesh.
                config = MseHttMsePyConfig(self, indicator)
                element_type_dict, element_parameter_dict, element_map_dict, msepy_manifold = (
                    config(element_layout, **kwargs)
                )
                # We need to save the msepy manifold
                self._msepy_manifold = COMM.bcast(msepy_manifold, root=MASTER_RANK)
            else:
                raise NotImplementedError()

            self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)

            self._global_element_type_dict = element_type_dict
            self._global_element_map_dict = element_map_dict

            element_type_dict, element_parameter_dict, element_map_dict = (
                self._distribute_elements_to_ranks(
                    element_type_dict, element_parameter_dict, element_map_dict
                )
            )

        self._make_elements_(element_type_dict, element_parameter_dict, element_map_dict)

    @staticmethod
    def _check_elements(element_type_dict, element_parameter_dict, element_map_dict):
        """"""
        assert len(element_type_dict) == len(element_parameter_dict) == len(element_map_dict), f"must be!"
        for i in element_type_dict:
            assert i in element_parameter_dict and i in element_map_dict, f"must be!"

    def _distribute_elements_to_ranks(
            self,
            all_element_type_dict,
            all_element_parameter_dict,
            all_element_map_dict
    ):
        """"""
        if SIZE == 1:
            return all_element_type_dict, all_element_parameter_dict, all_element_map_dict
        else:
            pass

        if RANK == MASTER_RANK:
            elements_type = [{} for _ in range(SIZE)]
            elements_parameter = [{} for _ in range(SIZE)]
            elements_map = [{} for _ in range(SIZE)]

            elements_indices = list(all_element_type_dict.keys())
            elements_indices.sort()
            num_total_elements = len(elements_indices)

            num_piles = 3 * (SIZE - 1) + 1  # master rank takes 1 pile, other ranks take 3 piles each.
            num_elements_each_pile = num_total_elements / num_piles  # OK to be decimal

            start = 0
            element_distribution = {}
            for rank in range(SIZE):
                take_num_piles = 1 if rank == MASTER_RANK else 3
                end = int(start + take_num_piles * num_elements_each_pile) + 1
                element_distribution[rank] = elements_indices[start:end]
                for i in elements_indices[start:end]:
                    elements_type[rank][i] = all_element_type_dict[i]
                    elements_parameter[rank][i] = all_element_parameter_dict[i]
                    elements_map[rank][i] = all_element_map_dict[i]
                start = end
            self._element_distribution = element_distribution

        else:
            assert all_element_type_dict is None, f"we must distribute only from the master core."
            assert all_element_parameter_dict is None, f"we must distribute only from the master core."
            assert all_element_map_dict is None, f"we must distribute only from the master core."

            elements_type, elements_parameter, elements_map = None, None, None

        rank_elements_type = COMM.scatter(elements_type, root=MASTER_RANK)
        rank_elements_parameter = COMM.scatter(elements_parameter, root=MASTER_RANK)
        rank_elements_map = COMM.scatter(elements_map, root=MASTER_RANK)

        return rank_elements_type, rank_elements_parameter, rank_elements_map
