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
        self._msepy_manifold = None  # only for configuring msepy elements. It can be None for else situation.
        self._elements = None
        self._visualize = None
        self._config_method = ''
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
                element = element_distributor(
                    i, rank_elements_type[i], rank_elements_parameter[i], rank_elements_map[i],
                    msepy_manifold=self._msepy_manifold,
                )
                rank_element_dict[i] = element
        else:
            raise NotImplementedError()

        assert self._config_method != '', f"must change this indicator!"
        if self._config_method == 'msepy':
            # _config_method == 'msepy' means we config the great mesh from a msepy mesh.
            self._elements = MseHttGreatMeshElements(self, rank_element_dict, element_face_topology_mismatch=False)
        else:
            self._elements = MseHttGreatMeshElements(self, rank_element_dict)

    def _config(self, indicator, element_layout, **kwargs):
        """"""
        assert self._config_method == '', f"I must be not-configured yet!"
        if RANK != MASTER_RANK:
            if isinstance(indicator, str) and indicator in PredefinedMsePyManifoldDistributor._predefined_manifolds():
                # We config the great mesh through a predefined msepy mesh, and we need to save the msepy manifold
                self._msepy_manifold = COMM.bcast(None, root=MASTER_RANK)
                self._config_method = 'msepy'
                # _config_method == 'msepy' means we config the great mesh from a msepy mesh.
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
                self._config_method = 'msepy'
                # _config_method == 'msepy' means we config the great mesh from a msepy mesh.
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
            all_element_map_dict,
            method='naive',
    ):
        """"""
        if SIZE == 1:
            elements_indices = list(all_element_type_dict.keys())
            elements_indices.sort()
            self._element_distribution = {MASTER_RANK: elements_indices}
            return all_element_type_dict, all_element_parameter_dict, all_element_map_dict
        else:
            pass

        if RANK == MASTER_RANK:
            elements_type = [{} for _ in range(SIZE)]
            elements_parameter = [{} for _ in range(SIZE)]
            elements_map = [{} for _ in range(SIZE)]
            element_distribution = {}  # only in the master rank

            # ----------- Different element distribution methods -------------------------------------
            if method == 'naive':
                # ------ most trivial method ---------------------------------------------------------
                elements_indices = list(all_element_type_dict.keys())
                elements_indices.sort()
                num_total_elements = len(elements_indices)

                num_piles = 3 * (SIZE - 1) + 1  # master rank takes 1 pile, other ranks take 3 piles each.
                num_elements_each_pile = num_total_elements / num_piles  # OK to be decimal

                start = 0
                for rank in range(SIZE):
                    take_num_piles = 1 if rank == MASTER_RANK else 3
                    end = int(start + take_num_piles * num_elements_each_pile) + 1
                    element_distribution[rank] = elements_indices[start:end]
                    for i in element_distribution[rank]:
                        elements_type[rank][i] = all_element_type_dict[i]
                        elements_parameter[rank][i] = all_element_parameter_dict[i]
                        elements_map[rank][i] = all_element_map_dict[i]
                    start = end

            else:
                raise NotImplementedError(f"Please implement better element distributor late. It helps a lot.")

            # ------------- check distribution ----------------------------------------------------------------
            total_element_indices_set = set()
            for rank in element_distribution:
                rank_indices = set(element_distribution[rank])
                num_elements = len(rank_indices)
                assert len(elements_type[rank]) == num_elements, f"elements_type dict wrong."
                assert len(elements_parameter[rank]) == num_elements, f"elements_parameter dict wrong."
                assert len(elements_map[rank]) == num_elements, f"elements_map dict wrong."
                for index in rank_indices:
                    assert index in elements_type[rank], f"element #{index} missing in elements_type dict."
                    assert index in elements_parameter[rank], f"element #{index} missing in elements_type dict."
                    assert index in elements_map[rank], f"element #{index} missing in elements_type dict."
                total_element_indices_set.update(rank_indices)
            for i in total_element_indices_set:
                assert i in all_element_type_dict, f"element #{i} missing in elements_type dict."
                assert i in all_element_parameter_dict, f"element #{i} missing in elements_type dict."
                assert i in all_element_map_dict, f"element #{i} missing in elements_type dict."
            assert len(total_element_indices_set) == len(all_element_type_dict), f"elements_type dict wrong."
            assert len(total_element_indices_set) == len(all_element_parameter_dict), f"elements_parameter dict wrong."
            assert len(total_element_indices_set) == len(all_element_map_dict), f"elements_map dict wrong."
            # =================================================================================================

        else:
            assert all_element_type_dict is None, f"we must distribute only from the master core."
            assert all_element_parameter_dict is None, f"we must distribute only from the master core."
            assert all_element_map_dict is None, f"we must distribute only from the master core."

            elements_type, elements_parameter, elements_map = None, None, None

        # ------ distribute and save data ---------------------------------------------------
        rank_elements_type = COMM.scatter(elements_type, root=MASTER_RANK)
        rank_elements_parameter = COMM.scatter(elements_parameter, root=MASTER_RANK)
        rank_elements_map = COMM.scatter(elements_map, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            # noinspection PyUnboundLocalVariable
            self._element_distribution = element_distribution
        else:
            pass
        return rank_elements_type, rank_elements_parameter, rank_elements_map
