# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, SIZE, COMM
from msepy.manifold.predefined.distributor import PredefinedMsePyManifoldDistributor
from msehtt.static.manifold.predefined.distributor import Predefined_Msehtt_Manifold_Distributor

from msehtt.static.mesh.great.config.msepy_ import MseHttMsePyConfig
from msehtt.static.mesh.great.config.msepy_trf import MseHttMsePy_Trf_Config
from msehtt.static.mesh.great.config.vtu import MseHttVtuConfig
from msehtt.static.mesh.great.config.vtu import MseHttVtuInterface
from msehtt.static.mesh.great.config.msehtt_ import MseHtt_Static_PreDefined_Config

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
            self._element_distribution = None  # {rank: element_indices}
        else:
            pass
        self.selfcheck()
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__}" + super_repr

    @property
    def ___is_msehtt_great_mesh___(self):
        """Just a signature."""
        return True

    @property
    def elements(self):
        """Return all great elements instance if it exists."""
        if self._elements is None:
            raise Exception('No great elements found!')
        else:
            return self._elements

    @property
    def visualize(self):
        """"""
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
            for i in rank_elements_type:
                element = element_distributor(
                    i, rank_elements_type[i], rank_elements_parameter[i], rank_elements_map[i]
                )
                rank_element_dict[i] = element

        assert self._config_method != '', f"must change this indicator!"
        if self._config_method == 'msepy':
            # _config_method == 'msepy' means we config the great mesh from a msepy mesh.
            self._elements = MseHttGreatMeshElements(self, rank_element_dict, element_face_topology_mismatch=False)
        else:
            self._elements = MseHttGreatMeshElements(self, rank_element_dict)

    def _config(self, indicator, element_layout=None, crack_config=None, ts=False, **kwargs):
        """Note that, periodic setting should be done before this configuration. This configuration can
        process crack and triangle/tetrahedron-split, but not periodic setting.

        Parameters
        ----------
        indicator
            case 1: `indicator` is str and `indicator` indicates a pre-defined msepy-manifold.
                So, `indicator`, `element_layout` and `kwargs` will be used for initialize a msepy mesh first.
                For example:
                    msehtt.config(tgm)('crazy', element_layout=K, c=c, bounds=([0, 1], [0, 1]), periodic=False)
                    msehtt.config(tgm)('crazy', element_layout=K, c=c, bounds=([0, 1], [0, 1]), periodic=True)
                    msehtt.config(tgm)('backward_step', element_layout=K)
                    msehtt.config(tgm)(
                        'crazy',
                        element_layout=K,
                        c=c,
                        bounds=([0.25, 1.25], [0.25, 1.25], [0.25, 1.25]),
                        periodic=False
                    )

                This msepy mesh then is parsed as a msehtt mesh.

                Furthermore, if `trf` in `kwargs`, we will do a `triangular refining` upon the msehtt mesh.
                For example:
                    msehtt.config(tgm)('crazy', element_layout=K, c=c, periodic=False, trf=1)
                    msehtt.config(tgm)('crazy', element_layout=K, c=c, periodic=True, trf=1)

            case 2: 'vtu_interface'
                `indicator` is a instance of class `MseHttVtuInterface`. Then we will parse the mesh
                from this `MseHttVtuInterface` object which actually is an interface to vtk mesh.

            case 3: `indicator` is str and `indicator` indicates a pre-defined msehtt-manifold.
                So, `indicator`, `element_layout` and `kwargs` will be used for initialize a msehtt mesh.
                For example:
                    msehtt.config(tgm)('chaotic', element_layout=K, c=c, periodic=False)

                Furthermore, if if `trf` in `kwargs`, we will do a `triangular refining` upon this msepy mesh.
                For example:
                    msehtt.config(tgm)('chaotic', element_layout=K, c=c, periodic=False, trf=1)

        element_layout:
            `element_layout`; it is not always needed for particular configuration. But usually it is needed.
        crack_config :
            In ADDITION to configuration indicating, it will make cracks in the mesh.
        ts
            In ADDITION to configuration indicating, it will do ``triangle/tetrahedron-split``.
        kwargs :
            Other kwargs to be passed to particular configuration.

        Returns
        -------

        """
        assert self._config_method == '', f"I must be not-configured yet!"

        if isinstance(indicator, str) and indicator in PredefinedMsePyManifoldDistributor._predefined_manifolds():

            if 'trf' in kwargs:  # we do a triangular refining upon this msepy mesh
                input_case = 'pre-defined + triangle-refining'
            else:
                input_case = 'pre-defined'

        elif indicator.__class__ is MseHttVtuInterface:
            input_case = 'vtu_interface'

        elif isinstance(indicator, str) and indicator in Predefined_Msehtt_Manifold_Distributor.defined_manifolds():

            if 'trf' in kwargs:
                raise NotImplementedError
            else:
                input_case = 'pre-defined-msehtt-static'

        else:
            raise NotImplementedError()

        if RANK != MASTER_RANK:
            if input_case == 'pre-defined':
                # We config the great mesh through a predefined msepy mesh, and we need to save the msepy manifold
                self._msepy_manifold = COMM.bcast(None, root=MASTER_RANK)
                self._config_method = 'msepy'
                # _config_method == 'msepy' means we config the great mesh from a msepy mesh.

                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        None, None, None
                    )
                )

            elif input_case == 'pre-defined + triangle-refining':
                self._msepy_manifold = COMM.bcast(None, root=MASTER_RANK)
                self._config_method = 'msepy-trf'

                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        None, None, None
                    )
                )

            elif input_case == 'vtu_interface':
                config = MseHttVtuConfig(self, indicator)
                element_type_dict, element_parameter_dict, element_map_dict = config()
                self._config_method = 'msehtt-vtu'
                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)

            elif input_case == 'pre-defined-msehtt-static':
                self._config_method = 'msehtt-static-predefined'
                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        None, None, None
                    )
                )

            else:
                raise NotImplementedError()

        else:
            if input_case == 'pre-defined':
                # We config the great mesh through a predefined msepy mesh.
                config = MseHttMsePyConfig(self, indicator)
                element_type_dict, element_parameter_dict, element_map_dict, msepy_manifold, _ = (
                    config(element_layout, **kwargs)
                )
                # We need to save the msepy manifold
                self._msepy_manifold = COMM.bcast(msepy_manifold, root=MASTER_RANK)
                self._config_method = 'msepy'
                # _config_method == 'msepy' means we config the great mesh from a msepy mesh.

                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)

                self._global_element_type_dict = element_type_dict
                self._global_element_map_dict = element_map_dict

                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        element_type_dict, element_parameter_dict, element_map_dict
                    )
                )

            elif input_case == 'pre-defined + triangle-refining':
                # We config the great mesh through a predefined msepy mesh. Plus triangular refining
                config = MseHttMsePy_Trf_Config(self, indicator)
                element_type_dict, element_parameter_dict, element_map_dict, msepy_manifold = (
                    config(element_layout, **kwargs)
                )
                # We need to save the msepy manifold
                self._msepy_manifold = COMM.bcast(msepy_manifold, root=MASTER_RANK)
                self._config_method = 'msepy-trf'
                # _config_method == 'msepy' means we config the great mesh from a msepy mesh.

                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)

                self._global_element_type_dict = element_type_dict
                self._global_element_map_dict = element_map_dict

                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        element_type_dict, element_parameter_dict, element_map_dict
                    )
                )

            elif input_case == 'vtu_interface':
                config = MseHttVtuConfig(self, indicator)
                element_type_dict, element_parameter_dict, element_map_dict = config()
                self._config_method = 'msehtt-vtu'
                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)

                self._global_element_type_dict = indicator._global_element_type_dict
                self._global_element_map_dict = indicator._global_element_map_dict
                self._element_distribution = indicator._element_distribution

            elif input_case == 'pre-defined-msehtt-static':
                config = MseHtt_Static_PreDefined_Config(indicator)
                element_type_dict, element_parameter_dict, element_map_dict = (
                    config(element_layout, **kwargs)
                )

                self._config_method = 'msehtt-static-predefined'
                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)
                self._global_element_type_dict = element_type_dict
                self._global_element_map_dict = element_map_dict
                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        element_type_dict, element_parameter_dict, element_map_dict
                    )
                )

            else:
                raise NotImplementedError(f"msehtt-great-mesh config not implemented for input_case={input_case}")

        # --- check element map: node must be numbered (index must be int) -----------------
        for e in element_map_dict:
            e_map = element_map_dict[e]
            assert all([isinstance(_, int) for _ in e_map]), \
                f"element map must be of integers only. map of element:{e} is illegal."

        if ts:  # triangle-split -> quadrilateral or tetrahedron split -> hexahedron-combination.
            element_type_dict, element_parameter_dict, element_map_dict = self.___ts___(
                element_type_dict, element_parameter_dict, element_map_dict
            )
        else:
            pass

        element_map_dict = self._make_crack(crack_config, element_map_dict)
        self._make_elements_(element_type_dict, element_parameter_dict, element_map_dict)

    @staticmethod
    def _check_elements(element_type_dict, element_parameter_dict, element_map_dict):
        """"""
        assert len(element_type_dict) > 0, f"I need at least one element, right?"
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
            # elements_indices.sort()
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
                # elements_indices.sort()
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

    def selfcheck(self):
        r"""do a self check!"""

    def _make_crack(self, crack_config, element_map_dict):
        """Define a crack along interface of elements."""
        if crack_config is None:  # define no crack.
            return element_map_dict
        else:
            raise NotImplementedError()

    def ___ts___(self, element_type_dict, element_parameter_dict, element_map_dict):
        r""""""
        new_element_type_dict = {}
        new_element_parameter_dict = {}
        new_element_map_dict = {}
        for e_index in element_type_dict:
            e_type = element_type_dict[e_index]
            e_para = element_parameter_dict[e_index]
            e_map = element_map_dict[e_index]
            if e_type == 5:  # vtk-5: triangle
                A, B, C = e_para
                Ax, Ay = A
                Bx, By = B
                Cx, Cy = C
                D = ((Ax + Bx + Cx) / 3, (Ay + By + Cy) / 3)
                E = ((Ax + Bx) / 2, (Ay + By) / 2)
                F = ((Cx + Bx) / 2, (Cy + By) / 2)
                G = ((Ax + Cx) / 2, (Ay + Cy) / 2)
                _0_, _1_, _2_ = e_map
                _01_ = [_0_, _1_]
                _12_ = [_1_, _2_]
                _02_ = [_0_, _2_]
                _01_.sort()
                _12_.sort()
                _02_.sort()
                _01_ = tuple(_01_)
                _12_ = tuple(_12_)
                _02_ = tuple(_02_)
                _012_ = [_0_, _1_, _2_]
                _012_.sort()
                _012_ = tuple(_012_)
                element_index = str(e_index)
                e_i_0 = element_index + '-0'
                e_i_1 = element_index + '-1'
                e_i_2 = element_index + '-2'
                new_element_type_dict[e_i_0] = 9
                new_element_type_dict[e_i_1] = 9
                new_element_type_dict[e_i_2] = 9
                new_element_parameter_dict[e_i_0] = [A, E, D, G]
                new_element_parameter_dict[e_i_1] = [B, F, D, E]
                new_element_parameter_dict[e_i_2] = [C, G, D, F]
                new_element_map_dict[e_i_0] = [_0_, _01_, _012_, _02_]
                new_element_map_dict[e_i_1] = [_1_, _12_, _012_, _01_]
                new_element_map_dict[e_i_2] = [_2_, _02_, _012_, _12_]
            elif e_type == 9:  # vtk-9: quad
                #         A           J
                #           --------------------- D
                #           |         |         |
                #           |         |E        |
                #         F |-------------------|H
                #           |         |         |
                #           |         |         |
                #         B --------------------- C
                #                     G
                #
                A, B, C, D = e_para
                Ax, Ay = A
                Bx, By = B
                Cx, Cy = C
                Dx, Dy = D
                E = ((Ax + Bx + Cx + Dx) / 4, (Ay + By + Cy + Dy) / 4)
                F = ((Ax + Bx) / 2, (Ay + By) / 2)
                G = ((Bx + Cx) / 2, (By + Cy) / 2)
                H = ((Cx + Dx) / 2, (Cy + Dy) / 2)
                J = ((Dx + Ax) / 2, (Dy + Ay) / 2)
                _0_, _1_, _2_, _3_ = e_map
                _01_ = [_0_, _1_]
                _12_ = [_1_, _2_]
                _23_ = [_2_, _3_]
                _30_ = [_3_, _0_]
                _01_.sort()
                _12_.sort()
                _23_.sort()
                _30_.sort()
                _01_ = tuple(_01_)
                _12_ = tuple(_12_)
                _23_ = tuple(_23_)
                _30_ = tuple(_30_)
                _0123_ = [_0_, _1_, _2_, _3_]
                _0123_.sort()
                _0123_ = tuple(_0123_)
                element_index = str(e_index)
                e_i_0 = element_index + '-0'
                e_i_1 = element_index + '-1'
                e_i_2 = element_index + '-2'
                e_i_3 = element_index + '-3'
                new_element_type_dict[e_i_0] = 9
                new_element_type_dict[e_i_1] = 9
                new_element_type_dict[e_i_2] = 9
                new_element_type_dict[e_i_3] = 9
                new_element_parameter_dict[e_i_0] = [A, F, E, J]
                new_element_parameter_dict[e_i_1] = [F, B, G, E]
                new_element_parameter_dict[e_i_2] = [E, G, C, H]
                new_element_parameter_dict[e_i_3] = [J, E, H, D]
                new_element_map_dict[e_i_0] = [_0_, _01_, _0123_, _30_]
                new_element_map_dict[e_i_1] = [_01_, _1_, _12_, _0123_]
                new_element_map_dict[e_i_2] = [_0123_, _12_, _2_, _23_]
                new_element_map_dict[e_i_3] = [_30_, _0123_, _23_, _3_]
            elif e_type == 'orthogonal rectangle':  # orthogonal rectangle
                #         A           J
                #           --------------------- D
                #           |         |         |
                #           |         |E        |
                #         F |-------------------|H
                #           |         |         |
                #           |         |         |
                #         B --------------------- C
                #                     G
                #
                origin_x, origin_y = e_para['origin']
                delta_x, delta_y = e_para['delta']
                A = (origin_x, origin_y)
                C = (origin_x + delta_x, origin_y + delta_y)
                B = (origin_x + delta_x, origin_y)
                D = (origin_x, origin_y + delta_y)
                Ax, Ay = A
                Bx, By = B
                Cx, Cy = C
                Dx, Dy = D
                E = ((Ax + Bx + Cx + Dx) / 4, (Ay + By + Cy + Dy) / 4)
                F = ((Ax + Bx) / 2, (Ay + By) / 2)
                G = ((Bx + Cx) / 2, (By + Cy) / 2)
                H = ((Cx + Dx) / 2, (Cy + Dy) / 2)
                J = ((Dx + Ax) / 2, (Dy + Ay) / 2)
                _0_, _1_, _3_, _2_ = e_map
                _01_ = [_0_, _1_]
                _12_ = [_1_, _2_]
                _23_ = [_2_, _3_]
                _30_ = [_3_, _0_]
                _01_.sort()
                _12_.sort()
                _23_.sort()
                _30_.sort()
                _01_ = tuple(_01_)
                _12_ = tuple(_12_)
                _23_ = tuple(_23_)
                _30_ = tuple(_30_)
                _0123_ = [_0_, _1_, _2_, _3_]
                _0123_.sort()
                _0123_ = tuple(_0123_)
                element_index = str(e_index)
                e_i_0 = element_index + '-0'
                e_i_1 = element_index + '-1'
                e_i_2 = element_index + '-2'
                e_i_3 = element_index + '-3'
                new_element_type_dict[e_i_0] = 9
                new_element_type_dict[e_i_1] = 9
                new_element_type_dict[e_i_2] = 9
                new_element_type_dict[e_i_3] = 9
                new_element_parameter_dict[e_i_0] = [A, F, E, J]
                new_element_parameter_dict[e_i_1] = [F, B, G, E]
                new_element_parameter_dict[e_i_2] = [E, G, C, H]
                new_element_parameter_dict[e_i_3] = [J, E, H, D]
                new_element_map_dict[e_i_0] = [_0_, _01_, _0123_, _30_]
                new_element_map_dict[e_i_1] = [_01_, _1_, _12_, _0123_]
                new_element_map_dict[e_i_2] = [_0123_, _12_, _2_, _23_]
                new_element_map_dict[e_i_3] = [_30_, _0123_, _23_, _3_]
            else:
                raise NotImplementedError(f'triangle/tetrahedron-split does not work for etype:{e_type}.')

        NEW_element_map_dict = COMM.gather(new_element_map_dict, root=MASTER_RANK)
        NEW_element_type_dict = COMM.gather(new_element_type_dict, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            int_nodes = []
            for NEW in NEW_element_map_dict:
                for e in NEW:
                    _map = NEW[e]
                    for node in _map:
                        if isinstance(node, int):
                            int_nodes.append(node)
                        else:
                            pass
            i = max(int_nodes) + 1
            new_node_pool = {}
            element_distribution = {}
            global_element_type_dict = {}
            for rank, NEW in enumerate(NEW_element_map_dict):
                global_element_type_dict.update(NEW_element_type_dict[rank])
                element_distribution[rank] = list(NEW.keys())
                for e in NEW:
                    _map = NEW[e]
                    for node in _map:
                        if isinstance(node, tuple):
                            if node in new_node_pool:
                                pass
                            else:
                                new_node_pool[node] = i
                                i += 1
                        else:
                            pass
            del NEW_element_map_dict
            self._element_distribution = element_distribution
            self._global_element_type_dict = global_element_type_dict

        else:
            new_node_pool = {}

        new_node_pool = COMM.bcast(new_node_pool, root=MASTER_RANK)
        for e in new_element_map_dict:
            _map = new_element_map_dict[e]
            for i, node in enumerate(_map):
                if isinstance(node, tuple):
                    _map[i] = new_node_pool[node]
                else:
                    pass

        element_map_dict = COMM.gather(new_element_map_dict, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            _global_element_map_dict = {}
            for DICT in element_map_dict:
                _global_element_map_dict.update(DICT)
            self._global_element_map_dict = _global_element_map_dict

        return new_element_type_dict, new_element_parameter_dict, new_element_map_dict
