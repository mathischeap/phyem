# -*- coding: utf-8 -*-
r"""
"""
import pickle
import numpy as np
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, SIZE, COMM
from msepy.manifold.predefined.distributor import PredefinedMsePyManifoldDistributor
from msehtt.static.manifold.predefined.distributor import Predefined_Msehtt_Manifold_Distributor

from msehtt.static.mesh.great.config.vtu import ___split___

from msehtt.static.mesh.great.config.msepy_ import MseHttMsePyConfig
from msehtt.static.mesh.great.config.msepy_trf import MseHttMsePy_Trf_Config
from msehtt.static.mesh.great.config.vtu import MseHttVtuConfig
from msehtt.static.mesh.great.config.vtu import MseHttVtuInterface
from msehtt.static.mesh.great.config.msehtt_ import MseHtt_Static_PreDefined_Config
from msehtt.static.mesh.great.config.msehtt_trf import MseHtt_Static_PreDefined_Trf_Config
from msehtt.static.mesh.great.config.meshpy_ import MseHtt_API_2_MeshPy

from msehtt.static.mesh.great.elements.main import MseHttGreatMeshElements
from msehtt.static.mesh.great.visualize.main import MseHttGreatMeshVisualize
from msehtt.static.mesh.great.elements.types.distributor import MseHttGreatMeshElementDistributor

from msehtt.static.mesh.great.config.tqr import TriQuadRegions
from msehtt.static.mesh.great.config.tqr import MseHtt_TQR_config


class MseHttGreatMesh(Frozen):
    r""""""

    def __init__(self):
        r""""""
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
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__}" + super_repr

    @property
    def ___is_msehtt_great_mesh___(self):
        r"""Just a signature."""
        return True

    @property
    def elements(self):
        r"""Return all great elements instance if it exists."""
        if self._elements is None:
            raise Exception('No great elements found!')
        else:
            return self._elements

    @property
    def visualize(self):
        r""""""
        if self._visualize is None:
            self._visualize = MseHttGreatMeshVisualize(self)
        return self._visualize

    def saveto(self, filename):
        r"""Save to a `phm` file such that we can config a great mesh with this file."""
        if '.' in filename:
            assert filename.count('.') == 1 and filename[-4:] == '.phm', \
                f"filename={filename} illegal, it must be with extension `.phm`."
        else:
            filename += '.phm'

        if RANK != MASTER_RANK:
            pass
        else:
            assert self._global_element_type_dict is not None, f"The great mesh is not configured yet"
            assert self._global_element_map_dict is not None, f"The great mesh is not configured yet"
            assert self._element_distribution is not None, f"The great mesh is not configured yet"

        parameters = {}
        for e in self.elements:
            element = self.elements[e]
            etype = element.etype
            if isinstance(etype, str) and 'unique' in etype:
                raise Exception(f"This great mesh is of {etype} elements, cannot save unique elements.")
            else:
                pass

            parameters[e] = element.parameters

        parameters = COMM.gather(parameters, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            pd = {}
            for _ in parameters:
                pd.update(_)
            parameters = pd

            save_dict = {
                'etype_dict': self._global_element_type_dict,
                'map_dict': self._global_element_map_dict,
                'parameter_dict': parameters,
            }

            with open(filename, 'wb') as output:
                # noinspection PyTypeChecker
                pickle.dump(save_dict, output, pickle.HIGHEST_PROTOCOL)
            output.close()
        else:
            pass

    @staticmethod
    def ___read___(filename):
        r"""read configurations from a '.phm' file."""
        assert isinstance(filename, str) and filename[-4:] == '.phm' and filename.count('.') == 1, \
            f"filename={filename} illegal"

        with open(filename, 'rb') as inputs:
            mesh_data_dict = pickle.load(inputs)
        inputs.close()

        etype_dict = mesh_data_dict['etype_dict']
        map_dict = mesh_data_dict['map_dict']
        parameter_dict = mesh_data_dict['parameter_dict']
        return etype_dict, parameter_dict, map_dict

    def _make_elements_(self, rank_elements_type, rank_elements_parameter, rank_elements_map):
        r""""""
        assert self._elements is None, f"elements exist, do not renew them!"
        self._check_elements(rank_elements_type, rank_elements_parameter, rank_elements_map)
        if RANK == MASTER_RANK:
            self._check_global_element_map_dict()
        else:
            pass
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
        r"""Note that, periodic setting should be done before this configuration. This configuration can
        process crack and triangle/tetrahedron-split, but not periodic setting.

        Parameters
        ----------
        indicator
            case 0: reading from `.phm` file. `.phm` file can be made by calling the `saveto` method of
            a configured great mesh object.

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

            case 2: an instance of `MseHttVtuInterface` class.
                `indicator` is an instance of class `MseHttVtuInterface`. Then we will parse the mesh
                from this `MseHttVtuInterface` object which actually is an interface to vtk mesh.

            case 3: `indicator` is str and `indicator` indicates a pre-defined msehtt-manifold.
                So, `indicator`, `element_layout` and `kwargs` will be used for initialize a msehtt mesh.
                For example:
                    msehtt.config(tgm)('chaotic', element_layout=K, c=c, periodic=False)

                Furthermore, if `trf` in `kwargs`, we will do a `triangular refining` upon this msepy mesh.
                For example:
                    msehtt.config(tgm)('chaotic', element_layout=K, c=c, periodic=False, trf=1)

            case 4: 'meshpy'
                We will call the api to meshpy library using `kwargs`.

            case 5: Msehtt_PyConfig_QuadRegions
                We provide an instance of class QuadRegions showing that the domain is made of a few quad
                regions.

        element_layout :
            `element_layout`; it is not always needed for particular configuration. But usually it is needed.

        crack_config :
            In ADDITION to configuration indicating, it will make cracks in the mesh.

        ts :
            In ADDITION to configuration indicating, it will do ``triangle/tetrahedron-split``.

        kwargs :
            Other kwargs to be passed to particular configuration.

        Returns
        -------

        """
        assert self._config_method == '', f"I must be not-configured yet!"

        # --------------------------------------------------------------------------------------------------------
        if isinstance(indicator, str) and '.phm' in indicator:
            # we read from a `.phm` file.
            input_case = 'reading'

        elif isinstance(indicator, str) and indicator in PredefinedMsePyManifoldDistributor._predefined_manifolds():
            # case 1
            if 'trf' in kwargs:  # we do a triangular refining upon this msepy mesh
                input_case = 'pre-defined + triangle-refining'
            else:
                input_case = 'pre-defined'

        elif indicator.__class__ is MseHttVtuInterface:
            # case 2
            input_case = 'vtu_interface'

        elif isinstance(indicator, str) and indicator in Predefined_Msehtt_Manifold_Distributor.defined_manifolds():
            # case 3
            if 'trf' in kwargs:
                input_case = 'pre-defined-msehtt-static + triangle-refining'
            else:
                input_case = 'pre-defined-msehtt-static'

        elif isinstance(indicator, str) and indicator == 'meshpy':
            # case 4
            input_case = 'meshpy'

            kwargs_for_call_method = {}
            keys = list(kwargs.keys())
            for key in keys:
                if key == 'periodic_setting':
                    raise Exception(f"msehtt meshpy configuration takes no `periodic_setting`.")
                else:
                    pass

                if key == 'max_volume':
                    kwargs_for_call_method[key] = kwargs[key]
                    del kwargs[key]
                else:
                    pass

        elif indicator.__class__ is TriQuadRegions:  # quad regions as the indicator
            # case 5
            input_case = 'tqr'

        elif isinstance(indicator, (tuple, list)) and isinstance(indicator[0], str) and indicator[0] == 'tqr':
            # case 6
            # Instead of giving a TriQuadRegions instance, we can make the TriQuadRegions instance here.
            # If we receive a list or tuple whose first entry is 'tqr', then we make a TriQuadRegions instance
            # using all other entries in the list or tuple.
            indicator = TriQuadRegions(*indicator[1:])
            input_case = 'tqr'

        else:
            if isinstance(indicator, dict):  # a standard input format for the indicator.

                assert 'indicator' in indicator, f"key 'indicator' must be in indicator dict to guide the type."
                assert 'args' in indicator, \
                    f"key 'args' must be in the indicator dict carrying the mandatory arguments."

                if 'kwargs' in indicator:
                    assert isinstance(indicator['kwargs'], dict), f"Providing kwargs? put them in a dict."
                    kwargs = indicator['kwargs']
                    assert len(indicator) == 3, (f"standard config dict indicator must maximum only have three keys:"
                                                 f" 'indicator', 'args', 'kwargs'.")
                else:
                    assert len(indicator) == 2, (f"standard config dict indicator must only have two keys:"
                                                 f" 'indicator', 'args' when there is no kwargs.")
                    kwargs = {}

                if indicator['indicator'] == 'tqr':
                    # provide a TriQuadRegions configuration through a dict indicator.
                    indicator = TriQuadRegions(*indicator['args'], **kwargs)
                    input_case = 'tqr'

                else:
                    raise NotImplementedError(f"standard config indicator ={indicator['indicator']} "
                                              f"not understandable.")

            else:
                raise NotImplementedError(f"cannot parse the config method!")

        # =======================================================================================================

        if RANK != MASTER_RANK:
            if input_case == 'reading':
                self._config_method = 'reading'
                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        None, None, None
                    )
                )
            elif input_case == 'pre-defined':
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

            elif input_case == 'pre-defined-msehtt-static + triangle-refining':
                self._config_method = 'msehtt-static-predefined-trf'
                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        None, None, None
                    )
                )

            elif input_case == 'meshpy':
                self._config_method = 'meshpy'
                vif = MseHttVtuInterface(
                    {}, {}, {},
                    redistribute=True
                )

                config = MseHttVtuConfig(self, vif)
                element_type_dict, element_parameter_dict, element_map_dict = config()

                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)

            elif input_case == 'tqr':
                self._config_method = 'msehtt-static-tqr'
                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        None, None, None
                    )
                )

            else:
                raise NotImplementedError()

        else:
            if input_case == 'reading':
                self._config_method = 'reading'
                element_type_dict, element_parameter_dict, element_map_dict = self.___read___(indicator)
                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)
                self._global_element_type_dict = element_type_dict
                self._global_element_map_dict = element_map_dict
                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        element_type_dict, element_parameter_dict, element_map_dict
                    )
                )
            elif input_case == 'pre-defined':
                # We config the great mesh through a predefined msepy mesh.
                config = MseHttMsePyConfig(self, indicator)
                element_type_dict, element_parameter_dict, element_map_dict, msepy_manifold, _ = (
                    config(element_layout, **kwargs)
                )
                # We need to save the msepy manifold
                self._msepy_manifold = COMM.bcast(msepy_manifold, root=MASTER_RANK)
                self._config_method = 'msepy'

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

            elif input_case == 'pre-defined-msehtt-static + triangle-refining':
                # We config the great mesh through a predefined msehtt mesh + triangular refining
                config = MseHtt_Static_PreDefined_Trf_Config(indicator)
                element_type_dict, element_parameter_dict, element_map_dict = (
                    config(element_layout, **kwargs)
                )
                self._config_method = 'msehtt-static-predefined-trf'
                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)
                self._global_element_type_dict = element_type_dict
                self._global_element_map_dict = element_map_dict
                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        element_type_dict, element_parameter_dict, element_map_dict
                    )
                )

            elif input_case == 'meshpy':
                MeshPy_interface_config = MseHtt_API_2_MeshPy(**kwargs)
                # noinspection PyUnboundLocalVariable
                vif = MeshPy_interface_config(**kwargs_for_call_method)
                config = MseHttVtuConfig(self, vif)

                self._config_method = 'meshpy'
                element_type_dict, element_parameter_dict, element_map_dict = config()

                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)

                self._global_element_type_dict = vif._global_element_type_dict
                self._global_element_map_dict = vif._global_element_map_dict
                self._element_distribution = vif._element_distribution

            elif input_case == 'tqr':
                config = MseHtt_TQR_config(indicator)
                element_type_dict, element_parameter_dict, element_map_dict = config(element_layout)

                self._config_method = 'msehtt-static-tqr'
                self._check_elements(element_type_dict, element_parameter_dict, element_map_dict)

                self._global_element_type_dict = element_type_dict
                self._global_element_map_dict = element_map_dict

                element_type_dict, element_parameter_dict, element_map_dict = (
                    self._distribute_elements_to_ranks(
                        element_type_dict, element_parameter_dict, element_map_dict
                    )
                )

            else:
                raise NotImplementedError(
                    f"msehtt-great-mesh config not implemented for input_case={input_case}")

        # --- check element map: node must be numbered (index must be int) -----------------
        for e in element_map_dict:
            e_map = element_map_dict[e]
            assert all([isinstance(_, int) for _ in e_map]), \
                f"element map must be of integers only. map of element:{e} is illegal."

        # ----------- do triangle/tetrahedron-split----------------------------------------------
        if ts is False:
            ts = 0
        elif ts is True:
            ts = 1
        else:
            assert isinstance(ts, int) and ts >= 0, \
                f"ts={ts} wrong, it must be False, True or non-negative integer."

        element_type_dict, element_parameter_dict, element_map_dict = self._parse_ts_(
            ts,
            element_type_dict, element_parameter_dict, element_map_dict
        )
        # =======================================================================================
        element_map_dict = self._make_crack(crack_config, element_map_dict)
        self._make_elements_(element_type_dict, element_parameter_dict, element_map_dict)

    @staticmethod
    def _check_elements(element_type_dict, element_parameter_dict, element_map_dict):
        r""""""
        assert len(element_type_dict) > 0, (f"I need at least one element, right? Likely that there are "
                                            f"more ranks than elements. Reduce SIZE.")
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
        r""""""
        if SIZE == 1:
            elements_indices = list(all_element_type_dict.keys())
            # elements_indices.sort()
            self._element_distribution = {MASTER_RANK: elements_indices}
            return all_element_type_dict, all_element_parameter_dict, all_element_map_dict
        else:
            pass

        if RANK == MASTER_RANK:
            element_distribution = {}  # only in the master rank

            # ----------- Different element distribution methods -------------------------------------
            if method == 'naive':
                # ------ most trivial method ---------------------------------------------------------
                elements_indices = list(all_element_type_dict.keys())
                # elements_indices.sort()
                num_total_elements = len(elements_indices)

                if num_total_elements < SIZE:
                    raise r"number of elements is lower than SIZE, reduce SIZE."
                elif num_total_elements == SIZE:
                    for rank, e in enumerate(all_element_type_dict):
                        element_distribution[rank] = [e]
                else:
                    if self._msepy_manifold is not None:
                        if num_total_elements < 4 * SIZE:
                            rank_indices = np.array_split(range(num_total_elements), SIZE)
                            elements_indices = list(self._global_element_type_dict.keys())
                            for rank, indices in enumerate(rank_indices):
                                low, upper = min(indices), max(indices) + 1
                                element_distribution[rank] = elements_indices[low:upper]
                        else:
                            num_piles = 3 * (SIZE - 1) + 1  # master rank takes 1 pile, other ranks take 3 piles each.
                            num_elements_each_pile = num_total_elements / num_piles  # OK to be decimal
                            start = 0
                            for rank in range(SIZE):
                                take_num_piles = 1 if rank == MASTER_RANK else 3
                                end = int(start + take_num_piles * num_elements_each_pile) + 1
                                element_distribution[rank] = elements_indices[start:end]
                                start = end

                    else:
                        element_distribution = ___Naive_element_distribution___(
                            all_element_type_dict, all_element_parameter_dict, all_element_map_dict
                        )

            else:
                raise NotImplementedError(
                    f"Please implement better element distributor late. It helps a lot.")

            elements_type = [{} for _ in range(SIZE)]
            elements_parameter = [{} for _ in range(SIZE)]
            elements_map = [{} for _ in range(SIZE)]
            for rank in range(SIZE):
                for i in element_distribution[rank]:
                    elements_type[rank][i] = all_element_type_dict[i]
                    elements_parameter[rank][i] = all_element_parameter_dict[i]
                    elements_map[rank][i] = all_element_map_dict[i]

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
        r"""Define a crack along interface of elements."""
        if crack_config is None:  # define no crack.
            return element_map_dict
        else:
            raise NotImplementedError()

    def _parse_ts_(self, ts, element_type_dict, element_parameter_dict, element_map_dict):
        r""""""
        if ts == 0:
            return element_type_dict, element_parameter_dict, element_map_dict
        else:
            element_type_dict, element_parameter_dict, element_map_dict = self.___ts___(
                element_type_dict, element_parameter_dict, element_map_dict
            )
            new_ts = ts - 1
            return self._parse_ts_(
                new_ts, element_type_dict, element_parameter_dict, element_map_dict
            )

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
                _012_ = (_0_, _1_, _2_)
                element_index = str(e_index)
                e_i_0 = element_index + ':5>0'
                e_i_1 = element_index + ':5>1'
                e_i_2 = element_index + ':5>2'
                new_element_type_dict[e_i_0] = 9
                new_element_type_dict[e_i_1] = 9
                new_element_type_dict[e_i_2] = 9
                new_element_parameter_dict[e_i_0] = [A, E, D, G]
                new_element_parameter_dict[e_i_1] = [B, F, D, E]
                new_element_parameter_dict[e_i_2] = [C, G, D, F]
                new_element_map_dict[e_i_0] = [_0_, _01_, _012_, _02_]
                new_element_map_dict[e_i_1] = [_1_, _12_, _012_, _01_]
                new_element_map_dict[e_i_2] = [_2_, _02_, _012_, _12_]

            elif e_type == 'unique curvilinear triangle':
                raise Exception(
                    f"element of etype:`unique curvilinear triangle` cannot "
                    f"perform triangle/tetrahedron-split."
                )

            elif e_type == 9:  # vtk-9: quad
                #         A(node0)     J      (node3)
                #           --------------------- D
                #           |         |         |
                #           |         |E        |
                #         F |-------------------|H
                #           |         |         |
                #           |         |         |
                #         B --------------------- C
                #        (node1)      G         (node2)
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
                _0123_ = (_0_, _1_, _2_, _3_)
                element_index = str(e_index)
                e_i_0 = element_index + ':9>0'
                e_i_1 = element_index + ':9>1'
                e_i_2 = element_index + ':9>2'
                e_i_3 = element_index + ':9>3'
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
                #         A(node0)     J       (node2)
                #           _____________________ D
                #           |         |         |
                #           |         |         |
                #           |         |E        |
                #         F |-------------------|H
                #           |         |         |
                #           |         |         |
                #         B |_________|_________| C
                #        (node1)      G         (node3)
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
                _0123_ = (_0_, _1_, _2_, _3_)
                element_index = str(e_index)
                e_i_0 = element_index + ':or>0'
                e_i_1 = element_index + ':or>1'
                e_i_2 = element_index + ':or>2'
                e_i_3 = element_index + ':or>3'
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

            elif e_type == 'unique curvilinear quad':  # unique curvilinear quad
                #         A(node0)    J            (node3)
                #           _____________________ D
                #           |         |         |
                #           |         |         |
                #           |         |E        |
                #         F |---------|---------|H
                #           |         |         |
                #           |         |         |
                #         B |_________|_________| C
                #        (node1)      G          (node2)
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
                _0123_ = (_0_, _1_, _2_, _3_)
                element_index = str(e_index)
                e_i_0 = element_index + ':UCQ>0'
                e_i_1 = element_index + ':UCQ>1'
                e_i_2 = element_index + ':UCQ>2'
                e_i_3 = element_index + ':UCQ>3'
                new_element_type_dict[e_i_0] = e_type
                new_element_type_dict[e_i_1] = e_type
                new_element_type_dict[e_i_2] = e_type
                new_element_type_dict[e_i_3] = e_type
                hUCQ = ___ts_helper_UCQ___(e_para)
                new_element_parameter_dict[e_i_0] = \
                    {'mapping': hUCQ.mapping_NW, 'Jacobian_matrix': hUCQ.JM_NW}
                new_element_parameter_dict[e_i_1] = \
                    {'mapping': hUCQ.mapping_SW, 'Jacobian_matrix': hUCQ.JM_SW}
                new_element_parameter_dict[e_i_2] = \
                    {'mapping': hUCQ.mapping_SE, 'Jacobian_matrix': hUCQ.JM_SE}
                new_element_parameter_dict[e_i_3] = \
                    {'mapping': hUCQ.mapping_NE, 'Jacobian_matrix': hUCQ.JM_NE}
                new_element_map_dict[e_i_0] = [_0_, _01_, _0123_, _30_]
                new_element_map_dict[e_i_1] = [_01_, _1_, _12_, _0123_]
                new_element_map_dict[e_i_2] = [_0123_, _12_, _2_, _23_]
                new_element_map_dict[e_i_3] = [_30_, _0123_, _23_, _3_]

            else:
                raise NotImplementedError(
                    f'triangle/tetrahedron-split does not work for etype:{e_type} yet.')

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

    def _check_global_element_map_dict(self):
        r""""""
        eMap = self._global_element_map_dict
        element_classes = MseHttGreatMeshElementDistributor.implemented_element_types()

        checked_etype_face_setting = []

        element_face_map_pool = {}
        for e in eMap:
            etype = self._global_element_type_dict[e]
            element_class = element_classes[etype]
            face_setting = element_class.face_setting()
            if etype in checked_etype_face_setting:
                pass
            else:
                face_ids = list(face_setting.keys())
                face_ids.sort()
                assert face_ids == [_ for _ in range(len(face_ids))], \
                    f"face id must be like 0, 1, 2, ..."
                checked_etype_face_setting.append(etype)

            element_map = eMap[e]
            for face_id in face_setting:
                face_nodes = face_setting[face_id]
                face_nodes_numbering = [element_map[_] for _ in face_nodes]
                face_nodes_numbering.sort()
                face_nodes_numbering = tuple(face_nodes_numbering)

                if face_nodes_numbering in element_face_map_pool:
                    # noinspection PyUnresolvedReferences
                    element_face_map_pool[face_nodes_numbering].append(
                        (etype, e, face_id)
                    )
                else:
                    element_face_map_pool[face_nodes_numbering] = [(etype, e, face_id)]

        for face in element_face_map_pool:
            positions = element_face_map_pool[face]
            if len(positions) <= 2:
                pass
            else:
                raise Exception(
                    f"{len(positions)} positions:\n<" +
                    '>\n<'.join(
                        [f'face {_[2]} on element indexed {_[1]} of etype:{_[0]} element'
                         for _ in positions]
                    )
                    + f'>\n share same nodes: numbered {face}, which is impossible.'
                )


def ___Naive_element_distribution___(all_element_type_dict, all_element_parameter_dict, all_element_map_dict):
    r"""

    Parameters
    ----------
    all_element_type_dict
    all_element_parameter_dict
    all_element_map_dict

    Returns
    -------

    """
    element_centers = {}

    element_classes = MseHttGreatMeshElementDistributor.implemented_element_types()
    for e in all_element_type_dict:
        etype = all_element_type_dict[e]
        parameters = all_element_parameter_dict[e]
        ec = element_classes[etype]
        element_centers[e] = ec._find_element_center_coo(parameters)

    num_total_elements = len(all_element_map_dict)
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
    remaining_elements = list(all_element_map_dict.keys())

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

    empty_ranks = list()
    for rank in element_distribution:
        if len(element_distribution[rank]) == 0:
            empty_ranks.append(rank)
        else:
            pass

    if len(empty_ranks) > 0:  # there are empty ranks, try to resolve them.
        for er in empty_ranks:
            for rank in range(SIZE):
                if len(element_distribution[rank]) > 1:
                    element_distribution[er] = [element_distribution[rank].pop(), ]
                    break
                else:
                    pass
    else:
        pass

    NUM_TOTAL_ELEMENTS = 0
    for rank in element_distribution:
        NUM_TOTAL_ELEMENTS += len(element_distribution[rank])
    assert NUM_TOTAL_ELEMENTS == num_total_elements, f"must be!"

    for rank in element_distribution:
        assert len(element_distribution[rank]) != 0, f"must no empty element rank."

    return element_distribution


def ___ts_helper_UCQ___(e_para):
    r""""""
    return ___TSH_UCQ___(e_para)


class ___TSH_UCQ___(Frozen):
    r""""""
    def __init__(self, base_element_parameters):
        r""""""
        self._bmp = base_element_parameters['mapping']
        self._bJm = base_element_parameters['Jacobian_matrix']
        self._freeze()

    def mapping_NW(self, xi, et):
        r""""""
        r = (xi + 1) / 2 - 1
        s = (et + 1) / 2 - 1
        return self._bmp(r, s)

    def mapping_SW(self, xi, et):
        r""""""
        r = (xi + 1) / 2
        s = (et + 1) / 2 - 1
        return self._bmp(r, s)

    def mapping_SE(self, xi, et):
        r""""""
        r = (xi + 1) / 2
        s = (et + 1) / 2
        return self._bmp(r, s)

    def mapping_NE(self, xi, et):
        r""""""
        r = (xi + 1) / 2 - 1
        s = (et + 1) / 2
        return self._bmp(r, s)

    def JM_NW(self, xi, et):
        r""""""
        r = (xi + 1) / 2 - 1
        s = (et + 1) / 2 - 1
        dx, dy = self._bJm(r, s)
        dx_dr, dx_ds = dx
        dy_dr, dy_ds = dy
        dx_dxi = dx_dr * 0.5
        dx_det = dx_ds * 0.5
        dy_dxi = dy_dr * 0.5
        dy_det = dy_ds * 0.5
        return (
            [dx_dxi, dx_det],
            [dy_dxi, dy_det]
        )

    def JM_SW(self, xi, et):
        r""""""
        r = (xi + 1) / 2
        s = (et + 1) / 2 - 1
        dx, dy = self._bJm(r, s)
        dx_dr, dx_ds = dx
        dy_dr, dy_ds = dy
        dx_dxi = dx_dr * 0.5
        dx_det = dx_ds * 0.5
        dy_dxi = dy_dr * 0.5
        dy_det = dy_ds * 0.5
        return (
            [dx_dxi, dx_det],
            [dy_dxi, dy_det]
        )

    def JM_SE(self, xi, et):
        r""""""
        r = (xi + 1) / 2
        s = (et + 1) / 2
        dx, dy = self._bJm(r, s)
        dx_dr, dx_ds = dx
        dy_dr, dy_ds = dy
        dx_dxi = dx_dr * 0.5
        dx_det = dx_ds * 0.5
        dy_dxi = dy_dr * 0.5
        dy_det = dy_ds * 0.5
        return (
            [dx_dxi, dx_det],
            [dy_dxi, dy_det]
        )

    def JM_NE(self, xi, et):
        r""""""
        r = (xi + 1) / 2 - 1
        s = (et + 1) / 2
        dx, dy = self._bJm(r, s)
        dx_dr, dx_ds = dx
        dy_dr, dy_ds = dy
        dx_dxi = dx_dr * 0.5
        dx_det = dx_ds * 0.5
        dy_dxi = dy_dr * 0.5
        dy_det = dy_ds * 0.5
        return (
            [dx_dxi, dx_det],
            [dy_dxi, dy_det]
        )
