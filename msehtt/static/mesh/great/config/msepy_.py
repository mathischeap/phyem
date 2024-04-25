# -*- coding: utf-8 -*-
"""We config the msehtt great mesh as a msepy mesh.
"""
import inspect
import numpy as np
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK
from src.config import get_embedding_space_dim
from msepy.manifold.predefined.distributor import PredefinedMsePyManifoldDistributor
from msepy.manifold.main import MsePyManifold
from src.manifold import Manifold
from src.mesh import Mesh
from msepy.mesh.main import MsePyMesh
from msepy.mesh.elements.main import MsePyMeshElements
from msepy.manifold.regions.standard.ct import UniqueRegionException


class MseHttMsePyConfig(Frozen):
    """"""

    def __init__(self, tgm, domain_indicator):
        """"""
        assert RANK == MASTER_RANK, f"Do msepy only in the master rank."
        self._tgm = tgm
        distributor = PredefinedMsePyManifoldDistributor()
        self._msepy_domain_class_or_function = distributor(domain_indicator)
        self._freeze()

    def __call__(self, element_layout, **kwargs):
        """"""
        class_or_function = self._msepy_domain_class_or_function

        m = get_embedding_space_dim()

        the_great_abstract_manifold = Manifold(m)

        msepy_manifold = MsePyManifold(the_great_abstract_manifold)

        if inspect.isfunction(class_or_function):
            parameters = class_or_function(msepy_manifold, **kwargs)
        elif inspect.isclass(class_or_function):  # we get a class.
            parameters = class_or_function(msepy_manifold, **kwargs)()
        else:
            raise Exception()

        region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict, default_element_layout = parameters

        msepy_manifold._parse_regions_from_region_map(
            0,  # region map type 0 for regular region
            region_map,
            mapping_dict,
            Jacobian_matrix_dict,
            mtype_dict,
            check_periodic=False
        )
        msepy_manifold._default_element_layout_maker = default_element_layout

        the_great_abstract_mesh = Mesh(the_great_abstract_manifold)
        msepy_mesh = MsePyMesh(the_great_abstract_mesh)

        msepy_mesh._manifold = msepy_manifold
        msepy_mesh._elements = MsePyMeshElements(msepy_mesh)  # initialize the mesh elements.
        layouts = msepy_mesh._parse_elements_from_element_layout(element_layout)

        assert msepy_mesh.elements._index_mapping is not None, \
            f"we should have set elements._index_mapping"
        assert msepy_mesh.elements._map is not None, \
            f"we should have set elements._map"

        element_map_array = self._generate_element_map_dict(msepy_mesh)

        element_type_dict = {}
        element_parameter_dict = {}
        element_map_dict = {}

        elements = msepy_mesh.elements
        regions = msepy_manifold.regions

        if elements._element_mtype_dict is None:
            element_mtype_dict = dict()

            for i in regions:
                layout_of_region = layouts[i]
                element_numbering_of_region = elements._numbering[i]
                region = regions[i]
                ctm = region._ct.mtype
                try:
                    r_emd = ctm._distribute_to_element(layout_of_region, element_numbering_of_region)
                    for key in r_emd:
                        if key in element_mtype_dict:
                            element_mtype_dict[key].extend(r_emd[key])
                        else:
                            element_mtype_dict[key] = r_emd[key]
                except UniqueRegionException:
                    pass

            for key in element_mtype_dict:
                element_mtype_dict[key].sort()

        else:
            element_mtype_dict = elements._element_mtype_dict

        for i in elements:
            element = elements[i]
            metric_signature = None
            for ms in element_mtype_dict:
                if i in element_mtype_dict[ms]:
                    metric_signature = ms
                    break
                else:
                    pass
            if metric_signature is None:
                # unique element
                element_type_dict[i] = 'unique msepy curvilinear quadrilateral'
                element_parameter_dict[i] = {
                    'region': element._region,
                    'origin': element.ct._origin,
                    'delta': element.ct._delta,
                }
            elif metric_signature[:7] == 'Linear:' and m == 2:
                # orthogonal rectangle element
                rct = regions[element._region]._ct
                _origin = element.ct._origin
                _delta = element.ct._delta
                end = (_origin[0] + _delta[0], _origin[1] + _delta[1])
                origin = rct.mapping(*_origin)
                end = rct.mapping(*end)
                delta = (end[0] - origin[0], end[1] - origin[1])
                element_type_dict[i] = 'orthogonal rectangle'
                element_parameter_dict[i] = {
                    'origin': origin,
                    'delta': delta,
                }
            else:
                raise NotImplementedError()

            element_map_dict[i] = list(element_map_array[i])

        return element_type_dict, element_parameter_dict, element_map_dict, msepy_manifold

    def _generate_element_map_dict(self, msepy_mesh):
        """"""
        msepy_element_map = msepy_mesh.elements._map
        num_elements, m = msepy_element_map.shape
        if m == 4:
            return self._generate_element_map_dict_2d(msepy_mesh)
        elif m == 8:
            return self._generate_element_map_dict_3d(msepy_mesh)
        else:
            raise Exception()

    @staticmethod
    def _generate_element_map_dict_2d(msepy_mesh):

        """A very old scheme, ugly but works."""
        # the idea of numbering 0-form on 2-manifold is the following
        # 1) we go through all elements
        # 2) we check its UL corner and number the dof
        # 3) We check its L edge and number the dofs
        # 4) we check its DL corner and number the dof
        # 5) We check its U edge and number the dofs
        # 6) we number internal dofs
        # 7) We check its D edge and number the dofs
        # 8) we check its UR corner and number the dof
        # 9) we check its R edge and number the dofs
        # 10) we check its DR corner and number the dof
        p = (1, 1)
        mp = msepy_mesh.elements.map
        gm = - np.ones((msepy_mesh.elements._num, p[0] + 1, p[1] + 1), dtype=int)
        _dict_ = {'U': (0, 0), 'D': (0, -1), 'L': (1, 0), 'R': (1, -1)}
        _cd_ = {'UL': (0, 0), 'DL': (-1, 0), 'UR': (0, -1), 'DR': (-1, -1)}
        _n2id_ = {'UL': 0, 'DL': 1, 'UR': 2, 'DR': 3}
        edge_pair = {0: 1, 1: 0, 2: 3, 3: 2}
        ind = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
        num_dof_dict = {'L': p[0] - 1, 'U': p[1] - 1, 'D': p[1] - 1, 'R': p[0] - 1}
        ngc = msepy_mesh.topology.corner_numbering
        gac = msepy_mesh.topology.corner_attachment

        current_num = 0
        for i in range(msepy_mesh.elements._num):  # do step 1).
            for number_where in ('UL', 'L', 'DL', 'U', 'I', 'D', 'UR', 'R', 'DR'):
                # ________ element corners ________________________________________
                if number_where in ('UL', 'DL', 'UR', 'DR'):  # change tuple to change sequence
                    index_x, index_y = _cd_[number_where]
                    if gm[i, index_x, index_y] != -1:  # this corner numbered
                        pass  # do nothing, as it is numbered
                    else:  # not numbered, we number it.
                        attachment = gac[ngc[i][_n2id_[number_where]]]
                        for numbering_element_numbering_corner in attachment:
                            numbering_element, numbering_corner = \
                                numbering_element_numbering_corner.split('-')
                            numbering_element = int(numbering_element)
                            index_x, index_y = _cd_[numbering_corner]
                            gm[numbering_element, index_x, index_y] = current_num
                        current_num += 1
                # _______ element edges (except corners) __________________________
                elif number_where in ('L', 'U', 'D', 'R'):  # change tuple to change sequence
                    numbering_element = i
                    numbering_edge_id = ind[number_where]
                    attached_2_numbering_edge = mp[numbering_element][numbering_edge_id]
                    # _____ element edge on domain boundary________________________
                    if attached_2_numbering_edge == -1:
                        # the numbering edge is on domain boundary
                        axis, start_end = _dict_[number_where]
                        if axis == 0:
                            gm[i, start_end, 1:-1] = np.arange(
                                current_num, current_num + num_dof_dict[number_where])
                        elif axis == 1:
                            gm[i, 1:-1, start_end] = np.arange(
                                current_num, current_num + num_dof_dict[number_where])
                        else:
                            raise Exception()
                        current_num += num_dof_dict[number_where]
                    # ___ element edge attached to another mesh element____________
                    else:
                        attached_element = attached_2_numbering_edge
                        attached_edge_id = edge_pair[numbering_edge_id]
                        assert edge_pair[attached_edge_id] == numbering_edge_id
                        # __ another mesh element is not numbered yet _____________
                        if attached_element > numbering_element:
                            # the attached_edge can not be numbered, we number numbering_edge
                            axis, start_end = _dict_[number_where]
                            if axis == 0:
                                gm[i, start_end, 1:-1] = np.arange(
                                    current_num, current_num + num_dof_dict[number_where])
                            elif axis == 1:
                                gm[i, 1:-1, start_end] = np.arange(
                                    current_num, current_num + num_dof_dict[number_where])
                            else:
                                raise Exception()
                            current_num += num_dof_dict[number_where]
                        # __another mesh element is numbered_______________________
                        else:  # we take the numbering from attached_edge
                            axis, start_end = _dict_[number_where]
                            attached_se = {0: -1, -1: 0}[start_end]
                            if axis == 0:
                                gm[i, start_end, 1:-1] = gm[attached_element][attached_se, 1:-1]
                            elif axis == 1:
                                gm[i, 1:-1, start_end] = gm[attached_element][1:-1, attached_se]
                            else:
                                raise Exception()
                # _____ internal corners __________________________________________
                elif number_where == 'I':
                    gm[i, 1:-1, 1:-1] = np.arange(
                        current_num, current_num+(p[0]-1)*(p[1]-1)).reshape((p[0]-1, p[1]-1), order='F')
                    current_num += (p[0]-1)*(p[1]-1)
                # ____ ELSE _____________________________________________
                else:
                    raise Exception(f"cannot reach here!")

        gm = np.array([gm[j].ravel('F') for j in range(msepy_mesh.elements._num)])
        return gm

    def _generate_element_map_dict_3d(self, msepy_mesh):
        """A very old scheme, ugly but works."""
        mesh = msepy_mesh
        p = (1, 1, 1)
        corner_position = mesh.topology.corner_attachment
        edge_position = mesh.topology.edge_attachment
        side_position = mesh.topology.side_attachment
        corner_gn = mesh.topology.corner_numbering
        edge_gn = mesh.topology.edge_numbering
        side_gn = mesh.topology.side_numbering
        gn = - np.ones((mesh.elements._num, p[0]+1, p[1]+1, p[2]+1), dtype=int)
        current_num = 0
        corner_index_dict = {
            'NWB': 0, 'SWB': 1, 'NEB': 2, 'SEB': 3, 'NWF': 4, 'SWF': 5, "NEF": 6, 'SEF': 7}
        edge_index_dict = {
            'WB': 0, 'EB': 1, 'WF': 2, 'EF': 3, 'NB': 4, 'SB': 5,
            'NF': 6, 'SF': 7, 'NW': 8, 'SW': 9, 'NE': 10, 'SE': 11}
        side_index_dict = {'N': 0, 'S': 1, 'W': 2, 'E': 3, 'B': 4, 'F': 5}
        for k in range(mesh.elements._num):
            # we go through all positions of each element.
            for position in ('NWB', 'WB', 'SWB', 'NB', 'B', 'SB', 'NEB', 'EB',
                             'SEB', 'NW', 'W', 'SW', 'NWF', 'WF', 'SWF', 'N', 'I',
                             'S', 'NE', 'E', 'SE', 'NF', 'F', 'SF',
                             'NEF', 'EF', 'SEF'):
                if position in ('NWB', 'SWB', 'NEB', 'SEB', 'NWF', 'SWF', 'NEF', 'SEF'):
                    triple_trace_element_no = corner_gn[k, corner_index_dict[position]]
                    triple_trace_element_position = corner_position[triple_trace_element_no]
                    gn, current_num = self.___number_triple_trace_element_position___(
                        gn, current_num, triple_trace_element_position
                    )
                elif position in ('WB', 'NB', 'SB', 'EB',
                                  'NW', 'SW', 'WF', 'NE',
                                  'SE', 'NF', 'SF', 'EF'):
                    dump_element_no = edge_gn[k, edge_index_dict[position]]
                    dump_element_position = edge_position[dump_element_no]
                    gn, current_num = self.___number_dump_element_position___(
                        p, gn, current_num, dump_element_position
                    )
                elif position in ('N', 'S', 'W', 'E', 'B', 'F'):
                    trace_element_no = side_gn[k, side_index_dict[position]]
                    trace_element_position = side_position[trace_element_no]
                    gn, current_num = self.___number_trace_element_position___(
                        p, gn, current_num, trace_element_position
                    )
                elif position == 'I':
                    PPP = (p[0]-1) * (p[1]-1) * (p[2]-1)
                    if PPP > 0:
                        gn[k, 1:-1, 1:-1, 1:-1] = np.arange(
                            current_num, current_num+PPP).reshape(
                            (p[0]-1, p[1]-1, p[2]-1), order='F')
                        current_num += PPP
                else:
                    raise Exception()

        gn = np.array([gn[j].ravel('F') for j in range(mesh.elements._num)])
        return gn

    @staticmethod
    def ___number_triple_trace_element_position___(gn, current_num, triple_trace_element_position):
        """"""
        numbered = None
        for position_tt in triple_trace_element_position:
            if position_tt[0] == '<':
                pass
            else:
                mesh_element_no, position = position_tt.split('-')
                mesh_element_no = int(mesh_element_no)
                if position == 'NWB':
                    if gn[mesh_element_no, 0, 0, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 0, 0] = current_num
                elif position == 'SWB':
                    if gn[mesh_element_no, -1, 0, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 0, 0] = current_num
                elif position == 'NEB':
                    if gn[mesh_element_no, 0, -1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, -1, 0] = current_num
                elif position == 'SEB':
                    if gn[mesh_element_no, -1, -1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, -1, 0] = current_num
                elif position == 'NWF':
                    if gn[mesh_element_no, 0, 0, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 0, -1] = current_num
                elif position == 'SWF':
                    if gn[mesh_element_no, -1, 0, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 0, -1] = current_num
                elif position == 'NEF':
                    if gn[mesh_element_no, 0, -1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, -1, -1] = current_num
                elif position == 'SEF':
                    if gn[mesh_element_no, -1, -1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, -1, -1] = current_num
                else:
                    raise Exception()
        if numbered is None:
            current_num += 1
        return gn, current_num

    @staticmethod
    def ___number_dump_element_position___(p, gn, current_num, dump_element_position):
        """ """
        pxyz = p
        numbered = None
        p = None
        for position_d in dump_element_position:
            mesh_element_no, position = position_d.split('-')
            mesh_element_no = int(mesh_element_no)
            # ___ dz edges _________________________________________________________
            if position == 'NW':
                if p is None:
                    p = pxyz[2]
                else:
                    assert p == pxyz[2]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 0, 0, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 0, 1:-1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'SW':
                if p is None:
                    p = pxyz[2]
                else:
                    assert p == pxyz[2]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, -1, 0, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 0, 1:-1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'SE':
                if p is None:
                    p = pxyz[2]
                else:
                    assert p == pxyz[2]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, -1, -1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, -1, 1:-1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'NE':
                if p is None:
                    p = pxyz[2]
                else:
                    assert p == pxyz[2]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 0, -1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, -1, 1:-1] = np.arange(
                            current_num, current_num+p-1)
            # ___ dy edges _________________________________________________________
            elif position == 'NB':
                if p is None:
                    p = pxyz[1]
                else:
                    assert p == pxyz[1]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 0, 1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 1:-1, 0] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'SB':
                if p is None:
                    p = pxyz[1]
                else:
                    assert p == pxyz[1]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, -1, 1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 1:-1, 0] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'SF':
                if p is None:
                    p = pxyz[1]
                else:
                    assert p == pxyz[1]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, -1, 1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 1:-1, -1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'NF':
                if p is None:
                    p = pxyz[1]
                else:
                    assert p == pxyz[1]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 0, 1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 1:-1, -1] = np.arange(
                            current_num, current_num+p-1)
            # ___ dx edges _________________________________________________________
            elif position == 'WB':
                if p is None:
                    p = pxyz[0]
                else:
                    assert p == pxyz[0]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 1, 0, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 0, 0] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'EB':
                if p is None:
                    p = pxyz[0]
                else:
                    assert p == pxyz[0]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 1, -1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, -1, 0] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'WF':
                if p is None:
                    p = pxyz[0]
                else:
                    assert p == pxyz[0]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 1, 0, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 0, -1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'EF':
                if p is None:
                    p = pxyz[0]
                else:
                    assert p == pxyz[0]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 1, -1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, -1, -1] = np.arange(
                            current_num, current_num+p-1)
            # __ ELSE: ERRORING ____________________________________________________
            else:
                raise Exception()
            # ----------------------------------------------------------------------
        if numbered is None:
            current_num += p - 1
        return gn, current_num

    @staticmethod
    def ___number_trace_element_position___(p, gn, current_num, trace_element_position):
        """ """
        pxyz = p
        numbered = None
        p, p1, p2 = None, None, None
        for position_t in trace_element_position:
            mesh_element_no, position = position_t.split('-')
            mesh_element_no = int(mesh_element_no)
            if position == 'N':
                if p is None:
                    p1, p2 = (pxyz[1]-1), (pxyz[2]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[1]-1) and p2 == (pxyz[2]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 0, 1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 1:-1, 1:-1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'S':
                if p is None:
                    p1, p2 = (pxyz[1]-1), (pxyz[2]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[1]-1) and p2 == (pxyz[2]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, -1, 1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 1:-1, 1:-1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'W':
                if p is None:
                    p1, p2 = (pxyz[0]-1), (pxyz[2]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[0]-1) and p2 == (pxyz[2]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 1, 0, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 0, 1:-1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'E':
                if p is None:
                    p1, p2 = (pxyz[0]-1), (pxyz[2]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[0]-1) and p2 == (pxyz[2]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 1, -1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, -1, 1:-1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'B':
                if p is None:
                    p1, p2 = (pxyz[0]-1), (pxyz[1]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[0]-1) and p2 == (pxyz[1]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 1, 1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 1:-1, 0] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'F':
                if p is None:
                    p1, p2 = (pxyz[0]-1), (pxyz[1]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[0]-1) and p2 == (pxyz[1]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 1, 1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 1:-1, -1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            else:
                raise Exception()
        if numbered is None:
            current_num += p
        return gn, current_num
