# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM, MPI
from msehtt.static.mesh.great.elements.types.distributor import MseHttGreatMeshElementDistributor

from random import random

from tools.functions.time_space._2d.wrappers.vector import T2dVector
# from tools.functions.time_space._2d.wrappers.tensor import T2dTensor


class MseHttGreatMeshElements(Frozen):
    """"""

    def __init__(self, tgm, elements_dict, element_face_topology_mismatch=True):
        """"""
        self._tgm = tgm
        self._elements_dict = elements_dict
        self._parse_statistics()
        self._mn = None
        self._periodic_face_pairing = None
        if element_face_topology_mismatch:
            self._parse_form_face_dof_topology_mismatch()
            self._element_face_topology_mismatch = True   # This is an unstructured mesh.
        else:
            # When we are sure that there is no mismatch, we can skip it. For example, when we build the
            # great mesh from a msepy mesh, then there is no mismatch for sure.
            self._element_face_topology_mismatch = False
        self._freeze()

    def __repr__(self):
        """"""
        return f"<elements of {self._tgm}>"

    @property
    def ___is_msehtt_great_mesh_elements___(self):
        """Just a signature"""
        return True

    @property
    def tgm(self):
        """Return the great mesh I am built on."""
        return self._tgm

    def __getitem__(self, item):
        """Return the element indexed ``item``."""
        return self._elements_dict[item]

    def __len__(self):
        """how many elements in this rank?"""
        return len(self._elements_dict)

    def __contains__(self, item):
        """If the element indexed ``item`` is an valid element in this rank?"""
        return item in self._elements_dict

    def __iter__(self):
        """iterate over all element indices in this rank."""
        for element_index in self._elements_dict:
            yield element_index

    def map_(self, i):
        """Return the element map of a local element."""
        return self[i].map_

    @property
    def global_map(self):
        """The element map of all elements across all ranks. This information is usually only stored in the
        master rank.
        """
        if RANK == MASTER_RANK:
            return self._tgm._global_element_map_dict
        else:
            return None

    @property
    def global_etype(self):
        """The element type of all elements across all ranks. This information is usually only stored in the
        master rank.
        """
        if RANK == MASTER_RANK:
            return self._tgm._global_element_type_dict
        else:
            return None

    def _parse_boundary_faces_of_a_patch_of_elements(self, element_range):
        """"""
        if RANK != MASTER_RANK:
            raise Exception('Can only parse boundary faces of a bunch of elements in the master rank.')
        else:
            pass

        global_map = self.global_map
        global_etype = self.global_etype

        # ----- find all element faces: keys are the face nodes -------------------------------------
        all_element_faces = dict()
        implemented_element_types = MseHttGreatMeshElementDistributor.implemented_element_types()
        for i in element_range:
            map_ = global_map[i]
            element_class = implemented_element_types[global_etype[i]]
            element_face_setting = element_class.face_setting()
            element_n = element_class.n()
            for face_id in element_face_setting:
                if element_n == 2:  # 2d element: only have two face nodes.
                    face_start_index, face_end_index = element_face_setting[face_id]
                    face_nodes = (map_[face_start_index], map_[face_end_index])
                    node0 = min(face_nodes)
                    node1 = max(face_nodes)
                    undirected_face_indices = (node0, node1)
                elif element_n == 3:  # 3d elements
                    face_node_indices = element_face_setting[face_id]
                    face_nodes = [map_[_] for _ in face_node_indices]
                    face_nodes.sort()
                    undirected_face_indices = tuple(face_nodes)
                else:
                    raise NotImplementedError()
                if undirected_face_indices in all_element_faces:
                    pass
                else:
                    all_element_faces[undirected_face_indices] = 0
                all_element_faces[undirected_face_indices] += 1

        # -------- find those element faces only appear once ---------------------------------------
        boundary_element_face_undirected_indices = []
        for indices in all_element_faces:
            if all_element_faces[indices] == 1:
                if len(indices) == 2:
                    # for those faces only have two nodes, we add (n0, n1) and (n1, n0) to boundary face indicators.
                    indices_reverse = (indices[1], indices[0])
                    boundary_element_face_undirected_indices.extend([indices, indices_reverse])
                else:
                    boundary_element_face_undirected_indices.append(indices)
            else:
                pass

        # ------- pick up those faces on boundary -------------------------------------------------
        boundary_faces = []
        for i in element_range:
            map_ = global_map[i]
            element_class = implemented_element_types[global_etype[i]]
            element_face_setting = element_class.face_setting()
            element_n = element_class.n()
            for face_id in element_face_setting:
                if element_n == 2:  # 2d element: only have two face nodes.
                    face_start_index, face_end_index = element_face_setting[face_id]
                    face_nodes = (map_[face_start_index], map_[face_end_index])
                    if face_nodes in boundary_element_face_undirected_indices:
                        boundary_faces.append(
                            {
                                'element index': i,        # this face is on the element indexed ``i``.
                                'face id': face_id,        # this face is of this face id in element indexed ``i``.
                                'local node indices': element_face_setting[face_id],   # face nodes local indices
                                'global node numbering': face_nodes,                    # face node global numbering.
                            }
                        )
                    else:
                        pass
                else:
                    face_node_indices = element_face_setting[face_id]
                    face_nodes = [map_[_] for _ in face_node_indices]
                    face_nodes.sort()
                    face_nodes = tuple(face_nodes)
                    if face_nodes in boundary_element_face_undirected_indices:
                        boundary_faces.append(
                            {
                                'element index': i,        # this face is on the element indexed ``i``.
                                'face id': face_id,        # this face is of this face id in element indexed ``i``.
                                'local node indices': face_node_indices,   # face nodes local indices
                                'global node numbering': tuple([map_[_] for _ in face_node_indices]),
                                # face node global numbering.
                            }
                        )
                    else:
                        pass
        # =========================================================================================
        return boundary_faces

    @property
    def statistics(self):
        """Return some global statistic numbers. Since they are global, they will be same in all ranks."""
        return self._statistics

    def _parse_statistics(self):
        """Parse some global statistics. Return same in all ranks."""
        etype_pool = {}
        for i in self:
            element = self[i]
            etype = element.etype
            if etype not in etype_pool:
                etype_pool[etype] = 0
            else:
                pass
            etype_pool[etype] += 1

        rank_etype = COMM.gather(etype_pool, root=MASTER_RANK)

        if RANK == MASTER_RANK:

            etype_pool = {}
            for pool in rank_etype:
                for etype in pool:
                    if etype not in etype_pool:
                        etype_pool[etype] = 0
                    else:
                        pass
                    etype_pool[etype] += pool[etype]

            total_amount_elements = 0
            for etype in etype_pool:
                total_amount_elements += etype_pool[etype]

            statistics = {
                'total amount elements': total_amount_elements,
                'amount element types': len(etype_pool),
                'total amount different elements': etype_pool
            }

        else:
            statistics = None

        self._statistics = COMM.bcast(statistics, root=MASTER_RANK)

    @property
    def mn(self):
        """The `m` and `n` of the elements I have. All the elements across all ranks will take into account.
        So if the elements in a local rank are of one (m, n), (m, n) could be different in other ranks. So
        self.mn will return a tuple if two pairs of (m, n).

        For example, if self.mn = (2, 2), then all the elements I have are 2d (n=2) elements in 2d (m=2) space.

        If self.mn == ((3, 2), (2, 2)), then some of the elements I have are 2d (n=2) elements in 3d (m=2) space,
        and some other elements are 2d (n=2) element in 2d (m=2) space. Of course, this is not very likely.
        """
        if self._mn is None:
            mn_pool = list()
            for element_index in self:
                element = self[element_index]
                mn = (element.m(), element.n())
                if mn not in mn_pool:
                    mn_pool.append(mn)
                else:
                    pass
            mn_pool = COMM.gather(mn_pool, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                total_mn_pool = set()
                for _ in mn_pool:
                    total_mn_pool.update(_)
                total_mn_pool = list(total_mn_pool)

                if len(total_mn_pool) == 1:
                    self._mn = total_mn_pool[0]
                else:
                    self._mn = tuple(total_mn_pool)
            else:
                self._mn = None

            self._mn = COMM.bcast(self._mn, root=MASTER_RANK)
        return self._mn

    @property
    def periodic_face_pairing(self):
        """A dictionary shows the periodic pairing.

        For example,
            in 2d:

                periodic_face_pairing = {

                }

        """
        if self._periodic_face_pairing is not None:
            return self._periodic_face_pairing
        else:
            pass

        if self.mn == (2, 2):
            self._periodic_face_pairing = self.___periodic_face_pairing_m2n2___()
        else:
            raise NotImplementedError(f"not implemented for (m,n) == {self.mn}")

        return self._periodic_face_pairing

    def ___periodic_face_pairing_m2n2___(self):
        r""""""
        face_coo_pool = {}
        face_element_pool = {}

        for e in self:
            element = self[e]
            face_representative_str = element.___face_representative_str___()
            element_map = element._map
            face_setting = element.face_setting()

            for local_face_index in face_setting:
                nodes = face_setting[local_face_index]
                face_map = [element_map[_] for _ in nodes]
                face_map.sort()
                face_map = tuple(face_map)
                coo = face_representative_str[local_face_index]
                if face_map in face_coo_pool:
                    if coo in face_coo_pool[face_map]:
                        pass
                    else:
                        face_coo_pool[face_map].append(coo)
                else:
                    face_coo_pool[face_map] = [coo]

                if face_map in face_element_pool:
                    face_element_pool[face_map].append((e, local_face_index))
                else:
                    face_element_pool[face_map] = [(e, local_face_index)]

        face_element_POOL = COMM.gather(face_element_pool, root=MASTER_RANK)
        face_coo_POOL = COMM.gather(face_coo_pool, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            POOL = face_element_POOL[0]
            for pool in face_element_POOL[1:]:
                for face_map in pool:
                    if face_map in POOL:
                        POOL[face_map].extend(pool[face_map])
            face_element_POOL = POOL

            POOL = face_coo_POOL[0]
            for pool in face_coo_POOL[1:]:
                for face_map in pool:
                    if face_map in POOL:
                        POOL[face_map].extend(pool[face_map])
            face_coo_POOL = POOL

            periodic_face_pairing_m2n2 = {}
            for face_undirected in face_coo_POOL:
                amount_coo = set(face_coo_POOL[face_undirected])
                if len(amount_coo) == 1:
                    pass
                else:
                    periodic_face_pairing_m2n2[face_undirected] = face_element_POOL[face_undirected]

        else:
            periodic_face_pairing_m2n2 = None

        periodic_face_pairing_m2n2 = COMM.bcast(periodic_face_pairing_m2n2, root=MASTER_RANK)
        local_periodic_face_pairing_m2n2 = {}
        for undirected_face in periodic_face_pairing_m2n2:
            if undirected_face in face_element_pool:
                local_periodic_face_pairing_m2n2[undirected_face] = periodic_face_pairing_m2n2[undirected_face]
            else:
                pass

        return local_periodic_face_pairing_m2n2

    @staticmethod
    def ___random_m2n2___():
        """"""
        if RANK == MASTER_RANK:
            _ = [random() for _ in range(6)]
        else:
            _ = None

        _ = COMM.bcast(_, root=MASTER_RANK)
        a, b, c, d, e, f = _

        def test_field_1(t, x, y):
            return 10 * np.sin(2 * np.pi * (x - a)) * np.sin(2 * np.pi * (y + f)) + t * 0 + 10

        def test_field_2(t, x, y):
            return 10 * np.cos(2 * np.pi * (x - b)) * np.sin(2 * np.pi * (y + c)) + t * 0 + 10

        def test_field_3(t, x, y):
            return 10 * np.sin(2 * np.pi * (x + d)) * np.sin(2 * np.pi * (y - e)) + t * 0 + 10

        def test_field_4(t, x, y):
            return 10 * np.cos(2 * np.pi * (x + e)) * np.cos(2 * np.pi * (y - f)) + t * 0 + 10

        return test_field_1, test_field_2, test_field_3, test_field_4

    def _parse_form_face_dof_topology_mismatch(self):
        """"""
        if self.mn == (2, 2):
            self.___parse_form_face_dof_topology_mismatch_m2n2___()
        else:
            raise NotImplementedError(f"not implemented for mesh of (m, n) == {self.mn}.")

    def ___parse_form_face_dof_topology_mismatch_m2n2___(self):
        r""""""
        periodic = len(self.periodic_face_pairing) != 0
        periodic = COMM.allreduce(periodic, op=MPI.LOR)

        if periodic:
            # noinspection PyUnusedLocal
            def test_field_1(t, x, y):
                return np.ones_like(x)

            # noinspection PyUnusedLocal
            def test_field_2(t, x, y):
                return - np.ones_like(x)

            # noinspection PyUnusedLocal
            def test_field_3(t, x, y):
                return 0.5 * np.ones_like(x)

            # noinspection PyUnusedLocal
            def test_field_4(t, x, y):
                return - 0.5 * np.ones_like(x)
        else:
            test_field_1, test_field_2, test_field_3, test_field_4 = self.___random_m2n2___()

        vector = T2dVector(test_field_2, test_field_3)
        # tensor = T2dTensor(test_field_1, test_field_2, test_field_3, test_field_4)

        to_be_reversed = {
            "m2n2k1_inner": self.___get_form_face_dof_topology_mismatch_m2n2k1_inner___(vector),
            "m2n2k1_outer": self.___get_form_face_dof_topology_mismatch_m2n2k1_outer___(vector),
        }

        for form_indicator in to_be_reversed:
            for position in to_be_reversed[form_indicator]:
                element_index, face_index = position
                if element_index in self:  # we find a local element.
                    element = self[element_index]
                    if form_indicator in element._dof_reverse_info:
                        element._dof_reverse_info[form_indicator].append(face_index)
                    else:
                        element._dof_reverse_info[form_indicator] = [face_index]

        for e in self:
            element = self[e]
            for indicator in element._dof_reverse_info:
                element._dof_reverse_info[indicator].sort()

    def ___get_form_face_dof_topology_mismatch_m2n2k1_inner___(self, vector):
        # --------- m2n2 Lambda 1-form inner --------------------------------------------------
        from msehtt.static.space.gathering_matrix.Lambda.GM_m2n2k1 import gathering_matrix_Lambda__m2n2k1_inner
        gm_m2n2k1_inner = gathering_matrix_Lambda__m2n2k1_inner(self._tgm, 1, do_cache=False)
        gm = gm_m2n2k1_inner._gm

        from msehtt.static.space.reduce.Lambda.Rd_m2n2k1 import reduce_Lambda__m2n2k1_inner
        referring_cochain = reduce_Lambda__m2n2k1_inner(vector[0], self, 1, raw=True)

        face_cochain_indices = {
            5: [0, 2, 1],  # when degree=1, edge #0 -> local dof #0, on edge #1 -> local dof #2, and so on
            'unique msepy curvilinear triangle': [0, 2, 1],
            'orthogonal rectangle': [2, 3, 0, 1],
            'unique msepy curvilinear quadrilateral': [2, 3, 0, 1],
            9: [2, 3, 0, 1],
            'unique curvilinear quad': [2, 3, 0, 1],
        }

        priority = [9, 'unique curvilinear quad', 'unique msepy curvilinear triangle', 5]
        # we will reverse dofs in element of etype later in this list.
        # for example, a 9-typed element is paired to a 5-typed element, we
        # always change sign of dofs in the 5-typed element.

        return self.___get_form_face_dof_topology_mismatch_m2n2k1___(
            gm, referring_cochain, face_cochain_indices, priority
        )

    def ___get_form_face_dof_topology_mismatch_m2n2k1_outer___(self, vector):
        # --------- m2n2 Lambda 1-form inner --------------------------------------------------
        from msehtt.static.space.gathering_matrix.Lambda.GM_m2n2k1 import gathering_matrix_Lambda__m2n2k1_outer
        gm_m2n2k1_outer = gathering_matrix_Lambda__m2n2k1_outer(self._tgm, 1, do_cache=False)
        gm = gm_m2n2k1_outer._gm

        from msehtt.static.space.reduce.Lambda.Rd_m2n2k1 import reduce_Lambda__m2n2k1_outer
        referring_cochain = reduce_Lambda__m2n2k1_outer(vector[0], self, 1, raw=True)

        face_cochain_indices = {
            5: [1, 0, 2],  # when degree=1, edge #0 -> local dof #1, on edge #1 -> local dof #0, and so on
            'unique msepy curvilinear triangle': [1, 0, 2],
            'orthogonal rectangle': [0, 1, 2, 3],
            'unique msepy curvilinear quadrilateral': [0, 1, 2, 3],
            9: [0, 1, 2, 3],
            'unique curvilinear quad': [0, 1, 2, 3],
        }

        priority = [9, 'unique curvilinear quad', 'unique msepy curvilinear triangle', 5]
        # we will reverse dofs in element of etype later in this list.
        # for example, a 9-typed element is paired to a 5-typed element, we
        # always change sign of dofs in the 5-typed element.

        return self.___get_form_face_dof_topology_mismatch_m2n2k1___(
            gm, referring_cochain, face_cochain_indices, priority
        )

    def ___get_form_face_dof_topology_mismatch_m2n2k1___(
            self, gm, referring_cochain, face_cochain_indices, priority
    ):

        referring_cochain = COMM.gather(referring_cochain, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            GEM = self.global_map
            GET = self.global_etype

            RC = {}
            for _ in referring_cochain:
                RC.update(_)
            referring_cochain = RC

            pool_position = {}
            pool_value = {}
            pool_gm = {}

            all_element_types = MseHttGreatMeshElementDistributor.implemented_element_types()

            for e in gm:
                e_class = all_element_types[GET[e]]
                etype = e_class._etype()  # for example, pixel in fact has class etype 'orthogonal rectangle'.
                element_face_setting = e_class.face_setting()
                element_map = GEM[e]
                erc = referring_cochain[e]
                gme = gm[e]

                for face_index, cochain_index in enumerate(face_cochain_indices[etype]):

                    position = (etype, e, face_index)
                    value = float(erc[cochain_index])
                    gm_face = gme[cochain_index]

                    local_nodes = element_face_setting[face_index]
                    face_nodes = [element_map[_] for _ in local_nodes]
                    face_nodes.sort()
                    face_nodes = tuple(face_nodes)

                    if face_nodes in pool_position:
                        pool_position[face_nodes].append(position)
                    else:
                        pool_position[face_nodes] = [position]

                    if face_nodes in pool_value:
                        pool_value[face_nodes].append(value)
                    else:
                        pool_value[face_nodes] = [value]

                    if face_nodes in pool_gm:
                        assert gm_face == pool_gm[face_nodes], \
                            f"must be! to check the faces have the same numbering."
                    else:
                        pool_gm[face_nodes] = gm_face

            reversing_dof_places = []

            for undirected_face in pool_position:
                positions = pool_position[undirected_face]
                if len(positions) == 1:
                    pass
                elif len(positions) == 2:
                    values = pool_value[undirected_face]
                    v0, v1 = values
                    if np.isclose(v0, v1):
                        pass
                    else:
                        assert np.isclose(v0, -v1), f"must be v0 + v1 == 0, now v0={v0}, v1={v1}"
                        pos0, pos1 = positions
                        etype0, etype1 = pos0[0], pos1[0]
                        if etype0 in priority:
                            priority0 = priority.index(etype0)
                        else:
                            priority0 = -1
                        if etype1 in priority:
                            priority1 = priority.index(etype1)
                        else:
                            priority1 = -1

                        assert not priority0 == priority1 == -1, \
                            (f'We must found an element to reverse dof on its face. '
                             f'Something wrong in the paired elements? '
                             f'The found element types are {etype0}, {etype1}. It cannot be that both element types '
                             f'are structured elements (like `orthogonal rectangle` or '
                             f'`unique msepy curvilinear quadrilateral`). Or we should renew the priority list.')

                        if priority0 >= priority1:
                            touch = pos0
                        else:
                            touch = pos1

                        reversing_dof_places.append(
                            (touch[1], touch[2])
                        )

                else:
                    raise Exception(positions)
        else:
            reversing_dof_places = None

        reversing_dof_places = COMM.bcast(reversing_dof_places, root=MASTER_RANK)
        return reversing_dof_places
