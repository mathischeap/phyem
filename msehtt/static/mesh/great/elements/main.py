# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM, SIZE
from msehtt.static.mesh.great.elements.types.distributor import MseHttGreatMeshElementDistributor


class MseHttGreatMeshElements(Frozen):
    """"""

    def __init__(self, tgm, elements_dict, element_face_topology_mismatch=True):
        """"""
        self._tgm = tgm
        self._elements_dict = elements_dict
        self._parse_statistics()
        if element_face_topology_mismatch:
            self._parse_form_face_dof_topology_mismatch()
        else:
            # When we are sure that there is no mismatch, we can skip it. For example, when we build the
            # great mesh from a msepy mesh, then there is no mismatch for sure.
            pass
        self._freeze()

    def __repr__(self):
        """"""
        return f"<elements of {self._tgm}>"

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
                    raise NotImplementedError()
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
                else:
                    raise NotImplementedError()
                if face_nodes in boundary_element_face_undirected_indices:
                    boundary_faces.append(
                        {
                            'element index': i,        # this face is on the element indexed ``i``.
                            'face id': face_id,        # this face is of this face id in element indexed ``i``.
                            'local node indices': element_face_setting[face_id],   # face nodes local indices
                            'global node numbering': face_nodes,                    # face node global numbering.
                        }
                    )

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

    def _parse_form_face_dof_topology_mismatch(self):
        """"""
        if RANK == MASTER_RANK:
            implemented_element_types = MseHttGreatMeshElementDistributor.implemented_element_types()
            global_element_type = self._tgm._global_element_type_dict
            global_map = self._tgm._global_element_map_dict
            involved_forms = None
            for element_index in global_element_type:
                etype = global_element_type[element_index]
                element_class = implemented_element_types[etype]
                topology = element_class._form_face_dof_direction_topology()
                if involved_forms is None:
                    involved_forms = list(topology.keys())
                else:
                    for key in topology:
                        assert key in involved_forms, f"Some element miss topology info for form {key}."

            paired = {}
            for form_indicator in involved_forms:
                paired[form_indicator] = {}

            for element_index in global_element_type:
                etype = global_element_type[element_index]
                element_class = implemented_element_types[etype]
                topology = element_class._form_face_dof_direction_topology()
                face_setting = element_class.face_setting()
                element_map = global_map[element_index]
                for form_indicator in topology:
                    the_pairing = paired[form_indicator]
                    info = topology[form_indicator]
                    # --------- m2n2k1_outer ------------------------------------------------------------
                    if form_indicator == 'm2n2k1_outer':
                        for face_index in info:
                            sign = info[face_index]
                            start, end = face_setting[face_index]
                            global_start, global_end = element_map[start], element_map[end]
                            if global_start < global_end:
                                node0 = global_start
                                node1 = global_end
                            else:
                                node0 = global_end
                                node1 = global_start
                            if (node0, node1) in the_pairing:
                                pass
                            else:
                                the_pairing[(node0, node1)] = list()
                            the_pairing[(node0, node1)].append((element_index, face_index, sign))
                    # --------- m2n2k1_inner ------------------------------------------------------------
                    elif form_indicator == 'm2n2k1_inner':
                        for face_index in info:
                            sign = info[face_index]
                            start, end = face_setting[face_index]
                            global_start, global_end = element_map[start], element_map[end]
                            if global_start < global_end:
                                node0 = global_start
                                node1 = global_end
                            else:
                                node0 = global_end
                                node1 = global_start
                            if (node0, node1) in the_pairing:
                                pass
                            else:
                                the_pairing[(node0, node1)] = list()
                            the_pairing[(node0, node1)].append((element_index, face_index, sign))
                    # ====================================================================================
                    else:
                        raise NotImplementedError()

            # now we decide to apply '-' to which dofs according the information we have collected
            reverse_places = {}

            for form_indicator in paired:
                form_reverse_places = list()
                the_pairing = paired[form_indicator]
                # --------- m2n2k1_outer ------------------------------------------------------------
                if form_indicator == 'm2n2k1_outer':
                    for face_nodes in the_pairing:
                        element_faces = the_pairing[face_nodes]
                        if len(element_faces) == 1:
                            pass
                        elif len(element_faces) == 2:
                            face0, face1 = element_faces
                            sign0, sign1 = face0[-1], face1[-1]
                            if sign0 != sign1:  # one enter the element, the other leave the element, fine!
                                pass
                            else:
                                reverse_dof_place = face0[:2]
                                form_reverse_places.append(reverse_dof_place)
                        else:
                            raise Exception('A face cannot appear at more than two places (element).')
                # --------- m2n2k1_inner ------------------------------------------------------------
                elif form_indicator == 'm2n2k1_inner':
                    for face_nodes in the_pairing:
                        element_faces = the_pairing[face_nodes]
                        if len(element_faces) == 1:
                            pass
                        elif len(element_faces) == 2:
                            face0, face1 = element_faces
                            sign0, sign1 = face0[-1], face1[-1]
                            if sign0 != sign1:  # one enter the element, the other leave the element, fine!
                                pass
                            else:
                                reverse_dof_place = face0[:2]
                                form_reverse_places.append(reverse_dof_place)
                        else:
                            raise Exception('A face cannot appear at more than two places (element).')
                # ====================================================================================
                else:
                    raise NotImplementedError()

                reverse_places[form_indicator] = form_reverse_places

            # now we distribute the reverse_places to ranks -----------------
            form_indicators = list(reverse_places.keys())
        else:
            form_indicators = None

        form_indicators = COMM.bcast(form_indicators, root=MASTER_RANK)

        for fid in form_indicators:
            if RANK == MASTER_RANK:
                # noinspection PyUnboundLocalVariable
                places = reverse_places[fid]
                to_be_distributed = [list() for _ in range(SIZE)]
                for element_index__face_index in places:
                    element_index = element_index__face_index[0]
                    rank_of_element = -1
                    for rank in self._tgm._element_distribution:
                        if element_index in self._tgm._element_distribution[rank]:
                            rank_of_element = rank
                            break
                        else:
                            pass
                    assert rank_of_element != -1, f"must have found a rank!"
                    to_be_distributed[rank_of_element].append(element_index__face_index)

            else:
                to_be_distributed = None

            to_be_distributed = COMM.scatter(to_be_distributed, root=MASTER_RANK)

            for element_index__face_index in to_be_distributed:
                element_index, face_index = element_index__face_index
                assert element_index in self, f'must be!'
                element = self[element_index]
                if fid in element._dof_reverse_info:
                    pass
                else:
                    element._dof_reverse_info[fid] = list()
                element._dof_reverse_info[fid].append(face_index)
