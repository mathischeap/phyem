# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM
from msehtt.static.mesh.great.elements.types.distributor import MseHttGreatMeshElementDistributor


class MseHttGreatMeshElements(Frozen):
    """"""

    def __init__(self, tgm, elements_dict):
        """"""
        self._tgm = tgm
        self._elements_dict = elements_dict
        self._parse_statistics()
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
        all_element_faces = dict()
        implemented_element_types = MseHttGreatMeshElementDistributor.implemented_element_types()
        for i in element_range:
            map_ = global_map[i]
            element_face_setting = implemented_element_types[global_etype[i]].face_setting()
            for face_id in element_face_setting:
                face_start_index, face_end_index = element_face_setting[face_id]
                face_nodes = (map_[face_start_index], map_[face_end_index])
                node0 = min(face_nodes)
                node1 = max(face_nodes)
                undirected_face_indices = (node0, node1)
                if undirected_face_indices in all_element_faces:
                    pass
                else:
                    all_element_faces[undirected_face_indices] = 0
                all_element_faces[undirected_face_indices] += 1

        boundary_element_face_undirected_indices = []
        for indices in all_element_faces:
            if all_element_faces[indices] == 1:
                indices_reverse = (indices[1], indices[0])
                boundary_element_face_undirected_indices.extend([indices, indices_reverse])

        boundary_faces = []
        for i in element_range:
            map_ = global_map[i]
            element_face_setting = implemented_element_types[global_etype[i]].face_setting()
            for face_id in element_face_setting:
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
