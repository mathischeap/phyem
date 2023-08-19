# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MsePyBoundarySectionMeshCooTrans(Frozen):
    """Data are in dict, keys are boundary section face id (from 0 to <amount of faces> - 1)."""

    def __init__(self, bs):
        """"""
        self._bs = bs
        base = bs.base
        imp = base.elements._index_mapping
        e2c = imp._e2c

        cache_indices_dict = dict()

        for i in self._bs.faces:
            face = self._bs.faces[i]
            element = face._element
            m, n = face._m, face._n
            cache_index = e2c[element]
            face_ct_cache_key = (cache_index, m, n)

            if face_ct_cache_key not in cache_indices_dict:
                cache_indices_dict[face_ct_cache_key] = list()
            else:
                pass

            cache_indices_dict[face_ct_cache_key].append(i)

        self._cache_indices_dict = cache_indices_dict
        # keys are the cache keys for the faces, values are the faces under these keys.

        self._freeze()

    def mapping(self, *xi_et):
        """"""
        mapping_dict = dict()
        for face_id in self._bs.faces:
            face = self._bs.faces[face_id]
            mapping_dict[face_id] = face.ct.mapping(*xi_et)  # have to compute all faces!
        return mapping_dict

    def Jacobian_matrix(self, *xi_et):
        """"""
        JM_dict = {}

        for cache_key in self._cache_indices_dict:
            faces_id = self._cache_indices_dict[cache_key]

            reference_face_id = faces_id[0]

            face = self._bs.faces[reference_face_id]

            reference_face_JM = face.ct.Jacobian_matrix(*xi_et)

            for face_id in faces_id:
                JM_dict[face_id] = reference_face_JM

        return JM_dict

    def outward_unit_normal_vector(self, *xi_et):
        """"""
        outward_unv_dict = {}

        for cache_key in self._cache_indices_dict:
            faces_id = self._cache_indices_dict[cache_key]

            reference_face_id = faces_id[0]

            face = self._bs.faces[reference_face_id]

            outward_unit_normal_vector = \
                face.ct.outward_unit_normal_vector(*xi_et)

            for face_id in faces_id:
                outward_unv_dict[face_id] = outward_unit_normal_vector

        return outward_unv_dict
