# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class _MPI_PY_2d_BS_CT(Frozen):
    """"""

    def __init__(self, bs):
        self._bs = bs
        self._freeze()

    def _parse_face_range(self, face_range):
        """"""
        if face_range is None:
            face_range = self._bs._indices
        else:
            pass
        return face_range

    def outward_unit_normal_vector(self, r, face_range=None):
        """"""
        face_range = self._parse_face_range(face_range)
        o_u_n_v = dict()
        cache = {}
        for face_index in face_range:
            face = self._bs[face_index]
            cache_key = face.metric_signature
            if cache_key in cache:
                o_u_n_v[face_index] = cache[cache_key]
            else:
                vector = face.ct.outward_unit_normal_vector(r)
                cache[cache_key] = vector
                o_u_n_v[face_index] = vector
        return o_u_n_v
