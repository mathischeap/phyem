# -*- coding: utf-8 -*-
r"""
"""
import math
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.msehtt.static.form.addons.nop_data_computer.nonlinear_2entries_base import NonlinearTwoEntriesBase
from phyem.msehtt.static.form.main import MseHttForm
from phyem.src.spaces.main import _degree_str_maker
from phyem.tools.quadrature import quadrature
from phyem.msehtt.tools.vector.dynamic import MseHttDynamicLocalVector
from phyem.msehtt.tools.vector.static.local import MseHttStaticLocalVector

_cache_2entries_LogAB_ = {}


class LogAB(NonlinearTwoEntriesBase):
    r""""""

    def __init__(self, A, B, base=math.e):
        r""""""
        super().__init__(A, B)
        assert A.tgm is B.tgm, f"the great meshes do not match!"
        assert A.tpm is B.tpm, f"the partial meshes do not match!"
        cache_key = list()
        for f in (A, B):
            assert f.__class__ is MseHttForm, f"{f} is not a {MseHttForm}!"
            cache_key.append(
                f.space.__repr__() + '@degree:' + _degree_str_maker(f.degree)
            )
        if base == math.e:
            insert_key = ' <ln_AB> '
        elif base == 10:
            insert_key = ' <log_10_AB> '
        else:
            insert_key = ' <log_%.5f_AB> ' % base
        cache_key = insert_key.join(cache_key)
        self._cache_key = cache_key

        self._melt()
        self._base = base
        self._tpm = A.tpm
        self._tgm = A.tgm
        self._freeze()

    @classmethod
    def clean_cache(cls):
        r""""""
        keys = list(_cache_2entries_LogAB_.keys())
        for key in keys:
            del _cache_2entries_LogAB_[key]

    def vector(self, about):
        r""""""
        if about == self._B:
            B_vector_caller = StaticVectorCaller(self._A, self._B, self._base)
            return MseHttDynamicLocalVector(B_vector_caller)
        else:
            raise Exception(f"log-func cannot be on the test-form "
                            f"(because we cannot guarantee that the test-function is positive locally everywhere). "
                            f"So about cannot be A")


class StaticVectorCaller(Frozen):
    r""""""
    def __init__(self, gA, tB, base):
        self._gA = gA
        self._tB = tB
        self._base = base
        self._freeze()

    def __call__(self, *args, **kwargs):
        # ----- prepare quadrature data -------------------------------------------------------------------
        if isinstance(self._gA.degree, (float, int)):
            quad_degree_A = self._gA.degree
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-A degree = {self._gA.degree}")
        if isinstance(self._tB.degree, (float, int)):
            quad_degree_B = self._tB.degree
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-B degree = {self._tB.degree}")

        quad_degree = int(max([quad_degree_A, quad_degree_B])) + 5

        if self._gA.tpm.abstract.m == self._gA.tpm.abstract.n == 2:
            indicator = 'm2n2'
            quad = quadrature((quad_degree, quad_degree), category='Gauss')
        elif self._gA.tpm.abstract.m == self._gA.tpm.abstract.n == 3:
            indicator = 'm3n3'
            quad = quadrature((quad_degree, quad_degree, quad_degree), category='Gauss')
        else:
            raise NotImplementedError()
        quad_nodes = quad.quad_nodes
        qw_ravel = quad.quad_weights_ravel
        metric_coo = [_.ravel('F') for _ in np.meshgrid(*quad_nodes, indexing='ij')]

        rmA = self._gA.reconstruction_matrix(*quad_nodes)
        rmB = self._tB.reconstruction_matrix(*quad_nodes)
        indicator += '=' + str((len(rmA), len(rmB)))

        elements = self._gA.tpm.composition
        if self._base == math.e:
            log_func = np.log
        else:
            raise NotImplementedError(f"pls define your log-func for base={self._base}")

        time = self._time_caller_(*args, **kwargs)
        gA_cochain = self._gA[time].cochain

        local_vectors = dict()
        for e in gA_cochain:
            element = elements[e]
            detJ = element.ct.Jacobian(*metric_coo)

            A_co_e = gA_cochain[e]
            RA = rmA[0][e]
            VA = RA @ A_co_e
            log_VA = log_func(VA)
            RB = rmB[0][e]
            vec_e = np.einsum('q, qk, q -> k', log_VA, RB, qw_ravel * detJ, optimize='optimal')
            local_vectors[e] = vec_e

        return MseHttStaticLocalVector(local_vectors, self._tB.cochain.gathering_matrix)

    def _time_caller_(self, *args, **kwargs):
        r""""""
        return self._gA.cochain._ati_time_caller(*args, **kwargs)


    # def _make_matrix_data(self):
    #     """"""
    #     # ---- if the data is already there -------------------------------------------
    #     if self._matrix_data is not None:
    #         pass
    #     # ---- if the data is cached ---------------------------------------------------
    #     elif self._cache_key in _cache_LogAB_matrix_:
    #         self._matrix_data = _cache_LogAB_matrix_[self._cache_key]
    #     # ------- make the data -------------------------------------------------------------
    #     else:
    #         md = self._generate_data_()
    #         self._matrix_data = md
    #         _cache_LogAB_matrix_[self._cache_key] = md
    #
    # def _generate_data_(self):
    #     r""""""
    #     if isinstance(self._A.degree, (float, int)):
    #         quad_degree_A = self._A.degree
    #     else:
    #         raise NotImplementedError(f"cannot find a quad degree from form-A degree = {self._A.degree}")
    #     if isinstance(self._B.degree, (float, int)):
    #         quad_degree_B = self._B.degree
    #     else:
    #         raise NotImplementedError(f"cannot find a quad degree from form-B degree = {self._B.degree}")
    #
    #     quad_degree = int(max([quad_degree_A, quad_degree_B])) + 5
    #
    #     if self._tpm.abstract.m == self._tpm.abstract.n == 2:
    #         indicator = 'm2n2'
    #         quad = quadrature((quad_degree, quad_degree), category='Gauss')
    #     elif self._tpm.abstract.m == self._tpm.abstract.n == 3:
    #         indicator = 'm3n3'
    #         quad = quadrature((quad_degree, quad_degree, quad_degree), category='Gauss')
    #     else:
    #         raise NotImplementedError()
    #
    #     quad_nodes = quad.quad_nodes
    #     qw_ravel = quad.quad_weights_ravel
    #
    #     metric_coo = [_.ravel('F') for _ in np.meshgrid(*quad_nodes, indexing='ij')]
    #
    #     rmA = self._A.reconstruction_matrix(*quad_nodes)
    #     rmB = self._B.reconstruction_matrix(*quad_nodes)
    #
    #     indicator += '=' + str((len(rmA), len(rmB)))
    #
    #     _cache_ = {}
    #     _matrix_data_ = {}
    #     elements = self._tpm.composition
    #
    #     if self._base == math.e:
    #         log_func = np.log
    #     else:
    #         raise NotImplementedError(f"pls define log function for base = {self._base}.")
    #
    #     for e in elements:
    #         element = elements[e]
    #         metric_signature = element.metric_signature
    #         etype = element.etype
    #
    #         if etype in (
    #                 "orthogonal rectangle",
    #                 "unique msepy curvilinear quadrilateral",
    #                 "orthogonal hexahedron",
    #         ):
    #             cache_key = metric_signature
    #
    #         elif etype == 9:
    #             reverse_info = element.dof_reverse_info
    #             if 'm2n2k1_outer' in reverse_info:
    #                 reverse_key_outer = str(reverse_info['m2n2k1_outer'])
    #             else:
    #                 reverse_key_outer = ''
    #             if 'm2n2k1_inner' in reverse_info:
    #                 reverse_key_inner = str(reverse_info['m2n2k1_inner'])
    #             else:
    #                 reverse_key_inner = ''
    #             cache_key = metric_signature + '-' + reverse_key_outer + ':' + reverse_key_inner
    #
    #         else:
    #             raise NotImplementedError()
    #
    #         if isinstance(cache_key, str) and cache_key in _cache_:
    #             _matrix_data_[e] = _cache_[cache_key]
    #
    #         else:
    #             detJ = element.ct.Jacobian(*metric_coo)
    #             if indicator == 'm2n2=(1, 1)':  # on 2d manifold in 2d space, A and B are scalar.
    #                 # so, A = a, B = b
    #                 # (log_base{A}, B) = int{ B * log_base(A)}
    #                 a = rmA[0][e]
    #                 b = rmB[0][e]
    #                 element_matrix_data = np.einsum(
    #                     'li, lj, l -> ij', log_func(a), b, qw_ravel * detJ, optimize='optimal'
    #                 )
    #
    #             elif indicator == 'm3n3=(1, 1)':  # on 3d manifold in 3d space, A and B are scalar.
    #                 # so, A = a, B = b
    #                 # (log_base{A}, B) = int{ B * log_base(A)}
    #                 a = rmA[0][e]
    #                 b = rmB[0][e]
    #                 element_matrix_data = np.einsum(
    #                     'li, lj, l -> ij', log_func(a), b, qw_ravel * detJ, optimize='optimal'
    #                 )
    #
    #             else:
    #                 raise NotImplementedError()
    #
    #             _matrix_data_[e] = element_matrix_data
    #             if isinstance(cache_key, str):
    #                 _cache_[cache_key] = element_matrix_data
    #             else:
    #                 pass
    #
    #     return _matrix_data_
