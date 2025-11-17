# -*- coding: utf-8 -*-
r"""This is the first renewer.

"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from phyem.tools.frozen import Frozen

from phyem.src.config import COMM, RANK, MASTER_RANK, SIZE
from phyem.tools._mpi import merge_dict

from phyem.msehtt.adaptive.form.renew.method_1_global_interpolation import ___WRAPPER_SCALAR___
from phyem.msehtt.adaptive.form.renew.method_1_global_interpolation import ___WRAPPER_VECTOR_2d___

from phyem.msehtt.static.space.reduce.Lambda.Rd_m2n2k0 import reduce_Lambda__m2n2k0
from phyem.msehtt.static.space.reduce.Lambda.Rd_m2n2k1 import reduce_Lambda__m2n2k1_inner
from phyem.msehtt.static.space.reduce.Lambda.Rd_m2n2k1 import reduce_Lambda__m2n2k1_outer
from phyem.msehtt.static.space.reduce.Lambda.Rd_m2n2k2 import reduce_Lambda__m2n2k2


class MseHtt_FormCochainRenew_Method2_BaseElementWise(Frozen):
    r""""""

    def __init__(self, ff, tf):
        r""""""
        assert ff.space.mn == tf.space.mn
        assert ff.space.indicator == tf.space.indicator
        self._ff = ff
        self._tf = tf
        self._freeze()

    def __call__(self, cochain_times_tobe_renewed, ddf=1, clean=False):
        r"""

        Parameters
        ----------
        cochain_times_tobe_renewed
        ddf :
            data density factor. How many data to be used for the global interpolation?

        Returns
        -------

        """
        not_changed_elements, changed_elements, all_same = compare_two_meshes(
            self._ff.tpm.composition, self._tf.tpm.composition
        )
        if RANK != MASTER_RANK:
            assert not_changed_elements is None, f"only return this info in the master rank."
            assert changed_elements is None, f"only return this info in the master rank."
        else:
            pass

        if all_same:
            for t in cochain_times_tobe_renewed:
                fc = self._ff[t].cochain
                tc = dict()
                assert len(self._tf.tpm.composition) == len(fc), f"Must be!"
                for e in self._tf.tpm.composition:
                    tc[e] = fc[e]
                self._tf[t].cochain = tc
        else:
            if RANK == MASTER_RANK:
                need_fitting_in_base_elements = ___find_needed_base_element_indices___(changed_elements)
                _element_distribution = self._tf.tgm._element_distribution
                rank_base_elements = ___find_rank_base_elements___(_element_distribution)
                RANKS_reconstruction_base_elements, data_id = ___find_rank_reconstruction_base_elements___(
                    need_fitting_in_base_elements, rank_base_elements
                )
            else:
                RANKS_reconstruction_base_elements = None
                data_id = None

            data_id = COMM.bcast(data_id, root=MASTER_RANK)
            rank_reconstruction_base_elements = ___scatter___(
                RANKS_reconstruction_base_elements, data_id
            )
            local_elements_tobe_reconstructed = ___find_local_reconstruction_elements___(
                rank_reconstruction_base_elements, self._ff.tpm.composition
            )

            if RANK == MASTER_RANK:
                changed_elements_id = id(changed_elements)
            else:
                changed_elements_id = None
            changed_elements_id = COMM.bcast(changed_elements_id, root=MASTER_RANK)

            rank_new_elements = ___find_rank_new_elements___(
                changed_elements, changed_elements_id, self._tf.tpm.composition
            )

            n = self._ff.space.n
            total_num_samples = 1200
            samples = int(total_num_samples ** (1 / n) * ddf) + 1
            if samples < 17:
                samples = 17
            elif samples > 81:
                samples = 81
            else:
                pass
            nodes = np.linspace(-1, 1, samples)

            for t in cochain_times_tobe_renewed:

                dtype, ff_reconstruction = self.___reconstruct_in_local_elements___(
                    t, local_elements_tobe_reconstructed, nodes
                )

                if dtype == '2d-scalar':
                    XY, U = ff_reconstruction
                    X, Y = XY
                    U = U[0]
                    X, Y, U = merge_dict(X, Y, U)

                elif dtype == '2d-vector':
                    XY, Val = ff_reconstruction
                    X, Y = XY
                    U, V = Val
                    X, Y, U, V = merge_dict(X, Y, U, V)

                else:
                    raise NotImplementedError(dtype)

                if RANK == MASTER_RANK:
                    if dtype == '2d-scalar':
                        # noinspection PyUnboundLocalVariable
                        itp_dict = ___interpolation_on_base_elements_2d_scalar___(
                            need_fitting_in_base_elements, X, Y, U
                        )

                    elif dtype == '2d-vector':
                        # noinspection PyUnboundLocalVariable
                        itp_dict = ___interpolation_on_base_elements_2d_vector___(
                            need_fitting_in_base_elements, X, Y, U, V
                        )

                    else:
                        raise NotImplementedError(dtype)

                    SCATTER = ()
                    for rank in range(SIZE):
                        rank_itp = {}
                        rank_reconstruct_base_elements = RANKS_reconstruction_base_elements[rank]
                        for e in rank_reconstruct_base_elements:
                            if e in itp_dict:
                                rank_itp[e] = itp_dict[e]
                            else:
                                pass
                        SCATTER += (rank_itp, )
                else:
                    SCATTER = None

                SCATTER = COMM.scatter(SCATTER, root=MASTER_RANK)

                OLD_COCHAIN = self._ff[t].cochain._merge_to(root=MASTER_RANK)
                OLD_COCHAIN = COMM.bcast(OLD_COCHAIN, root=MASTER_RANK)

                mn = self._tf.space.mn
                indicator = self._tf.space.indicator

                local_cochain = {}

                for index in self._tf.tpm.composition:
                    if index in rank_new_elements:
                        base_element_index = ___find_base_area___(index)
                        itp = SCATTER[base_element_index]
                        if dtype in ('2d-scalar', '3d-scalar'):
                            itp = ___WRAPPER_SCALAR___(itp)
                        elif dtype == '2d-vector':
                            itp = ___WRAPPER_VECTOR_2d___(itp)
                        else:
                            raise NotImplementedError()

                        if indicator == 'Lambda' and mn == (2, 2):
                            k = self._tf.space.abstract.k
                            orientation = self._tf.space.abstract.orientation
                            if k == 0:
                                local_element_cochain = reduce_Lambda__m2n2k0(
                                    itp, self._tf.tpm, self._tf.degree, element_range=[index,])
                            elif k == 1 and orientation == 'outer':
                                local_element_cochain = reduce_Lambda__m2n2k1_outer(
                                    itp, self._tf.tpm, self._tf.degree, element_range=[index,])
                            elif k == 1 and orientation == 'inner':
                                local_element_cochain = reduce_Lambda__m2n2k1_inner(
                                    itp, self._tf.tpm, self._tf.degree, element_range=[index,])
                            elif k == 2:
                                local_element_cochain = reduce_Lambda__m2n2k2(
                                    itp, self._tf.tpm, self._tf.degree, element_range=[index,])
                            else:
                                raise Exception()
                        else:
                            raise NotImplementedError()

                        # print(local_element_cochain[index])
                        local_cochain[index] = local_element_cochain[index]

                    else:
                        # print(OLD_COCHAIN[index])
                        local_cochain[index] = OLD_COCHAIN[index]

                self._tf[t].cochain = local_cochain

        if clean:
            self._ff.cochain.clean('all')
        else:
            pass

    def ___reconstruct_in_local_elements___(self, t, local_elements_tobe_reconstructed, nodes):
        r""""""
        fft = self._ff[t]
        mn = self._ff.space.mn
        indicator = self._ff.space.indicator

        if indicator == 'Lambda' and mn == (2, 2):
            NODES = (nodes, nodes)
            k = self._ff.space.abstract.k
            if k in (0, 2):
                dtype = '2d-scalar'
            else:
                dtype = '2d-vector'
        elif indicator == 'Lambda' and mn == (3, 3):
            NODES = (nodes, nodes, nodes)
            k = self._ff.space.abstract.k
            if k in (0, 3):
                dtype = '3d-scalar'
            else:
                dtype = '3d-vector'
        else:
            raise NotImplementedError()

        R = fft.reconstruct(*NODES, ravel=True, element_range=local_elements_tobe_reconstructed)

        return dtype, R


_local_cache_2_ = {
    'id': 0,
    'rankwise': {},
}


def ___find_rank_base_elements___(_element_distribution):
    r""""""
    ID = id(_element_distribution)
    if ID == _local_cache_2_['id']:
        return _local_cache_2_['rankwise']
    else:
        rankwise = {}
        for rank in _element_distribution:
            wise = []
            for index in _element_distribution[rank]:
                base_index = ___find_base_area___(index)
                if base_index in wise:
                    pass
                else:
                    wise.append(base_index)
            rankwise[rank] = wise
        _local_cache_2_['rankwise'] = rankwise
        _local_cache_2_['id'] = ID
        return rankwise


_local_cache_3_ = {
    'id': '',
    'rank_reconstruction': (),
}


def ___find_rank_reconstruction_base_elements___(need_fitting_in_base_elements, rank_base_elements):
    r""""""
    id0 = id(need_fitting_in_base_elements)
    id1 = id(rank_base_elements)
    data_id = str(id0) + '+' + str(id1)

    if data_id == _local_cache_3_['id']:
        return _local_cache_3_['rank_reconstruction']
    else:
        _local_cache_3_['id'] = data_id

    rank_reconstruction_base_elements = ()
    for rank in range(SIZE):
        rank_reconstruction_elements = []
        for e in rank_base_elements[rank]:
            if e in need_fitting_in_base_elements:
                rank_reconstruction_elements.append(e)
            else:
                pass
        rank_reconstruction_base_elements += (rank_reconstruction_elements, )

    # noinspection PyTypeChecker
    _local_cache_3_['rank_reconstruction'] = (rank_reconstruction_base_elements, data_id)

    return rank_reconstruction_base_elements, data_id


_local_cache_6_ = {
    'id': 0,
    'rankNewElements': [],
}


def ___find_rank_new_elements___(all_changed_elements, changed_elements_id, new_elements):
    r""""""
    if changed_elements_id == _local_cache_6_['id']:
        return _local_cache_6_['rankNewElements']
    else:
        _local_cache_6_['id'] = changed_elements_id

        all_changed_elements = COMM.bcast(all_changed_elements, root=MASTER_RANK)

        rank_new_elements = list()
        for e in new_elements:
            if e in all_changed_elements:
                rank_new_elements.append(e)
            else:
                pass
        _local_cache_6_['rankNewElements'] = rank_new_elements
        return rank_new_elements


def ___interpolation_on_base_elements_2d_scalar___(rank_reconstruction_base_elements, X, Y, U):
    r""""""
    ITP_Dict = {}
    for base_index in rank_reconstruction_base_elements:
        x = list()
        y = list()
        u = list()
        for index in X:
            if ___find_base_area___(index) == base_index:
                x.extend(X[index])
                y.extend(Y[index])
                u.extend(U[index])
            else:
                pass
        itp = LinearNDInterpolator(np.array([x, y]).T, np.array(u))
        ITP_Dict[base_index] = itp

    return ITP_Dict


def ___interpolation_on_base_elements_2d_vector___(rank_reconstruction_base_elements, X, Y, U, V):
    r""""""
    ITP_Dict = {}
    for base_index in rank_reconstruction_base_elements:
        x = list()
        y = list()
        u = list()
        v = list()
        for index in X:
            if ___find_base_area___(index) == base_index:
                x.extend(X[index])
                y.extend(Y[index])
                u.extend(U[index])
                v.extend(V[index])
            else:
                pass
        # print(np.array([x, y]).T, base_index)
        itp = LinearNDInterpolator(np.array([x, y]).T, np.array([u, v]).T)
        ITP_Dict[base_index] = itp

    return ITP_Dict
