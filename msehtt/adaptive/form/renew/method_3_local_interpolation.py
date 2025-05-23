# -*- coding: utf-8 -*-
r"""This is the first renewer. It collects the reconstruction data of the old form to the master rank.
Then it uses these data to make a global interpolation. Then this global interpolation is used for
the reduction of cochains of the new form.

"""
import numpy as np
from src.config import RANK, MASTER_RANK, COMM
from tools._mpi import merge_dict

from tools.frozen import Frozen
from scipy.interpolate import NearestNDInterpolator

from msehtt.adaptive.form.renew.method_1_global_interpolation import ___WRAPPER_SCALAR___
from msehtt.adaptive.form.renew.method_1_global_interpolation import ___WRAPPER_VECTOR_2d___

from msehtt.static.space.reduce.Lambda.Rd_m2n2k0 import reduce_Lambda__m2n2k0
from msehtt.static.space.reduce.Lambda.Rd_m2n2k1 import reduce_Lambda__m2n2k1_inner
from msehtt.static.space.reduce.Lambda.Rd_m2n2k1 import reduce_Lambda__m2n2k1_outer
from msehtt.static.space.reduce.Lambda.Rd_m2n2k2 import reduce_Lambda__m2n2k2

from msehtt.static.mesh.partial.elements.main import MseHttElementsPartialMesh


class MseHtt_FormCochainRenew_Method3_LocalInterpolation(Frozen):
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
        renew_info = {}

        not_changed_elements, changed_elements, all_same = compare_two_meshes(
            self._ff.tpm.composition, self._tf.tpm.composition
        )

        if RANK != MASTER_RANK:
            assert not_changed_elements is None, f"only return this info in the master rank."
            assert changed_elements is None, f"only return this info in the master rank."
        else:
            pass

        assert all_same == 1 or all_same == 0, f"all_same indicator must be 1 or 0."
        ALL_SAME = COMM.gather(all_same, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            assert len(set(ALL_SAME)) == 1, f"MUST BE SAME IN ALL RANKS."

        renew_info['all same'] = all_same

        COMM.barrier()

        if all_same:
            for t in cochain_times_tobe_renewed:
                fc = self._ff[t].cochain
                tc = dict()
                assert len(self._tf.tpm.composition) == len(fc), f"Must be!"
                for e in self._tf.tpm.composition:
                    tc[e] = fc[e]
                self._tf[t].cochain = tc
        else:
            self._partially_renew_(
                cochain_times_tobe_renewed,
                changed_elements,
                ddf=ddf,
            )

        if clean:
            self._ff.cochain.clean('all')
        else:
            pass

        return renew_info

    def _partially_renew_(
            self,
            cochain_times_tobe_renewed,
            changed_elements,
            ddf=1,
    ):
        r""""""

        COMM.barrier()

        if RANK == MASTER_RANK:
            base_elements = ___find_needed_base_element_indices___(changed_elements)
            data_id = id(base_elements)

        else:
            base_elements = None
            data_id = None

        COMM.barrier()

        data_id = COMM.bcast(data_id, root=MASTER_RANK)
        base_elements = ___bcast___(base_elements, data_id)

        COMM.barrier()

        local_ff_elements = ___find_ff_reconstruction_elements___(
            base_elements, self._ff.tpm.composition
        )

        COMM.barrier()

        local_tf_elements = ___find_tf_reconstruction_elements___(
            base_elements, self._tf.tpm.composition
        )

        n = self._ff.space.n
        total_num_samples = 11007
        samples = int(total_num_samples ** (1/n) * ddf) + 1

        if samples < 31:
            samples = 31
        elif samples > 181:
            samples = 181
        else:
            pass

        nodes = np.linspace(-1, 1, samples)
        nodes = (nodes[1:] + nodes[:-1]) / 2

        mn = self._ff.space.mn

        if self._ff.space.indicator == 'Lambda' and mn == (2, 2):
            NODES = (nodes, nodes)
        elif self._ff.space.indicator == 'Lambda' and mn == (3, 3):
            NODES = (nodes, nodes, nodes)
        else:
            raise NotImplementedError()

        COMM.barrier()

        for t in cochain_times_tobe_renewed:
            new_cochain = self.___renew_a_cochain___(t, NODES, local_ff_elements, local_tf_elements)
            old_cochain = self._ff[t].cochain._merge_to(root=MASTER_RANK)
            old_cochain = COMM.bcast(old_cochain, root=MASTER_RANK)

            cochain_dict = {}
            for e in self._tf.tpm.composition:
                if e in new_cochain:
                    cochain_dict[e] = new_cochain[e]
                else:
                    cochain_dict[e] = old_cochain[e]

            COMM.barrier()
            self._tf[t].cochain = cochain_dict
            self._tf[t].cochain.homogenous()

    def ___renew_a_cochain___(self, t, NODES, local_ff_elements, local_tf_elements):
        r""""""
        ff = self._ff[t]
        rec = ff.reconstruct(*NODES, ravel=True, element_range=local_ff_elements)
        COO, VAL = rec

        case = ''
        if len(COO) == 2:
            X, Y = COO
            case += '2d'
        elif len(COO) == 3:
            X, Y, Z = COO
            case += '3d'
        else:
            raise Exception()

        if len(VAL) == 1 and isinstance(VAL[0], dict):
            # salar
            U = VAL[0]
            V = None
            case += '-scalar'
        elif len(VAL) == 2 and all([isinstance(VAL[_], dict) for _ in range(2)]):
            # vector
            U, V = VAL
            case += '-vector'
        elif len(VAL) == 3 and all([isinstance(VAL[_], dict) for _ in range(3)]):
            # vector
            U, V, W = VAL
            case += '-vector'
        else:
            raise NotImplementedError()

        COMM.barrier()

        if case == '2d-scalar':
            new_cochain = self.___renew_2d_scalar___(X, Y, U, local_tf_elements)
        elif case == '2d-vector':
            new_cochain = self.___renew_2d_vector___(X, Y, U, V, local_tf_elements)
        else:
            raise NotImplementedError()

        return new_cochain

    def ___renew_2d_scalar___(self, X, Y, value, element_range):
        r""""""
        X, Y, value = merge_dict(X, Y, value)
        if RANK == MASTER_RANK:
            X = np.array(list(X.values()))
            Y = np.array(list(Y.values()))
            value = np.array(list(value.values()))
            X = X.ravel()
            Y = Y.ravel()
            value = value.ravel()
            coo = np.array([X, Y]).T
            itp = NearestNDInterpolator(coo, value)
        else:
            itp = None

        COMM.barrier()

        itp = COMM.bcast(itp, root=MASTER_RANK)
        cf_t = ___WRAPPER_SCALAR___(itp)

        mn = self._tf.space.mn
        indicator = self._tf.space.indicator

        if indicator == 'Lambda' and mn == (2, 2):
            k = self._tf.space.abstract.k

            if k == 0:
                cochain = reduce_Lambda__m2n2k0(cf_t, self._tf.tpm, self._tf.degree, element_range=element_range)

            elif k == 2:
                cochain = reduce_Lambda__m2n2k2(cf_t, self._tf.tpm, self._tf.degree, element_range=element_range)

            else:
                raise Exception()

        else:
            raise NotImplementedError()

        return cochain

    def ___renew_2d_vector___(self, X, Y, U, V, element_range):
        r""""""
        X, Y, U, V = merge_dict(X, Y, U, V)
        if RANK == MASTER_RANK:
            X = np.array(list(X.values()))
            Y = np.array(list(Y.values()))
            U = np.array(list(U.values()))
            V = np.array(list(V.values()))
            X = X.ravel()
            Y = Y.ravel()
            U = U.ravel()
            V = V.ravel()
            itp = NearestNDInterpolator(np.array([X, Y]).T, np.array([U, V]).T)
        else:
            itp = None

        COMM.barrier()

        itp = COMM.bcast(itp, root=MASTER_RANK)
        cf_t = ___WRAPPER_VECTOR_2d___(itp)

        mn = self._tf.space.mn
        indicator = self._tf.space.indicator

        if indicator == 'Lambda' and mn == (2, 2):
            k = self._tf.space.abstract.k
            orientation = self._tf.space.abstract.orientation
            if k == 1 and orientation == 'inner':
                cochain = reduce_Lambda__m2n2k1_inner(cf_t, self._tf.tpm, self._tf.degree, element_range=element_range)

            elif k == 1 and orientation == 'outer':
                cochain = reduce_Lambda__m2n2k1_outer(cf_t, self._tf.tpm, self._tf.degree, element_range=element_range)

            else:
                raise Exception()

        else:
            raise NotImplementedError()

        return cochain


___compare_two_meshes_cache___ = {
    'id0': 0,
    'id1': 0,
    'not_changed_elements': None,
    'changed_elements': None,
    'all_same': -1
}


def compare_two_meshes(elements0, elements1):
    r""""""
    if (id(elements0) == ___compare_two_meshes_cache___['id0'] and
            id(elements1) == ___compare_two_meshes_cache___['id1']):
        return (___compare_two_meshes_cache___['not_changed_elements'],
                ___compare_two_meshes_cache___['changed_elements'],
                ___compare_two_meshes_cache___['all_same'])
    else:
        ___compare_two_meshes_cache___['id0'] = id(elements0)
        ___compare_two_meshes_cache___['id1'] = id(elements1)

    assert isinstance(elements0, MseHttElementsPartialMesh) and isinstance(elements1, MseHttElementsPartialMesh)

    # -------------- first parse all not changed elements -----------------------------------------------

    all_element0 = []
    for ei in elements0:
        all_element0.append(ei)

    all_element1 = []
    for ei in elements1:
        all_element1.append(ei)

    ALL0 = COMM.gather(all_element0, root=MASTER_RANK)
    ALL1 = COMM.gather(all_element1, root=MASTER_RANK)
    if RANK == MASTER_RANK:
        all_element0 = []
        for _all in ALL0:
            all_element0.extend(_all)
        all_element1 = []
        for _all in ALL1:
            all_element1.extend(_all)
    else:
        del all_element0, all_element1

    del ALL0, ALL1

    COMM.barrier()

    if RANK == MASTER_RANK:
        not_changed_elements = []
        new_elements = []
        # noinspection PyUnboundLocalVariable
        for s1 in all_element1:
            # noinspection PyUnboundLocalVariable
            if s1 in all_element0:
                not_changed_elements.append(s1)
            else:
                new_elements.append(s1)

        # noinspection PyTypeChecker
        ___compare_two_meshes_cache___['not_changed_elements'] = not_changed_elements
        # noinspection PyTypeChecker
        ___compare_two_meshes_cache___['changed_elements'] = new_elements

        if not new_elements:
            all_same = 1
        else:
            all_same = 0

    else:
        not_changed_elements = None
        new_elements = None
        all_same = -1

    COMM.barrier()

    all_same = COMM.bcast(all_same, root=MASTER_RANK)
    ___compare_two_meshes_cache___['all_same'] = all_same

    return not_changed_elements, new_elements, all_same


def ___find_base_element_index___(e):
    r""""""
    if e[0] == '(':
        s_ind = e.split(')')[0][1:].split(',')[0]
    else:
        assert ':' in e, f"must be!"
        s_ind = e.split(':')[0]
    return int(s_ind)


_local_cache_1_ = {
    'id': 0,
    'indices': [],
}


def ___find_needed_base_element_indices___(changed_elements):
    r""""""
    ID = id(changed_elements)
    if ID == _local_cache_1_['id']:
        return _local_cache_1_['indices']
    else:
        indices = list()
        for e_ind in changed_elements:
            base_element_index = ___find_base_element_index___(e_ind)
            indices.append(base_element_index)

        _local_cache_1_['id'] = ID
        _local_cache_1_['indices'] = indices

        return indices


_local_cache_4_ = {
    'id': '',
    'bcast': None,
}


def ___bcast___(base_elements, data_id):
    r""""""
    if data_id == _local_cache_4_['id']:
        return _local_cache_4_['bcast']
    else:
        _local_cache_4_['id'] = data_id
        all_base_elements = COMM.bcast(base_elements, root=MASTER_RANK)
        _local_cache_4_['bcast'] = all_base_elements
        return all_base_elements


_local_cache_ff_ = {
    'id': '',
    'ff_reconstruction': [],
}


def ___find_ff_reconstruction_elements___(base_elements, elements):
    r""""""
    ID = str(id(base_elements)) + '=' + str(id(elements))
    if ID == _local_cache_ff_['id']:
        return _local_cache_ff_['ff_reconstruction']
    else:
        _local_cache_ff_['id'] = ID
        local_reconstruction_elements = list()
        for e in elements:
            base_ele_index = ___find_base_element_index___(e)
            # if RANK == 1:
            #     print(e, base_ele_index)
            if base_ele_index in base_elements:
                local_reconstruction_elements.append(e)
            else:
                pass
        _local_cache_ff_['ff_reconstruction'] = local_reconstruction_elements
        return local_reconstruction_elements


_local_cache_tf_ = {
    'id': '',
    'tf_reconstruction': [],
}


def ___find_tf_reconstruction_elements___(base_elements, elements):
    r""""""
    ID = str(id(base_elements)) + '=' + str(id(elements))
    if ID == _local_cache_tf_['id']:
        return _local_cache_tf_['tf_reconstruction']
    else:
        _local_cache_tf_['id'] = ID
        local_reconstruction_elements = list()
        for e in elements:
            base_ele_index = ___find_base_element_index___(e)
            if base_ele_index in base_elements:
                local_reconstruction_elements.append(e)
            else:
                pass
        _local_cache_tf_['tf_reconstruction'] = local_reconstruction_elements
        return local_reconstruction_elements
