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
# from scipy.interpolate import LinearNDInterpolator


class MseHtt_FormCochainRenew_Method1_GlobalInterpolation(Frozen):
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
        n = self._ff.space.n
        total_num_samples = 3753
        samples = int(total_num_samples ** (1/n) * ddf) + 1

        if samples < 32:
            samples = 32
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

        for t in cochain_times_tobe_renewed:
            self.___renew_a_cochain___(t, NODES)

        if clean:
            self._ff.cochain.clean('all')
        else:
            pass

    # noinspection PyUnboundLocalVariable
    def ___renew_a_cochain___(self, t, NODES):
        r""""""
        ff = self._ff[t]
        rec = ff.reconstruct(*NODES, ravel=True)
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
            value = VAL[0]
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

        if case == '2d-scalar':
            self.___renew_2d_scalar___(t, X, Y, value)
        elif case == '2d-vector':
            self.___renew_2d_vector___(t, X, Y, U, V)
        else:
            raise NotImplementedError()

    def ___renew_2d_scalar___(self, t, X, Y, value):
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
            # itp = LinearNDInterpolator(coo, value)
        else:
            itp = None

        itp = COMM.bcast(itp, root=MASTER_RANK)

        self._tf[t].reduce(___WRAPPER_SCALAR___(itp))

    def ___renew_2d_vector___(self, t, X, Y, U, V):
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

        itp = COMM.bcast(itp, root=MASTER_RANK)
        self._tf[t].reduce(___WRAPPER_VECTOR_2d___(itp))


class ___WRAPPER_SCALAR___(Frozen):
    r""""""
    def __init__(self, itp):
        r""""""
        self._itp_ = itp
        self._freeze()

    def __call__(self, *args):
        return (self._itp_(*args), )


class ___WRAPPER_VECTOR_2d___(Frozen):
    r""""""
    def __init__(self, itp):
        r""""""
        self._itp_ = itp
        self._freeze()

    def __call__(self, *args):
        val = self._itp_(*args)
        return val[..., -2], val[..., -1]
