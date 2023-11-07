# -*- coding: utf-8 -*-
r"""MPI version of solving with python.
"""
import numpy as np
from tools.frozen import Frozen
from msepy.tools.nonlinear_system.static.solve.Newton_Raphson import _check_stop_criterion_
from src.config import RANK, MASTER_RANK, SIZE, COMM, MPI


class LinerSystemSolverDivergenceError(Exception):
    """Raise when we try to define new attribute for a frozen object."""


class _PY_MPI_solvers(Frozen):
    """"""
    def __init__(self):
        self._freeze()

    @staticmethod
    def gmres(*args, **kwargs):
        return _gmres(*args, **kwargs)

    @staticmethod
    def lgmres(*args, **kwargs):
        return _lgmres(*args, **kwargs)


def _gmres(
        A, b, x0,
        m=75, maxiter=25, atol=1e-4
):
    """

    Parameters
    ----------
    A
    b
    x0
    m :
        Another name of `restart`.
    maxiter
    atol

    Returns
    -------
    x
    message
    info

    """
    restart = m
    local_ind = list()
    for i in range(restart):
        if (i % SIZE) == RANK:
            local_ind.append(i)
        else:
            pass
    Time_start = MPI.Wtime()
    A = A.M    # A is a shell of the distributed global matrix
    f = b.V    # b is a shell of the distributed global vector

    shape0, shape1 = A.shape
    assert f.shape[0] == x0.shape[0] == shape0 == shape1, "Ax=f shape dis-match."

    ITER = 0
    BETA = None

    AVJ = np.empty((shape0,), dtype=float)
    Hm = np.zeros((restart + 1, restart), dtype=float)
    # In the future, we can change this to sparse matrix.

    if RANK == MASTER_RANK:
        VV = np.empty((shape0,), dtype=float)
        Vm = np.empty((restart, shape0), dtype=float)
        HM = np.empty((restart + 1, restart), dtype=float)  # use to combine all Hm.
        SUM_Hij_vi = np.empty((shape0,), dtype=float)
        Vs = None
        local_ind_dict = None
    else:
        local_ind_dict = dict()
        for i, ind in enumerate(local_ind):
            local_ind_dict[ind] = i
        VV = None
        HM = None
        Vm = None
        SUM_Hij_vi = None
        Vs = np.empty((len(local_ind), shape0), dtype=float)

    # ------------- iteration start -----------------------------------------------------------
    while 1:  # always do till break.

        v0 = f - A @ x0
        COMM.Reduce(v0, VV, op=MPI.SUM, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            beta = np.sum(VV ** 2) ** 0.5
            v0 = VV / beta
        else:
            beta = None

        COMM.Bcast(v0, root=MASTER_RANK)
        beta = COMM.bcast(beta, root=MASTER_RANK)

        if BETA is None:
            BETA = [beta, ]  # this is right, do not initial BETA as an empty list.
        if len(BETA) > 20:
            BETA = BETA[:1] + BETA[-2:]
        else:
            pass
        BETA.append(beta)

        JUDGE, stop_iteration, info, JUDGE_explanation = _check_stop_criterion_(BETA, atol, ITER, maxiter)
        if stop_iteration:
            break
        else:
            pass

        if RANK == MASTER_RANK:
            Vm[0] = v0
        else:
            Vm = v0
            if 0 in local_ind:
                Vs[local_ind_dict[0]] = v0

        for j in range(restart):

            if RANK == MASTER_RANK:
                Avj = A @ Vm[j]
            else:
                Avj = A @ Vm
                if j in local_ind:
                    Vs[local_ind_dict[j]] = Vm

            COMM.Allreduce(Avj, AVJ, op=MPI.SUM)

            sum_Hij_vi = np.zeros((shape0,), dtype=float)

            if RANK == MASTER_RANK:
                for i in local_ind:
                    if i <= j:
                        _ = Vm[i]
                        Hij = np.sum(AVJ * _)
                        sum_Hij_vi += Hij * _
                        Hm[i, j] = Hij
                    else:
                        break

            else:
                for i in local_ind:
                    if i <= j:
                        _ = Vs[local_ind_dict[i]]
                        Hij = np.sum(AVJ * _)
                        sum_Hij_vi += Hij * _
                        Hm[i, j] = Hij
                    else:
                        break

            COMM.Reduce(sum_Hij_vi, SUM_Hij_vi, op=MPI.SUM, root=MASTER_RANK)

            if RANK == MASTER_RANK:
                hat_v_jp1 = AVJ - SUM_Hij_vi
                Hm[j + 1, j] = np.sum(hat_v_jp1 ** 2) ** 0.5

                if j < restart - 1:
                    v_jp1 = hat_v_jp1 / Hm[j+1, j]
                else:
                    del v_jp1

            else:
                if j == 0:
                    v_jp1 = np.empty((shape0,), dtype=float)
                else:
                    pass

            if j < restart - 1:
                # noinspection PyUnboundLocalVariable
                COMM.Bcast([v_jp1, MPI.DOUBLE], root=MASTER_RANK)
                if RANK == MASTER_RANK:
                    Vm[j+1] = v_jp1
                else:
                    Vm = v_jp1

        COMM.Reduce(Hm, HM, op=MPI.SUM, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            HMT = HM.T
            ls_A = HMT @ HM
            ls_b = HMT[:, 0] * beta
            ym = np.linalg.solve(ls_A, ls_b)
            # del HMT, ls_A, ls_b
            x0 += ym.T @ Vm

        COMM.Bcast(x0, root=MASTER_RANK)
        ITER += 1

    # ============ iteration done ===========================================================

    if info < 0:
        raise LinerSystemSolverDivergenceError(
            f"gmres0 diverges after {ITER} iterations with error reaching {beta}.")

    Time_end = MPI.Wtime()

    COST_total = Time_end - Time_start
    message = f" mpi_v2_gmres = [SYSTEM-SHAPE: {A.shape}] [ITER={ITER}]" \
        f" [residual=%.2e] costs %.2f, " \
        f"convergence info={info}, restart={restart}, maxiter={maxiter}, " \
        f"stop_judge={JUDGE}: {JUDGE_explanation}]" % (beta, COST_total)

    return x0, message, info


def _lgmres(
        A, b, x0,
        m=75, k=15, maxiter=25, atol=1e-4,
):
    """

    Parameters
    ----------
    A
    b
    x0
    m
    k
    maxiter
    atol

    Returns
    -------
    x
    message
    info

    """
    _m_, _k_ = m, k
    restart = m + k

    local_ind = list()
    for i in range(restart):
        if (i % SIZE) == RANK:
            local_ind.append(i)

    Time_start = MPI.Wtime()
    A = A.M    # A is a shell of the distributed global matrix
    f = b.V    # b is a shell of the distributed global vector

    shape0, shape1 = A.shape
    assert f.shape[0] == x0.shape[0] == shape0 == shape1, "Ax=f shape dis-match."

    ITER = 0
    BETA = None

    AVJ = np.empty((shape0,), dtype=float)
    Hm = np.zeros((restart + 1, restart), dtype=float)
    # In the future, we can change this to sparse matrix.
    if RANK == MASTER_RANK:
        VV = np.empty((shape0,), dtype=float)
        Vm = np.empty((restart, shape0), dtype=float)

        SUM_Hij_vi = np.empty((shape0,), dtype=float)
        Vs = None
        local_ind_dict = None

    else:
        local_ind_dict = dict()
        for i, ind in enumerate(local_ind):
            local_ind_dict[ind] = i

        VV = None
        SUM_Hij_vi = None
        Vs = np.empty((len(local_ind), shape0), dtype=float)
        Vm = None

    if RANK == MASTER_RANK:
        HM = np.empty((restart + 1, restart), dtype=float)  # use to combine all Hm.
    else:
        HM = None

    ZZZ: dict = dict()
    AZ_cache = dict()

    # ------------- iteration start -----------------------------------------------------------
    while 1:  # always do till break.

        if ITER < _k_:
            k = ITER
            m = restart - k
        else:
            m = _m_
            k = _k_

        v0 = f - A @ x0
        COMM.Reduce(v0, VV, op=MPI.SUM, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            beta = np.sum(VV ** 2) ** 0.5
            v0 = VV / beta
        else:
            beta = None

        COMM.Bcast(v0, root=MASTER_RANK)
        beta = COMM.bcast(beta, root=MASTER_RANK)

        # check stop iteration or not ...
        if BETA is None:
            BETA = [beta, ]  # this is right, do not initial BETA as an empty list.
        if len(BETA) > 20:
            BETA = BETA[:1] + BETA[-2:]
        BETA.append(beta)

        JUDGE, stop_iteration, info, JUDGE_explanation = _check_stop_criterion_(BETA, atol, ITER, maxiter)
        if stop_iteration:
            break
        else:
            pass

        if RANK == MASTER_RANK:
            Vm[0] = v0
        else:
            Vm = v0
            if 0 in local_ind:
                Vs[local_ind_dict[0]] = v0

        for j in range(restart):

            if j < m:
                if RANK == MASTER_RANK:
                    Avj = A @ Vm[j]
                else:
                    Avj = A @ Vm
                    if j in local_ind:
                        Vs[local_ind_dict[j]] = Vm
            else:
                index = ITER + m - 1 - j

                if index in AZ_cache:
                    pass
                else:
                    if ITER > _k_:
                        del AZ_cache[ITER - _k_ - 1]
                    AZ_cache[index] = A @ ZZZ[index]

                Avj = AZ_cache[index]

                if RANK == MASTER_RANK:
                    pass
                else:
                    if j in local_ind:
                        Vs[local_ind_dict[j]] = Vm

            COMM.Allreduce(Avj, AVJ, op=MPI.SUM)

            sum_Hij_vi = np.zeros((shape0,), dtype=float)

            if RANK == MASTER_RANK:
                for i in local_ind:
                    if i <= j:
                        _ = Vm[i]
                        Hij = np.sum(AVJ * _)
                        sum_Hij_vi += Hij * _
                        Hm[i, j] = Hij
                    else:
                        break
            else:
                for i in local_ind:
                    if i <= j:
                        _ = Vs[local_ind_dict[i]]
                        Hij = np.sum(AVJ * _)
                        sum_Hij_vi += Hij * _
                        Hm[i, j] = Hij
                    else:
                        break

            COMM.Reduce(sum_Hij_vi, SUM_Hij_vi, op=MPI.SUM, root=MASTER_RANK)

            if RANK == MASTER_RANK:
                hat_v_jp1 = AVJ - SUM_Hij_vi
                Hm[j + 1, j] = np.sum(hat_v_jp1 ** 2) ** 0.5

                if j < restart - 1:
                    v_jp1 = hat_v_jp1 / Hm[j + 1, j]
                else:
                    del v_jp1

            else:
                if j == 0:
                    v_jp1 = np.empty((shape0,), dtype=float)
                else:
                    pass

            if j < restart - 1:
                # noinspection PyUnboundLocalVariable
                COMM.Bcast([v_jp1, MPI.DOUBLE], root=MASTER_RANK)
                if RANK == MASTER_RANK:
                    Vm[j + 1] = v_jp1
                else:
                    Vm = v_jp1

        COMM.Reduce(Hm, HM, op=MPI.SUM, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            HMT = HM.T
            ls_A = HMT @ HM
            ls_b = HMT[:, 0] * beta
            ym = np.linalg.solve(ls_A, ls_b)
        else:
            pass

        if RANK == MASTER_RANK:
            if k == 0:
                Ws = Vm
            else:
                DL = list()
                iL = [ITER - _ - 1 for _ in range(k)]
                for _ in iL:
                    DL.append(ZZZ[_])
                Ws = np.vstack((Vm[:m], np.array(DL)))
        else:
            pass

        if RANK == MASTER_RANK:
            # noinspection PyUnboundLocalVariable
            ym = ym.T @ Ws
        else:
            ym = np.empty((shape0,), dtype=float)
            # Important: renew ``ym`` every single iteration.

        COMM.Bcast([ym, MPI.DOUBLE], root=MASTER_RANK)

        if ITER >= _k_ > 0:
            del ZZZ[ITER-_k_]
        ZZZ[ITER] = ym

        x0 += ym
        ITER += 1

    # ============ iteration done ===========================================================

    if info < 0:
        raise LinerSystemSolverDivergenceError(
            f"lGMRES_0 diverges after {ITER} iterations with error reaching {beta}.")

    Time_end = MPI.Wtime()

    COST_total = Time_end - Time_start
    message = f" mpi_v0_LGMRES = [SYSTEM-SHAPE: {A.shape}] [ITER={ITER}] " \
        f"[residual=%.2e] costs %.2f, " \
        f"convergence info={info}, m={_m_}, k={_k_}, maxiter={maxiter}, " \
        f"stop_judge={JUDGE}: {JUDGE_explanation}]" % (beta, COST_total)

    return x0, message, info
