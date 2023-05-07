# -*- coding: utf-8 -*-
from src.config import RANK, MASTER_RANK, SIZE, COMM
import inspect
import os
from tools.run.runners.base import ParallelRunnerBase
from tools.legacy.serialRunners.INSTANCES.matrix3d_input_runner import Matrix3dInputRunner
from tools.run.runners._3d_matrix_inputs.visualize.main import ___SlaveParallelMatrix3dInputRunnerVisualize___


class ParallelMatrix3dInputRunner(ParallelRunnerBase):
    """"""

    def __init__(self, solver):
        """"""
        super().__init__()

        self._solver_ = solver

        if RANK == MASTER_RANK:
            if solver.__class__.__name__ == 'Matrix3dInputRunner':
                # this is only for RunnerDataReader. Never use this manually
                self._SR_ = solver
            else:
                self._solver_source_code_ = inspect.getsource(solver)
                self._solver_dir_ = os.path.abspath(inspect.getfile(solver))
                self._SR_ = Matrix3dInputRunner(
                    solver,
                    solver_source_code=self._solver_source_code_,
                    solver_dir=self._solver_dir_
                )
        else:
            self._SR_ = None
            self._slave_visualize_ = ___SlaveParallelMatrix3dInputRunnerVisualize___(self)

        self._freeze()

    def ___iterate___(self, I1, I2, I3, criterion='standard', writeto=None, **kwargs):
        """

        :param I1: The 1st matrix to be passed to the solver.
        :param I2: The 2nd matrix to be passed to the solver.
        :param I3: The list or tuple of the 3rd variable to be passed to the solver
        :param criterion: The criterion of how to parse the inputs `I1`, `I2`, `I3`. It can be one of
            'standard' :
                The 'standard' criterion stands for that: `i0` and `i1` are main
                variables, and they do not change along `i2`. So we need `i0` and `i1`
                to be iterable. And each `i0[.]` or `i1[.]` need to be iterable.

                For example: `i0` and `i1`:
                    i0 = [[1, 2, 3],
                          [4, 5],
                          [6, 7, 8, 8]]
                    i1 = [[0.5, 0.1, 0.3],
                          [0, 2],
                          [3, 4, -2, -3]]
                    i2 = [0, 0.15, 0.2]
                Note that shape(i0[k]) must be equal to shape(i1[k]).

        :param writeto: The keywords variable to be passed to the solver.
        :param kwargs: The keywords variable to be passed to the solver. For a
            ParallelMatrix3dInputRunner, the same kwargs will be used for all runs.
            It is not possible to customize kwargs for each run.
        :return:
        """

        if SIZE == 1:
            self._SR_.iterate(I1, I2, I3, criterion=criterion, writeto=writeto, saveto=False, **kwargs)

        else:
            if RANK == MASTER_RANK:
                self._SR_.iterate(I1, I2, I3, criterion=criterion, writeto=writeto, saveto=False, **kwargs)
            else:
                I, J, K = COMM.recv(source=MASTER_RANK, tag=RANK + 1)  # position mark 1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                COMM.barrier()

                for k in range(K):  # we let the axis2 go at the last.
                    for i in range(I):  # we let the axis0 go secondly.
                        for j in range(J):  # we let the axis1 go firstly.

                            Compute_Or_Not = COMM.recv(source=MASTER_RANK, tag=RANK + 2)  # position mark 2 <<<<<<<
                            COMM.barrier()

                            if Compute_Or_Not:
                                INPUTS = COMM.recv(source=MASTER_RANK, tag=RANK + 3)  # position mark 3 <<<<<<<<<<<
                                COMM.barrier()
                                _ = self._solver_(INPUTS[0], INPUTS[1], INPUTS[2], **INPUTS[3])

                # we do nothing after all computation in slave cores -----------------------------

    @property
    def ___visualize___(self):
        if RANK == MASTER_RANK:
            return self._SR_.visualize
        else:
            return self._slave_visualize_

    @property
    def ___results___(self):
        if RANK == MASTER_RANK:
            R = self._SR_.rdf
        else:
            R = None

        if SIZE > 1:
            R = COMM.bcast(R, root=MASTER_RANK)
        else:
            pass

        return R
