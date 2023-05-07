# -*- coding: utf-8 -*-
"""Parallel runners.
"""
from src.config import COMM, RANK, MASTER_RANK, SIZE
from tools.legacy.serialRunners.INSTANCES.matrix3d_input_runner import Matrix3dInputRunner

from tools.run.runners._3d_matrix_inputs.main import ParallelMatrix3dInputRunner


def RunnerDataReader(filename):
    """
    To do such job, we do not need multiple cores. One master core can do all things as we basically just read a file.

    1). We will not be able to retrieve all information, but just obtain enough data for, like, visualization.
    Therefore, we are not able to restart a task with the data reader.

    2). Whenever we code a new Runner, we need to include it in this function.

    :param filename: The name of the file to be read.
    :return: An instance of the correct runner class.
    """
    if RANK == MASTER_RANK:
        with open(filename, 'r') as f:
            # so the header of a runner file must be like: <Runner>-<A particular runner classname>-<......>
            contents = f.readlines()
            c0 = contents[0]
            assert c0[1:7] == 'Runner' and '>-<' in c0 and c0.split('>-<')[1] != '', \
                f" <Runner> : it is not a Runner file! Its header is {c0} which does not match the standard format: " \
                f"<Runner>-<...A particular runner classname...>-<...something whatever else...>"
            runner_name = c0.split('>-<')[1]
    else:
        runner_name = None

    if SIZE > 1:
        runner_name = COMM.bcast(runner_name, root=MASTER_RANK)

    if runner_name == 'Matrix3dInputRunner':
        # Actually we read it with ``ParallelMatrix3dInputRunner``. Since the parallel version is extended from a serial
        # version, so something is old. And the way of reading is also special.
        if RANK == MASTER_RANK:
            SR = Matrix3dInputRunner.readfile(filename)
        else:
            SR = None
        RDO = ParallelMatrix3dInputRunner(SR)
    else:
        raise NotImplementedError(f"No class <{runner_name}> to read to data file: {filename}.")

    RDO.___lock_iterate___ = True  # we lock the runner such that it can not ``iterate`` as it does not have a solver.

    return RDO  # RDO stands for: `Runner` but with `Data` `Only`.
