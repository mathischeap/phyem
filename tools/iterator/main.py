# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from tools.frozen import Frozen
from tools.miscellaneous.numpy_styple import NumpyStyleDocstringReader
import inspect
import psutil
from src.config import RANK, MASTER_RANK

from tools.iterator.monitor import IteratorMonitor


class Iterator(Frozen):
    """"""
    def __init__(
            self,
            solver,
            initials,
            monitoring_factor=0.6,
            name=None,
    ):
        """

        Parameters
        ----------
        solver
        monitoring_factor
        name
        initials
        """
        self._monitoring_factor = monitoring_factor
        self._name = name
        self._cpu_load = 0
        ds = NumpyStyleDocstringReader(solver)
        solver_par = ds.Parameters
        solver_ret = ds.Returns
        assert solver_ret[0] == 'exit_code', "First output must be `exit_code`."
        assert solver_ret[1] == 'message', "Second output must be `message`."
        assert len(solver_ret) > 2, "need outputs."
        self._num_inputs = len(solver_par)
        self._num_outputs = len(solver_ret)
        self._solver_ret = solver_ret
        if initials is None:
            initials = list()
            for _ in range(self._num_outputs - 2):
                initials.append(np.NAN)
        else:
            pass
        assert len(initials) == self._num_outputs - 2, \
            f" len(initials) = {len(initials)} != (len(solver_ret) - 2)={self._num_outputs - 2}"
        self._initials = initials
        self._exit_code_ = 0
        self._message_ = ''
        self._solver_ = solver
        self._solver_source_code_ = inspect.getsource(solver)
        self._solver_dir_ = os.path.abspath(inspect.getfile(solver))
        self._monitor_ = None
        self._RDF = None
        self._freeze()

    @property
    def monitor(self):
        """The monitor of this iterator."""
        return self._monitor_

    @property
    def RDF(self):
        """(pandas.DataFrame) Result DataFrame."""
        return self._RDF

    @property
    def exit_code(self):
        """Return the exit code of the solver for the last run."""
        return self._exit_code_

    @exit_code.setter
    def exit_code(self, exit_code):
        # noinspection PyAttributeOutsideInit
        self._exit_code_ = exit_code

    @property
    def message(self):
        """List(str) Return the messages of the solver for the last run."""
        return self._message_

    @message.setter
    def message(self, message):
        if isinstance(message, str):
            message = [message, ]
        assert isinstance(message, (list, tuple)), "message must be str or list or tuple."
        for i, mi in enumerate(message):
            assert isinstance(mi, str), f"message must be tuple or list of str, " \
                                        f"now message[{i}]={mi} is not str"
        # noinspection PyAttributeOutsideInit
        self._message_ = message

    def _append_outputs_to_RDF(self, outputs):
        """"""
        self.RDF.loc[len(self.RDF)] = outputs[2:]

    def test(self, *args):
        """Do a test run of `times` iterations."""
        results = self._solver_(*args)
        print(f"test with arguments {args} leads to results: {results}.")

    def run(self, *ranges):
        """To run the iterator.

        To make it work properly, we have to make sure the solver return exactly the same
        outputs in all cores. Otherwise, cores may stop after different time steps. This very
        cause some serious problems.
        """
        # ---- for each run, we initialize a new monitor and a new RDF.
        if RANK == MASTER_RANK:
            self._monitor_ = IteratorMonitor(
                self,
                self._monitoring_factor,
                self._name
            )
            rdf_ret = self._solver_ret[2:]
            RDF = dict()
            for k, key in enumerate(rdf_ret):
                RDF[key] = self._initials[k]
            self._RDF = pd.DataFrame(RDF, index=[0, ])
        else:
            self._monitor_ = None
            self._RDF = None

        # --------
        num_iterations = None
        for i, rg in enumerate(ranges):
            assert hasattr(rg, '__iter__'), f"{i}th range={rg} is not iterable."
            if num_iterations is None:
                num_iterations = len(rg)
            else:
                assert num_iterations == len(rg), f"range length does not match."

        if RANK == MASTER_RANK:
            self.monitor._num_iterations = num_iterations  # update num_iterations for the monitor.
            self.monitor._start_time = time()
            pbar = tqdm(
                total=num_iterations,
                desc='<' + self.monitor.name + '>',
            )

        for inputs in zip(*ranges):

            if RANK == MASTER_RANK:
                psutil.cpu_percent(None)
                self.monitor._measure_start()

            outputs = self._solver_(*inputs)
            assert len(outputs) == self._num_outputs, f"amount of outputs wrong!"
            self.exit_code, self.message = outputs[:2]

            if RANK == MASTER_RANK:
                self._cpu_load = psutil.cpu_percent(None)
                self._append_outputs_to_RDF(outputs)
                self.monitor._measure_end()
                # noinspection PyUnboundLocalVariable
                pbar.update(1)

        if RANK == MASTER_RANK:
            pbar.close()
            print(flush=True)
            self.monitor.save()  # save to `.csv`
            self.monitor.report(over=True)  # make graphic report.
