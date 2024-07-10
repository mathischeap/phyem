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
        r"""

        Parameters
        ----------
        solver
        monitoring_factor:
            in [0, 1]. If it is 1, then monitor after each iteration. If it is 0, then the monitor
            frequency is very low.

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
        r"""The monitor of this iterator."""
        return self._monitor_

    @property
    def RDF(self):
        r"""(pandas.DataFrame) Result DataFrame."""
        return self._RDF

    @property
    def exit_code(self):
        r"""Return the exit code of the solver for the last run."""
        return self._exit_code_

    @exit_code.setter
    def exit_code(self, exit_code):
        r""""""
        # noinspection PyAttributeOutsideInit
        self._exit_code_ = exit_code

    @staticmethod
    def _exit_code_explanations(exit_code):
        r""""""
        explanations = {
            0: 'continue',
            1: 'normal stop'
        }
        return explanations[exit_code]

    @property
    def message(self):
        r"""List(str) Return the messages of the solver for the last run."""
        return self._message_

    @message.setter
    def message(self, message):
        r""""""
        if isinstance(message, str):
            message = [message, ]
        assert isinstance(message, (list, tuple)), "message must be str or list or tuple."
        for i, mi in enumerate(message):
            assert isinstance(mi, str), f"message must be tuple or list of str, " \
                                        f"now message[{i}]={mi} is not str"
        # noinspection PyAttributeOutsideInit
        self._message_ = message

    def _append_outputs_to_RDF(self, outputs):
        r""""""
        self.RDF.loc[len(self.RDF)] = outputs[2:]

    def test(self, test_range):
        r"""Do a test run of `times` iterations."""
        for args in test_range:
            if RANK == MASTER_RANK:
                print(f"=~= (TESTING) starts with input: <{args}>", flush=True)

            t_start = time()
            if hasattr(args, '__iter__'):
                results = self._solver_(*args)
            else:
                results = self._solver_(args)
            t_cost = time() - t_start

            if RANK == MASTER_RANK:
                print(f" ... leads to outputs: (cost %.3f) seconds\n" % t_cost)
                for i, res in enumerate(results):
                    print(f"  {i})-> {self._solver_ret[i]}: {res}")
                print('\n', flush=True)

    def run(self, *ranges, pbar=True):
        r"""To run the iterator.

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

        # -------------------------------------------------------------------------------
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
            desc = self.monitor.name
            if len(desc) > 25:
                desc = '...' + desc[-22:]
            else:
                pass
            if pbar:
                progress_bar = tqdm(
                    total=num_iterations,
                    desc='<' + desc + '>',
                )
            else:
                pass
        else:
            pass

        # -------- do the iterations ---------------------------------------------------------------
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
                if pbar:
                    # noinspection PyUnboundLocalVariable
                    progress_bar.update(1)
                else:
                    pass
            else:
                pass

            exit_code_explanation = self._exit_code_explanations(self.exit_code)
            if exit_code_explanation == 'continue':  # everything is fine, continue!
                pass
            elif exit_code_explanation == 'normal stop':  # Fine but enough, stop now.
                break
            else:
                raise NotImplementedError(f"cannot understand exit_code={self.exit_code}!")

        # --- after iteration ------------------------------------------------------------------------
        if RANK == MASTER_RANK:
            if pbar:
                progress_bar.close()
            else:
                pass
            print(flush=True)
            self.monitor.save(over=True)  # save to `.csv`
            self.monitor.report(over=True)  # make graphic report.
