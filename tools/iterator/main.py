# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time, sleep
from tools.frozen import Frozen
from tools.miscellaneous.numpy_styple import NumpyStyleDocstringReader
import inspect
import psutil
from src.config import RANK, MASTER_RANK, COMM

from tools.iterator.monitor import IteratorMonitor

if RANK == MASTER_RANK:
    import pickle
    from tools.miscellaneous.timer import MyTimer
else:
    pass


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
                initials.append(np.nan)
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

        self._cache_objs_ = None
        if RANK == MASTER_RANK:
            self._cache_filename_ = None
            self._cache_time_ = time()
            self.___cache_time___ = None

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

    def test(self, test_range, show_info=True):
        r"""Do a test run of `times` iterations."""
        all_test_results = list()
        for args in test_range:
            if show_info:
                if RANK == MASTER_RANK:
                    print(f"=~= (TESTING) starts with input: <{args}>", flush=True)
                else:
                    pass
            else:
                pass

            t_start = time()
            if hasattr(args, '__iter__'):
                results = self._solver_(*args)
            else:
                results = self._solver_(args)
            t_cost = time() - t_start

            if show_info:
                if RANK == MASTER_RANK:
                    print(f" ... leads to outputs: (cost %.3f) seconds\n" % t_cost)
                    if not hasattr(results, '__iter__'):
                        print(f"  Results: {results}")
                    else:
                        for i, res in enumerate(results):
                            print(f"  {i})-> {self._solver_ret[i]}: {res}")
                    print('\n', flush=True)
                else:
                    pass
            else:
                pass

            all_test_results.append(results)

        return all_test_results

    def cache(self, *cache_objs, cache_filename=None, time=None):
        r"""

        Parameters
        ----------
        cache_objs :
            The objects to be cached. If we wanna an object to be cache-able, we should first let it have
            a property `name`. Then it should have two methods: `_make_cache_data` and `_read_cache_data`.

        cache_filename
        time

        Returns
        -------

        """
        if cache_filename is None:
            cache_filename = self._name
        else:
            pass
        assert isinstance(cache_filename, str) and '.' not in cache_filename, \
            f"cache_filename={cache_filename} illegal. It must be str of no '.' in it."
        cache_filename += '.phc'
        self._cache_objs_ = cache_objs
        if RANK == MASTER_RANK:
            self._cache_filename_ = cache_filename
            if os.path.isfile(self._cache_filename_):  # we find an existing phyem cache file
                pass
            else:  # we did not find an existing phyem cache file, so we make an empty one.
                obj_names = list()
                for obj in cache_objs:
                    obj_names.append(obj.name)

                cache_dict = {
                    'obj names': obj_names,
                    'computed inputs': [[], [], []],
                    'RES': {},
                    'cache data': None,  # must start with None
                    'log': '',
                    'cache count': 0,
                }

                with open(cache_filename, 'wb') as output:
                    pickle.dump(cache_dict, output, pickle.HIGHEST_PROTOCOL)
                output.close()

            # We check the cache file
            self._check_cache(self._cache_filename_)
            if time is None:    # default caching time: 1 hour
                self.___cache_time___ = 3600
            elif time == np.inf:
                self.___cache_time___ = np.inf
            else:
                assert isinstance(time, (int, float)) and time > 0, \
                    f"time={time} wrong, must be positive number."
                self.___cache_time___ = time
        else:
            pass

    def _check_cache(self, cache_filename):
        r"""Check whether the cache file is of the correct format.

        If this method is about to be called in different ranks, make sure that it is not called at
        the same time.
        """
        with open(cache_filename, 'rb') as inputs:
            cache = pickle.load(inputs)
        inputs.close()
        assert isinstance(cache, dict), f"phyem cache file must be a dict."
        obj_names = cache['obj names']
        assert 'computed inputs' in cache, "cache has no 'computed range'"
        assert 'RES' in cache, "cache has no 'RES'"
        assert 'cache data' in cache, "cache has no 'cache data'"
        for i, obj in enumerate(self._cache_objs_):
            assert obj.name == obj_names[i], f"obj:{obj.name} is not in the cache."

    def _decoding_objs_(self, all_obj_data):
        r"""`_read_cache_data` must could take data only from the master rank."""
        if RANK == MASTER_RANK:
            if all_obj_data is None:
                do_it = False
            else:
                do_it = True
        else:
            do_it = None
        do_it = COMM.bcast(do_it, root=MASTER_RANK)

        if do_it:
            if RANK == MASTER_RANK:
                for data, obj in zip(all_obj_data, self._cache_objs_):
                    obj._read_cache_data(data)
            else:
                for obj in self._cache_objs_:
                    obj._read_cache_data(None)
        else:
            pass

    def _coding_objs_(self):
        r"""`_make_cache_data` must return all necessary data in the master rank."""
        all_obj_data = list()
        for obj in self._cache_objs_:
            data = obj._make_cache_data(t=None)   # only coding the newest t of objs.
            all_obj_data.append(data)
        return all_obj_data

    @staticmethod
    def _make_inputs_key(inputs):
        """Return `Ik`: Can only be int or str!"""
        if len(inputs) == 1:
            i = inputs[0]
            if isinstance(i, int):
                return i
            elif isinstance(i, float):
                if i % 1 == 0:
                    i = int(i)
                else:
                    i = "%.5f" % i
                return i
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    @staticmethod
    def _check_input_cache(computed_inputs, Ik):
        r""""""
        range_list, int_list, str_list = computed_inputs
        if isinstance(Ik, int):
            if Ik in int_list:
                return True
            else:
                for RANGE in range_list:
                    if Ik in RANGE:
                        return True
                    else:
                        pass
                return False
        elif isinstance(Ik, str):
            return Ik in str_list
        else:
            raise Exception(f"Ik={Ik} illegal, make it a int or str.")

    @staticmethod
    def _update_computed_inputs(computed_inputs, Ik):
        r""""""
        range_list, int_list, str_list = computed_inputs
        if isinstance(Ik, int):
            if Ik in int_list:
                pass
            else:
                in_or_out = False
                for RANGE in range_list:
                    if Ik in RANGE:
                        in_or_out = True
                        break
                    else:
                        pass
                if in_or_out:  # find it in a range
                    pass
                else:
                    int_list.append(Ik)
        elif isinstance(Ik, str):
            if Ik in str_list:
                pass
            else:
                str_list.append(Ik)
        else:
            raise Exception(f"Ik={Ik} illegal, make it a int or str.")

    @staticmethod
    def _compress_computed_input_list(computed_inputs):
        r"""compress the size of this list, so make range from int_list.

        [range_list, int_list, str_list]
        """
        range_list, int_list, str_list = computed_inputs
        int_list.sort()

        to_be_removed = list()
        for i in int_list:
            for r, RANGE in enumerate(range_list):
                if i == RANGE.stop:
                    range_list[r] = range(RANGE.start, i+1)
                    to_be_removed.append(i)
                    break
                elif i == RANGE.start - 1:
                    range_list[r] = range(i, RANGE.stop)
                    to_be_removed.append(i)
                    break
                else:
                    pass

        for i in to_be_removed:
            int_list.remove(i)

        new_range = None
        for i in int_list:
            if i + 1 in int_list:
                new_range = range(i, i+2)   # only store step = 1 ranges
                int_list.remove(i)
                int_list.remove(i+1)
                break

        if new_range is None:
            pass
        else:
            range_list.append(new_range)

    def run(self, *ranges, pbar=True):
        r"""To run the iterator.
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

        computed_inputs = None

        if self._cache_objs_ is None:
            pass
        else:
            keep_checking = True
            if RANK == MASTER_RANK:
                with open(self._cache_filename_, 'rb') as cf:
                    cache = pickle.load(cf)
                cf.close()
                obj_names = cache['obj names']
                computed_inputs = cache['computed inputs']
                cache_res = cache['RES']
                cache_data = cache['cache data']
                if 'log' in cache:
                    log = cache['log']
                else:
                    log = ''
                if 'cache count' in cache:
                    cache_count = cache['cache count']
                else:
                    cache_count = 0
            else:
                cache_data = None
                None_outputs = [None for _ in range(self._num_outputs-2)]
            computed_inputs = COMM.bcast(computed_inputs, root=MASTER_RANK)

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
        caching = False

        # -------- do the iterations ---------------------------------------------------------------
        for inputs in zip(*ranges):

            if RANK == MASTER_RANK:
                psutil.cpu_percent(None)
                self.monitor._measure_start()

            COMM.barrier()
            if self._cache_objs_ is None:
                outputs = self._solver_(*inputs)
            else:
                Ik = self._make_inputs_key(inputs)
                # noinspection PyUnboundLocalVariable
                if keep_checking:
                    # noinspection PyUnboundLocalVariable
                    if self._check_input_cache(computed_inputs, Ik):
                        if RANK == MASTER_RANK:
                            if pbar:
                                pass
                            else:
                                # noinspection PyUnboundLocalVariable
                                print(desc + f' <--- input: {Ik} <--- cache', flush=True)
                            # noinspection PyUnboundLocalVariable
                            res = cache_res[Ik]
                            exit_cods = res[0]
                            outputs = [exit_cods, 'cached'] + res[1:]
                        else:
                            exit_cods = -1
                        exit_cods = COMM.bcast(exit_cods, root=MASTER_RANK)
                        if RANK != MASTER_RANK:
                            # noinspection PyUnboundLocalVariable
                            outputs = [exit_cods, ''] + None_outputs
                        else:
                            pass
                    else:
                        keep_checking = False
                        caching = True
                        # noinspection PyUnboundLocalVariable
                        self._decoding_objs_(cache_data)
                        outputs = self._solver_(*inputs)
                        del cache_data
                        if RANK != MASTER_RANK:
                            del computed_inputs
                        else:
                            pass
                    sleep(0.01)
                else:
                    outputs = self._solver_(*inputs)

            # noinspection PyUnboundLocalVariable
            assert len(outputs) == self._num_outputs, f"amount of outputs wrong!"
            self.exit_code, self.message = outputs[:2]

            if caching:
                if RANK == MASTER_RANK:
                    # noinspection PyUnboundLocalVariable
                    if Ik not in cache_res:
                        exit_code = outputs[0]
                        cache_res[Ik] = [exit_code, ] + list(outputs[2:])
                    else:
                        pass
                    self._update_computed_inputs(computed_inputs, Ik)
                    self._compress_computed_input_list(computed_inputs)
                    now = time()
                    cache_waiting_time = now - self._cache_time_
                    do_cache = cache_waiting_time > self.___cache_time___  # every `___cache_time___` do a cache
                else:
                    do_cache = False
                do_cache = COMM.bcast(do_cache, root=MASTER_RANK)
                if do_cache:
                    obj_coding_data = self._coding_objs_()
                    if RANK == MASTER_RANK:
                        if pbar:
                            pass
                        else:
                            print("\n---------------------------------------------------------")
                            print("\n=========================================================\n")
                            print(desc + f'-> cache @' + MyTimer.current_time(), flush=True)
                            print("\n=========================================================")
                            print("\n---------------------------------------------------------\n\n")
                        # noinspection PyUnboundLocalVariable
                        log = self._make_cache_log(log, Ik, cache_count, cache_res, computed_inputs)
                        # noinspection PyUnboundLocalVariable
                        cache_dict = {
                            'obj names': obj_names,
                            'computed inputs': computed_inputs,
                            'RES': cache_res,
                            'cache data': obj_coding_data,
                            'log': log,
                            'cache_count': cache_count + 1
                        }
                        with open(self._cache_filename_, 'wb') as cf:
                            pickle.dump(cache_dict, cf, pickle.HIGHEST_PROTOCOL)
                        cf.close()
                        # noinspection PyUnboundLocalVariable
                        self._cache_time_ = now
                        del obj_coding_data, cache_dict
                    else:
                        del obj_coding_data
                else:
                    pass
            else:
                pass

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
        else:
            pass

    def _make_cache_log(self, log, Ik, cache_count, cache_res, computed_inputs):
        r"""update the log string."""
        time_str = MyTimer.current_time()
        cache_count += 1
        amount_data = len(cache_res)
        log += f'-----------------------------------------------------\n'
        log += f'CACHE::: {cache_count}th cache @{time_str}->newest input-key: {Ik}\n'
        log += f"PATH:::: {self.monitor.name}\n"
        log += f'OBJECTS:\n'
        for i, obj in enumerate(self._cache_objs_):
            log += f'~ {i}th: {obj.name}\n'
        log += '>>>>>> details __________\n'
        log += f'    - {amount_data} items cached.\n'
        range_list, int_list, str_list = computed_inputs
        len_range = len(range_list)
        len_int = len(int_list)
        len_str = len(str_list)
        if len_int + len_str < 50 and len_range < 10:
            range_items = str(range_list)
            int_items = str(int_list)
            str_items = str(str_list)
            log += f"    - cached items: \n"
            log += f"        *ranges: {range_items}\n"
            log += f"        *int: {int_items}\n"
            log += f"        *str: {str_items}\n"
        elif len_range < 10:
            range_items = str(range_list)
            log += f"    - cached items: \n"
            log += f"        *ranges: {range_items}\n"
            if len_int < 25:
                int_items = str(int_list)
                log += f"        *int: {int_items}\n"
            else:
                log += f"        *int: {len_int} items\n"
            if len_str < 25:
                str_items = str(str_list)
                log += f"        *str: {str_items}\n"
            else:
                log += f"        *str: {len_str} items\n"
        else:
            log += f"    - cached items: \n"
            log += f"        *ranges: {len_range} items\n"
            log += f"        *int: {len_int} items\n"
            log += f"        *str: {len_str} items\n"

        log += '\n'
        return log
