# -*- coding: utf-8 -*-
"""
Like the `ThreeInputsRunner`, but here we do not do the meshgrid. We put the three 
inputs: i0, i1, i2, into three 3-d matrices (3d arrays). And we go through all three
directions to run the solver.

if we want to skip one run, we just put there a `None` (in three matrices). 

<unittest> <unittests_P_Solvers> <test_No4_M3IR>.

Yi Zhang (C)
Created on Thu Apr 11 10:29:08 2019
Aerodynamics, AE
TU Delft
"""
import types
from tqdm import tqdm
from time import localtime, strftime, time, sleep
from tools.legacy.serialRunners._runner_ import Runner
from tools.legacy.serialRunners.INSTANCES.COMPONENTS.m_tir_tabular import M_TIR_Tabulate
from src.config import MASTER_RANK, RANK, SIZE, COMM


class TimeIteration:
    """ We use this contextmanager to time an iteration. """

    def __init__(self, m, num_iterations, total_cost_list, print_time):
        """
        Parameters
        ----------
        m : int
            Currently, it is mth iteration.
        num_iterations : int
            Total amount of iterations.
        total_cost_list : list

        """
        self.m = m
        self.num_iterations = num_iterations
        self.print_time = print_time
        if total_cost_list == list():
            self.already_cost = 0
        else:
            # noinspection PyUnresolvedReferences
            self.already_cost = total_cost_list[-1]

            # noinspection PyUnresolvedReferences
            if self.already_cost[0] == '[' and self.already_cost[-1] == ']':
                self.already_cost = self.already_cost[1:-1]
                hh, mm, ss = self.already_cost.split(':')
                hh = int(hh) * 3600
                mm = int(mm) * 60
                ss = int(ss)
                self.already_cost = hh + mm + ss
            else:
                self.already_cost = 0

    def __enter__(self):
        """ do something before executing the context."""
        self.t1 = time()
        if self.print_time:
            print("\n______________________________________________________________________")
            print(">>> Do {}th of {} computations......".format(self.m + 1, self.num_iterations))
            print("    start at [" + strftime("%Y-%m-%d %H:%M:%S", localtime()) + ']')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Do some tear down action; execute after each time the contents are run."""
        self.t2 = time()
        # mth iteration costs?_________________________________________________
        t = self.t2 - self.t1
        if t < 10:
            if self.print_time:
                print("   ~> {}th of {} computations costs: [{:.2f} seconds]".format(
                    self.m + 1, self.num_iterations, t))
        else:
            minutes, seconds = divmod(t, 60)
            hours, minutes = divmod(minutes, 60)
            if self.print_time:
                print("   ~> {}th of {} computations costs: [%02d:%02d:%02d (hh:mm:ss)]".format(
                    self.m + 1, self.num_iterations) % (hours, minutes, seconds))
        self.mth_iteration_cost = t
        minutes, seconds = divmod(self.mth_iteration_cost, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 99:
            hours, minutes, seconds = 99, 59, 59
        self.mth_iteration_cost_HMS = '[%02d:%02d:%02d]' % (hours, minutes, seconds)
        # m iterations cost?___________________________________________________
        minutes, seconds = divmod(t + self.already_cost, 60)
        hours, minutes = divmod(minutes, 60)
        if self.print_time:
            print("   ~> {} of {} computations cost: [%02d:%02d:%02d (hh:mm:ss)]".format(
                self.m + 1, self.num_iterations) % (hours, minutes, seconds))
        if hours > 99:
            hours, minutes, seconds = 99, 59, 59
        self.total_cost = '[%02d:%02d:%02d]' % (hours, minutes, seconds)
        # ERT?_________________________________________________________________
        minutes, seconds = divmod((t + self.already_cost) * (self.num_iterations / (self.m + 1)) -
                                  (t + self.already_cost), 60)
        hours, minutes = divmod(minutes, 60)
        if self.print_time:
            print("   ~> Estimated remaining time: [%02d:%02d:%02d (hh:mm:ss)]\n"
                  % (hours, minutes, seconds))
        if hours > 99:
            hours, minutes, seconds = 99, 59, 59
        self.ERT = '[%02d:%02d:%02d]' % (hours, minutes, seconds)
        return


class Matrix3dInputRunner(Runner):
    """ <unittest> <unittests_P_Solvers> <test_No4_M3IR>."""
    def __init__(self, solver=None, ___file___=None, task_name=None, solver_source_code=None, solver_dir=None):
        """ """
        assert RANK == MASTER_RANK, \
            f"We can only initialize a serial runner in master-core, rAnk={MASTER_RANK}, now we are at core {RANK}."
        super().__init__(solver=solver, ___file___=___file___, task_name=task_name)
        if solver is not None:
            # noinspection PyTypeChecker
            if isinstance(solver, types.FunctionType):
                # noinspection PyUnresolvedReferences
                assert solver.__code__.co_argcount >= 3, \
                    " <Matrix3dInputRunner> : function solver needs to at least have 3 inputs."
            elif isinstance(solver, types.MethodType):
                # noinspection PyUnresolvedReferences
                assert solver.__code__.co_argcount >= 4, \
                    " <Matrix3dInputRunner> : method needs to at least have 3 inputs (besides `self`)."
            elif solver.__class__.__name__ == 'CPUDispatcher':
                pass
            else:
                raise NotImplementedError()
            assert len(self._input_names_) == 3, " <Matrix3dInputRunner> : we need 3 positional inputs."

        self._input_shape_ = None
        self._I0Seq_ = None
        self._I1Seq_ = None
        self._I2Seq_ = None
        self._tabular_ = M_TIR_Tabulate(self)
        if solver_source_code is not None:
            assert isinstance(solver_source_code, str), \
                f"solver_source_code={solver_source_code} ({solver_source_code.__class__.__name__}) wrong, must be str"
            self._solver_source_code_ = solver_source_code
        if solver_dir is not None:
            assert isinstance(solver_dir, str), \
                f"solver_dir={solver_dir} ({solver_dir.__class__.__name__}) wrong, must be str"
            self._solver_dir_ = solver_dir

        print("-<solver_dir>-<{}>-".format(solver_dir), flush=True)
        self._freeze()

    @classmethod
    def ___file_name_extension___(cls):
        return '.m3ir'

    @classmethod
    def ___generate_self_empty_copy___(cls):
        return Matrix3dInputRunner()
    
    @property
    def input_shape(self):
        isp = self._input_shape_

        if isp is None and self.rdf is not None:
            if self._criterion_ == 'standard':
                ins = list(self.rdf.keys()[:3])
                i2 = set(self.rdf[ins[2]].tolist())
                sp2 = len(i2)
                isp = (None, None, sp2)
            else:
                raise NotImplementedError()

        return isp

    
    def ___parse_and_check_iterate_inputs___(self, i0, i1, i2, criterion='standard'):
        """
        We parse and check the shape of `i0, i1, i2` according to `criterion`. At the
        end, the inputs will be put into 3D matrices as the class name says.
        
        Parameters
        ----------
        criterion: str, optional
            'standard' :
                The 'standard' criterion stands for that: `i0` and `i1` are main 
                variables and they do not change along `i2`. So we need `i0` and `i1` 
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
        
        Attributes
        ----------
        self._input_shape_ : 
        self._I0Seq_ :
        self._I1Seq_ :
        self._I2Seq_ :
        self._num_iterations_ :
        
        """
        # _____ standard criterion ___________________________________________________
        if criterion == 'standard':
            shape_i0_i = len(i0)
            assert shape_i0_i == len(i1), " <Matrix3dInputRunner> : i0, i1 len do not match."
            shape_i0_j = 0
            for i, i0i in enumerate(i0):
                if len(i0i) > shape_i0_j:
                    shape_i0_j = len(i0i)
                assert len(i0i) == len(i1[i]), \
                    " <Matrix3dInputRunner> : i0[{}], i1[{}] len do not match.".format(i, i)
            shape0, shape1 = shape_i0_i, shape_i0_j
            shape2 = len(i2)
            self._input_shape_ = (shape0, shape1, shape2)
            self._I0Seq_ = initialize_3d_list(*self._input_shape_)
            self._I1Seq_ = initialize_3d_list(*self._input_shape_)
            self._I2Seq_ = initialize_3d_list(*self._input_shape_)
            self._num_iterations_ = 0
            for i in range(shape0):
                for j in range(shape1):
                    for k in range(shape2):
                        try:
                            self._I0Seq_[i][j][k] = i0[i][j]
                            self._I1Seq_[i][j][k] = i1[i][j]
                            self._I2Seq_[i][j][k] = i2[k]
                            self._num_iterations_ += 1
                        except IndexError:
                            pass
        # ------------------------------------------------------------------------------
        else:
            raise NotImplementedError(" <Matrix3dInputRunner> : criterion={} not coded.".format(criterion))

        self._criterion_ = criterion



    def iterate(self, i0, i1, i2, criterion='standard', writeto=None, saveto=True, show_progress=True, **kwargs):
        """ 
        Parameters
        ----------
        i0 :
            The first input.
        i1 :
            The second input.
        i2 : 
            The third input.
        criterion : optional
            Default: 'standard'. Which type of criterion we are going to parse the inputs: `i0`, `i1`, `i2`. The
            detailed explanation is given in the docstring of method ``___parse_and_check_iterate_inputs___``.
        writeto :
            Write to file named 'writeto' before all iterations and after each 
            iteration.
        saveto :
            if saveto is str:
                We save self to `saveto.m3ir`
            elif saveto is True and writeto is str:
                We save self to `writeto.m3ir`.
            else:
                We do no save.
        show_progress : bool

        kwargs :
            To be passed to the solver.

        """
        self.___parse_and_check_iterate_inputs___(i0, i1, i2, criterion=criterion)
        self.___kwargs___ = kwargs
        self.___prepare_write_file___(writeto)
        print("------------------------ > M3IR computations < -----------------------------", flush=True)
        I, J, K = self._input_shape_

        if SIZE > 1:
            assert RANK == MASTER_RANK
            for sc in range(SIZE):  # do not use bcast for safety
                if sc != MASTER_RANK:
                    COMM.send([I, J, K], dest=sc, tag=sc + 1)   # position mark 1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            COMM.barrier()

        if not show_progress:
            pbar = tqdm(total=I * J * K,  # this amount it not accurate, but it is OK. just leave it like this.
                        desc=self.__class__.__name__)

        n = 0
        for k in range(K):  # we let the axis2 go at the last.
            for i in range(I):  # we let the axis0 go secondly.
                for j in range(J):  # we let the axis1 go firstly.

                    if not show_progress:
                        # noinspection PyUnboundLocalVariable
                        pbar.update(1)
                    else:
                        pass

                    # _______ there is not input for computing _________________________
                    if self._I0Seq_[i][j][k] is None:

                        if SIZE > 1:
                            for sc in range(SIZE):  # do not use bcast for safety
                                if sc != MASTER_RANK:
                                    COMM.send(False, dest=sc, tag=sc + 2)  # position mark 2 <<<<<<<<<<<<<<<<
                            COMM.barrier()

                    # _______ It is already computed in the `writeto` file _____________
                    elif self._computed_pool_ != () and \
                        self.___check_inputs_in_computed_pool___(
                            [self._I0Seq_[i][j][k], self._I1Seq_[i][j][k], self._I2Seq_[i][j][k]]):
                        if show_progress:
                            print(f'\n -----> REPEATED {n+1}/{self.num_iterations} computation of '
                                  f'inputs:\n\t{[self._I0Seq_[i][j][k], self._I1Seq_[i][j][k], self._I2Seq_[i][j][k]]}')

                            print('\t__________________________________________________________________\n', flush=True)
                        else:
                            pass
                        n += 1
                        sleep(0.0125)
                        if SIZE > 1:
                            for sc in range(SIZE):  # do not use bcast for safety
                                if sc != MASTER_RANK:
                                    COMM.send(False, dest=sc, tag=sc + 2)  # position mark 2 <<<<<<<<<<<<<
                            COMM.barrier()

                    # _______ We have to compute this set of inputs ____________________
                    else:
                        if self._rdf_.empty:
                            m = 0
                            TTC = [' ', ]
                        else:
                            TTC = [0, self._rdf_['TTC'][self._rdf_.index[-1]]]
                            m = self._rdf_.index[-1] + 1

                        if SIZE > 1:
                            for sc in range(SIZE):  # do not use bcast for safety
                                if sc != MASTER_RANK:
                                    COMM.send(True, dest=sc, tag=sc + 2)  # position mark 2 <<<<<<<<<<<<<<<<<<<<<
                            COMM.barrier()

                        with TimeIteration(n, self.num_iterations, TTC, show_progress) as TIcontextmanager:
                            if show_progress:
                                print('\t> input[0]: {} = {}'  .format(self.input_names[0],
                                                                       self._I0Seq_[i][j][k]))
                                print('\t> input[1]: {} = {}'  .format(self.input_names[1],
                                                                       self._I1Seq_[i][j][k]))
                                print('\t> input[2]: {} = {}\n'.format(self.input_names[2],
                                                                       self._I2Seq_[i][j][k]), flush=True)

                            if SIZE == 1:
                                outputs = self._solver_(self._I0Seq_[i][j][k],
                                                        self._I1Seq_[i][j][k],
                                                        self._I2Seq_[i][j][k], **kwargs)

                            else:  # parallel case

                                INPUTS = [self._I0Seq_[i][j][k], self._I1Seq_[i][j][k], self._I2Seq_[i][j][k], kwargs]

                                if SIZE > 1:
                                    for sc in range(SIZE):  # do not use bcast for safety
                                        if sc != MASTER_RANK:
                                            COMM.send(INPUTS, dest=sc, tag=sc + 3)  # position mark 3 <<<<<<<<<<<<
                                    COMM.barrier()

                                outputs = self._solver_(INPUTS[0], INPUTS[1], INPUTS[2], **INPUTS[3])

                        self.___update_rdf___(m, (self._I0Seq_[i][j][k], 
                                                  self._I1Seq_[i][j][k], 
                                                  self._I2Seq_[i][j][k]),
                                              outputs,
                                              TIcontextmanager.mth_iteration_cost_HMS,
                                              TIcontextmanager.total_cost,
                                              TIcontextmanager.ERT,
                                              show_progress=show_progress)
                        self.___write_to___(m)
                        n += 1
                    # ------------------------------------------------------------------
        if not show_progress:
            pbar.close()
        self._results_ = self._rdf_
        print('>->->->->->->->->->->-> Computations Done >>>>> Post-processing ...', flush=True)
        self.___deal_with_saveto___(writeto, saveto)
        # self.___send_an_completion_reminder_email_to_me___(writeto, saveto)
        print("______________________ > M3IR Iterations Done < ____________________________", flush=True)
        
    @property
    def tabular(self):
        """ """
        return self._tabular_
        

def TestSolver(a, b, c):
    """ 
    Parameters
    ----------
    a :
    b :
    c :
        
    Returns
    -------
    d :
    e :


    """
    d = a + 5*b - c
    e = a + b
    return d, e


def initialize_3d_list(a, b, c):
    """

    :param a:
    :param b:
    :param c:
    :return:
    """
    lst = [[[None for _ in range(c)] for _ in range(b)] for _ in range(a)]
    return lst


if __name__ == '__main__':

    R = Matrix3dInputRunner(TestSolver)
    
    a = [[1, 2, 3],
         [1, 2]]
    b = [[1, 1, 1],
         [2, 2]]
    c = [0, 0.2]
    
    R.iterate(a, b, c, writeto='123.txt')

    R.visualize.loglog('b', 'd')

    import os

    # os.remove('123.txt')
    os.remove('123.m3ir')
