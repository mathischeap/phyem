# -*- coding: utf-8 -*-
"""
Here is a three inputs runner. We make it for 3 inputs becuase mostly we run a 
function at different basis function order p, elements layout k and crazy 
coefficient c. 

<unittest> <unittests_P_Solvers> <test_No3_TIR>.

Yi Zhang (C)
Created on Mon Oct 29 15:38:46 2018
Aerodynamics, AE
TU Delft
"""
import types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime, time
from tools.decorators.all import accepts
from tools.legacy.serialRunners._runner_ import Runner
from tools.legacy.serialRunners.INSTANCES.COMPONENTS.tir_drw import TIR_DRW
from tools.legacy.serialRunners.INSTANCES.COMPONENTS.m_tir_tabular import M_TIR_Tabulate
from src.config import SIZE
assert SIZE == 1, "Runners can only be run in single thread."

class TimeIteration:
    """ We use this contextmanager to time an iteration. """

    def __init__(self, m, num_iterations, total_cost_list):
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
        if total_cost_list == list():
            self.already_cost = 0
        else:
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
        print("\n\n______________________________________________________________________")
        print(">>> Do {}th of {} iterations......".format(self.m + 1, self.num_iterations))
        print("    start at [" + strftime("%Y-%m-%d %H:%M:%S", localtime()) + ']')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Do some tear down action; execute after each time the contents are run."""
        self.t2 = time()
        # mth iteration costs?_________________________________________________
        t = self.t2 - self.t1
        if t < 10:
            print("\n   ~> {}th of {} iterations costs: [{:.2f} seconds]".format(
                self.m + 1, self.num_iterations, t))
        else:
            minutes, seconds = divmod(t, 60)
            hours, minutes = divmod(minutes, 60)
            print("\n   ~> {}th of {} iterations costs: [%02d:%02d:%02d (hh:mm:ss)]".format(
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
        print("   ~> {} of {} iterations cost: [%02d:%02d:%02d (hh:mm:ss)]".format(
            self.m + 1, self.num_iterations) % (hours, minutes, seconds))
        if hours > 99:
            hours, minutes, seconds = 99, 59, 59
        self.total_cost = '[%02d:%02d:%02d]' % (hours, minutes, seconds)
        # ERT?_________________________________________________________________
        minutes, seconds = divmod((t + self.already_cost) * (self.num_iterations / (self.m + 1)) -
                                  (t + self.already_cost), 60)
        hours, minutes = divmod(minutes, 60)
        print("   ~> Estimated remaining time: [%02d:%02d:%02d (hh:mm:ss)]\n"
              % (hours, minutes, seconds))
        if hours > 99:
            hours, minutes, seconds = 99, 59, 59
        self.ERT = '[%02d:%02d:%02d]' % (hours, minutes, seconds)
        return

class ThreeInputsRunner(Runner):
    """ 
    We use this Class to run functions of three inputs. Normally, the three
    inputs are basis function degree, mesh density and a domain coefficient
    (like the deformation coefficient for Crazy mesh.). Therefore, we normally
    have input[0] or input[1] has many values but input[2] has very few values.
    
    This is not very good since we will meshgrid the inputs in input[0] and input[1],
    which may cause very computational costing run at case (input[0][-1],input[1][-1]).
    But anyway, this runner is still very useful at a lot of situations. And we have
    made a remedy: we have `skip` option.
    
    Therefore, for the matplot method, we can only plot against input[0] or
    input[1]. And let input[2] be a value and to help determine how many lines 
    we are going to plot.
    
    <test_unittests> <Test_solvers> <test_No3_TIR>.
    
    """
    def __init__(self, solver=None, ___file___=None, task_name=None):
        """
        Parameters
        ----------
        solver ：
            The solver. If solver is set to None, then we probably only use it
            to read data from somewhere, like '.txt' file, and then plot the 
            data.
        """
        super().__init__(solver=solver, ___file___=___file___, task_name=task_name)
        if solver is not None:
            # noinspection PyTypeChecker
            if isinstance(solver, types.FunctionType):
                # noinspection PyUnresolvedReferences
                assert solver.__code__.co_argcount >= 3, \
                    " <ThreeInputsRunner> : function solver needs to at least have 3 inputs."
            elif isinstance(solver, types.MethodType):
                # noinspection PyUnresolvedReferences
                assert solver.__code__.co_argcount >= 4, \
                    " <ThreeInputsRunner> : method needs to at least have 3 inputs (besides `self`)."
            elif solver.__class__.__name__ == 'CPUDispatcher':
                pass
            else:
                raise NotImplementedError()
            assert len(self._input_names_) == 3, " <ThreeInputsRunner> : we need 3 input names."
        self._I0seq_, self._I1seq_, self._I2seq_ = None, None, None
        self._S0seq_, self._S1seq_, self._S2seq_ = None, None, None
        self._TIR_DRW_ = TIR_DRW(self)
        self._tabular_ = M_TIR_Tabulate(self)
        self._freeze()

    @classmethod
    def ___file_name_extension___(cls):
        return '.3ir'

    @property
    def drw(self):
        """Data reader and writer."""
        return self._TIR_DRW_
    
    @property
    def input_shape(self):
        return len(self._I0seq_), len(self._I1seq_), len(self._I2seq_)
    
    @accepts('self', (list, tuple), (list, tuple), (list, tuple))
    def iterate(self, I0seq, I1seq, I2seq, writeto=None, saveto=None, 
                skips=None, **kwargs):
        """
        Parameters
        ----------
        I0seq: 
            The sequence for the first input of `solver`.
        I1seq: 
            The sequence for the second input of `solver`.
        I2seq: 
            The sequence for the third input of `solver`.
        writeto : None or str, optional
            If `writeto` is not None, we write the result after each iteration
            to the file named `writeto`. Notice that when there is already a
            file named `writeto`.txt, this means we already computed some 
            results and they are put in this .txt file. So we first read data
            from this .txt file (of course, we first check if the file is 
            correct), and then we only compute the remaining iterations and 
            append all these newly computed results to `writeto`.txt.
        saveto : Nonetype or str, optional
            If `saveto` is not None, after all iterations, we save self to 
            `saveto`.
        skips : list, tuple, optional
            The compuating sequence to be skipped. The outputs will be put as
            nan in the results. 
            
            Note that this `skips` option also result in a `meshgrid` structure, so a
            block of inputs matrix will be skipped.
            
        """
        if skips is None:
            skips = [None, None, None]
        self._I0seq_, self._I1seq_, self._I2seq_ = I0seq, I1seq, I2seq
        # noinspection PyTypeChecker
        self._S0seq_ = [skips[0],] if np.shape(skips[0])==() else skips[0]
        # noinspection PyTypeChecker
        self._S1seq_ = [skips[1],] if np.shape(skips[1])==() else skips[1]
        # noinspection PyTypeChecker
        self._S2seq_ = [skips[2],] if np.shape(skips[2])==() else skips[2]
        I, J, K = len(I0seq), len(I1seq), len(I2seq)
        print("-<ThreeInputsRunner>-<I0>: {}".format(I0seq))
        print("-<ThreeInputsRunner>-<I1>: {}".format(I1seq))
        print("-<ThreeInputsRunner>-<I2>: {}".format(I2seq))
        print("-<ThreeInputsRunner>-<kwargs>: {}".format(kwargs))
        num_iterations = I * J * K
        print("-<ThreeInputsRunner>-<total iterations>: {}.".format(num_iterations))
        self.___kwargs___ = kwargs
        self.___init_results___()
        if writeto is not None: self.drw.read(writeto)
        print("\n\n------------------------- > TIR Iterations < -------------------------")
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    m = i + j*I + k*I*J
                    if m not in self.drw._computed_m_:
                        if self._I0seq_[i] in self._S0seq_ and \
                            self._I1seq_[j] in self._S1seq_ and \
                            self._I2seq_[k] in self._S2seq_:
                            outputs = [np.NaN for _ in range(len(self._output_names_))]
                            if m == 0:
                                ITC = TTC = ERT = '[00:00:00]'
                            else:
                                ITC = '[00:00:00]'
                                TTC = self._results_['total_cost'][-1]
                                ERT = self._results_['ERT'][-1]
                            self.___update_results___(self._I0seq_[i], self._I1seq_[j], self._I2seq_[k],
                                                      outputs, ITC, TTC, ERT)
                        else:
                            with TimeIteration(m, num_iterations, self._results_['total_cost']) as TIcontextmanager:
                                print('\t> input[0]: {}'.format(self._I0seq_[i]))
                                print('\t> input[1]: {}'.format(self._I1seq_[j]))
                                print('\t> input[2]: {}'.format(self._I2seq_[k]))
                                print('-------------------------------------------------\n')
                                outputs = self._solver_(self._I0seq_[i], 
                                                        self._I1seq_[j], 
                                                        self._I2seq_[k], **kwargs)
                                assert len(outputs) == len(self._output_names_), " <ThreeInputsRunner> "
                            self.___update_results___(self._I0seq_[i], self._I1seq_[j], self._I2seq_[k]
                                                     ,outputs
                                                     ,TIcontextmanager.mth_iteration_cost_HMS
                                                     ,TIcontextmanager.total_cost
                                                     ,TIcontextmanager.ERT)
                        self.drw.write_iteration(m)
        self.___deal_with_saveto___(writeto, saveto)
        self.___send_an_completion_reminder_email_to_me___(writeto, saveto)
        print("_______________________ > TIR IterationsDone < _______________________\n\n")
    
    def ___init_results___(self):
        """ """
        self._results_ = dict()
        self._results_['I0'] = []
        self._results_['I1'] = []
        self._results_['I2'] = []
        for inn in self._input_names_: self._results_[inn] = []
        for on in self._output_names_: self._results_[on] = []
        self._results_['solver_time_cost'] = []
        self._results_['total_cost'] = []
        self._results_['ERT'] = []
    
    def ___update_results___(self, I0, I1, I2, outputs, time_cost, tc, ERT):
        """ """
        self._results_['I0'].append(I0)
        self._results_['I1'].append(I1)
        self._results_['I2'].append(I2)
        self._results_[self._input_names_[0]].append(I0)
        self._results_[self._input_names_[1]].append(I1)
        self._results_[self._input_names_[2]].append(I2)
        for m, on in enumerate(self.output_names):
            self._results_[on].append(outputs[m])
        self._results_['solver_time_cost'].append(time_cost)
        self._results_['total_cost'].append(tc)
        self._results_['ERT'].append(ERT)
        # noinspection PyStatementEffect
        self.results
    
    @property
    def _inputs_index_dict_(self):
        """ 
        Returns
        -------
        _icd_ : dict
            A dict which key: value means 'ith iteration': '(input[0], input[1]
            , input[2])'.
            
        """
        _icd_ = {}
        try:
            for i in range(len(self.results['I0'])):
                _icd_[i] = (self.results['I0'][i], self.results['I1'][i], self.results['I2'][i])
        except KeyError:
            pass
        return _icd_

    @property
    def tabular(self):
        """ """
        return self._tabular_
        
    @classmethod
    def readfile(cls, readfilename):
        """ 
        A overwirte of `readfile` in its parent 'Runner'. We overwrite it because this
        class is not standard, it has special data structure for writting.
        
        """
        assert '.txt' in readfilename, " <ThreeInputsRunner> : I only read .txt files."
        return cls.readtxt(readfilename)
        
    @classmethod
    def readtxt(cls, filename):
        """ 
        We use this method to read a '.txt' file and make self capble to
        plot the results in the '.txt' file.
        
        """
        with open(filename, 'r') as f:
            fstr = f.readlines()
        total_lines = len(fstr)
        assert fstr[0][:-1] == '<ThreeInputsRunner>', \
            " <TIR_DRW> : I need a <ThreeInputsRunner> file."
        TIRinstance = ThreeInputsRunner()
        i = fstr.index('<inputs>:\n')
        input_0, I0sequence = fstr[i+1].split(' sequence: ')
        input_1, I1sequence = fstr[i+2].split(' sequence: ')
        input_2, I2sequence = fstr[i+3].split(' sequence: ')
        TIRinstance._input_names_ = (input_0, input_1, input_2)
        TIRinstance._I0seq_ = list(eval(I0sequence))
        TIRinstance._I1seq_ = list(eval(I1sequence))
        TIRinstance._I2seq_ = list(eval(I2sequence))
        i = fstr.index('<kwargs>:\n')
        TIRinstance.___kwargs___ = fstr[i+1][:-1]
        i = fstr.index('<results>:\n')
        i += 1
        stored = fstr[i].split()
        num_stored = len(stored)
        j = stored.index('|')
        outputs = stored[4:j]
        TIRinstance._output_names_ = tuple(outputs)
        TIRinstance.___init_results___()
        while i < total_lines:
            try:
                int(fstr[i][0])
                fstr_i_split = list(fstr[i].split())
                if len(fstr_i_split) == num_stored:
                    # when this is happening, we stored full values at this line, so
                    # we can keep it to `self._TIR_.results`.
                    for k in range(j):
                        fstr_i_split[k] = float(fstr_i_split[k])
                    TIRinstance.___update_results___(*fstr_i_split[1:4], 
                                                     fstr_i_split[4:j], 
                                                     *fstr_i_split[j+1:])
                else:
                    break
            except ValueError:
                pass
            i += 1
        return TIRinstance
    
    def writetxt(self, filename):
        """We write self to the file named `filename.txt`."""
        self.drw.readwritefilename = filename
        self.drw.initialize_writing()
        for m in range(len(self.results['I0'])):
            self.drw.write_iteration(m)

    @property
    def I0seq(self):
        return self._I0seq_  
    
    @property
    def I1seq(self):
        return self._I1seq_  
    
    @property
    def I2seq(self):
        return self._I2seq_
    
    @property
    def rdf(self):
        """ The `results` in `DataFrame` format."""
        if self.results is None:
            return None
        else:
            self._rdf_ = pd.DataFrame(self.results)
            self._rdf_ = self._rdf_.drop(columns=['I0', 'I1', 'I2'])
            self._rdf_.columns = (*self.input_names, *self.output_names, 'ITC', 'TTC', 'ERT')
            return self._rdf_

    @classmethod
    def merge(cls, *args):
        """ 
        We try to merge serveral 'ThreeInputsRunner' into a single one.
        
        We only merge ThreeInputsRunner that have the same results keys. In
        principle, we shoulad also need the same sover and so on. But we skip
        that here to leave the users more freedom.
        
        """
        for i, arg in enumerate(args):
            assert arg.__class__.__name__ == 'ThreeInputsRunner', \
                " <ThreeInputsRunner> : {}th arg is not ThreeInputsRunner.".format(i)
        _input_names_ = args[0]._input_names_
        _output_names_ = args[0]._output_names_
        ___kwargs___ = args[0].___kwargs___
        for arg in args:
            assert arg._input_names_ == _input_names_, " <ThreeInputsRunner> "
            assert arg._output_names_ == _output_names_, " <ThreeInputsRunner> "
            assert arg.___kwargs___ == ___kwargs___, " <ThreeInputsRunner> "
        TIRinstance = ThreeInputsRunner()
        TIRinstance._input_names_ = _input_names_
        TIRinstance._output_names_ = _output_names_
        TIRinstance.___kwargs___ = ___kwargs___
        I0seq, I1seq, I2seq = set(), set(), set()
        for i, tir in enumerate(args):
            I0seq.update(set(tir.I0seq))
            I1seq.update(set(tir.I1seq))
            I2seq.update(set(tir.I2seq))
        TIRinstance._I0seq_ = list(I0seq)
        TIRinstance._I1seq_ = list(I1seq)
        TIRinstance._I2seq_ = list(I2seq)
        _inputs_dict_ = {}
        for i, tiri in enumerate(args):
            for inputs in tiri._inputs_index_dict_:
                _inputs_dict_[str(i) + '-' + str(inputs)] = tiri._inputs_index_dict_[inputs]
        result_keys = args[0].results.keys()
        TIRinstance._results_ = {}
        for key in result_keys:
            TIRinstance._results_[key] = []
        I, J, K = len(I0seq), len(I1seq), len(I2seq)
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    _inputs_ = (TIRinstance._I0seq_[i], TIRinstance._I1seq_[j], TIRinstance._I2seq_[k])
                    for thekey, value in _inputs_dict_.items():
                        if _inputs_ == value:
                            break
                    # noinspection PyUnboundLocalVariable
                    assert _inputs_dict_[thekey] == _inputs_, \
                        " <ThreeInputsRunner> : no data found for inputs: {}.".format(_inputs_)
                    ith_arg, jth_result = thekey.split('-')
                    ith_arg = int(ith_arg)
                    jth_result = int(jth_result)
                    for key in result_keys:
                        TIRinstance._results_[key].append(args[ith_arg].results[key][jth_result])
        return TIRinstance





    def matplot(self, res2plot, against, plot_type='loglog',
        hcp=None, show_order=True,  # h-convergence plot related
        title=None, left=0.15, bottom=0.15,
        ylabel=None, yticks=None,
        xlabel=None, xticks=None,
        labels=None, linewidth=None, style=None, color=None,
        line_styles=None, line_colors=None, legend_local='best',
        tick_size=15, label_size=15, legend_size=15, title_size=15,
        minor_tick_length=6, major_tick_length=12, tick_pad=8, legend_frame=False,
        figsize=(8, 6),
        usetex=False, saveto=None):
        """
        We use this method to do plot.

        Paramterts
        ----------
        res2plot : str
        hcp : NoneType or float, optional
            When is_h is NOT None, then we know we are doing h-convergence
            plotting, then the x-date will become h/x-data.
        show_order : bool
            If True, show the order.

        """
        assert self.results is not None, \
            " <ThreeInputsRunner> : SR_iterative is empty, no p-convergence plot."
        if isinstance(res2plot, str):
            res2plot = (res2plot,)
        for r2p in res2plot:
            assert r2p in self.results, " <ThreeInputsRunner> : res2plot={} is wrong.".format(r2p)
        num_r2p = len(res2plot)
        assert against in (0, 1), " <ThreeInputsRunner> : against={} is wrong.".format(against)
        assert len(self.results['I0']) == len(self.I0seq) * len(self.I1seq) * len(self.I2seq), \
            " <ThreeInputsRunner> : results are not complete."
        plt.rc('text', usetex=usetex)
        if usetex:
            plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
        if line_styles is None:
            line_styles = ('-o', '-v', '-s', '-<', '-H', '-8', '-^', '-p',
                           '-*', '-h', '->', '-D', '-d', '-P', '-X')
        if line_colors is None:
            line_colors = [(0.5, 0, 0, 0.7),
                           (0, 0, 0.5, 0.7),
                           (0, 0.5, 0, 0.7),
                           (1, 0, 0, 0.7),
                           (0.5, 0.5, 0, 0.7),
                           (0.5, 0, 0.5, 0.7),
                           (0, 0.5, 0.5, 0.7)]
        I, J, K = len(self._I0seq_), len(self._I1seq_), len(self._I2seq_)
        if against == 0:
            num_lines = num_r2p * J * K
            points_per_line = I
        elif against == 1:
            num_lines = num_r2p * I * K
            points_per_line = J
        else:
            raise Exception
        labels_real_time = []
        line_style_real_time = []
        line_color_real_time = []
        y_data = np.zeros((num_lines, points_per_line))
        for s in range(num_r2p):
            if against == 0:
                for k in range(K):
                    for j in range(J):
                        m = j + k * J
                        n = j + k * J + s * J * K
                        x_data = np.array(self.I0seq)
                        y_data[n, :] = self.results[res2plot[s]][m * I:(m + 1) * I]
                        line_style_real_time += [line_styles[j]]
                        line_color_real_time += [line_colors[k]]
                        if labels is None:
                            if len(res2plot) == 1:
                                labels_real_time.append(r'${}={}$, ${}={:.2f}$'.format(
                                    self.input_names[1], self._I1seq_[j],
                                    self.input_names[2], self._I2seq_[k]))
                            else:
                                labels_real_time.append(r'{}: ${}={}$, ${}={:.2f}$'.format(
                                    res2plot[s].replace('_', '-'),
                                    self.input_names[1], self._I1seq_[j],
                                    self.input_names[2], self._I2seq_[k]))
            elif against == 1:
                for k in range(K):
                    for i in range(I):
                        m = i + k * I + s * I * K
                        x_data = np.array(self.I1seq)
                        y_data[m, :] = self.results[res2plot[s]][k * I * J + i:(k + 1) * I * J:I]
                        line_style_real_time += [line_styles[i]]
                        line_color_real_time += [line_colors[k]]
                        if labels is None:
                            if len(res2plot) == 1:
                                labels_real_time.append(r'${}={}$, ${}={:.2f}$'.format(
                                    self.input_names[0], self._I0seq_[i],
                                    self.input_names[2], self._I2seq_[k]))
                            else:
                                labels_real_time.append(r'{}: ${}={}$, ${}={:.2f}$'.format(
                                    res2plot[s].replace('_', '-'),
                                    self.input_names[0], self._I0seq_[i],
                                    self.input_names[2], self._I2seq_[k]))
            else:
                raise Exception
        if hcp is not None:
            # noinspection PyUnboundLocalVariable
            x_data = hcp / x_data
        plt.figure(figsize=figsize)
        plt.gcf().subplots_adjust(left=left)
        plt.gcf().subplots_adjust(bottom=bottom)
        linewidth = 1 if linewidth is None else linewidth
        style = line_style_real_time if style is None else style
        color = line_color_real_time if color is None else color
        labels = labels_real_time if labels is None else labels

        if show_order:
            for m in range(num_lines):
                # noinspection PyUnboundLocalVariable
                order = (np.log10(y_data[m, -1]) - np.log10(y_data[m, -2])) / \
                        (np.log10(x_data[-1]) - np.log10(x_data[-2]))
                labels[m] = labels[m] + ', $\mathrm{order}=%.2f$' % order
        ploter = getattr(plt, plot_type)
        for i in range(num_lines):
            #            ploter(x_data , y_data[i,:])
            ploter(x_data, y_data[i, :], style[i],
                   color=color[i], label=labels[i], linewidth=linewidth)

        # plot title?__________________________________________________________
        if title is None:
            plt.title(r'' + str(res2plot).replace('_', '-'), fontsize=title_size)
        elif title is False:
            pass
        else:
            plt.title(r'' + title, fontsize=title_size)
        if xlabel is not None: plt.xlabel(xlabel, fontsize=label_size)
        if ylabel is not None: plt.ylabel(ylabel, fontsize=label_size)
        if xticks is not None: plt.xticks(xticks)
        if yticks is not None: plt.yticks(yticks)


        plt.tick_params(which='both', labeltop=False, labelright=False, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=minor_tick_length)
        plt.tick_params(axis='both', which='major', direction='in', length=major_tick_length)
        plt.tick_params(axis='both', which='both', labelsize=tick_size)
        plt.tick_params(axis='x', which='both', pad=tick_pad)
        plt.legend(fontsize=legend_size, loc=legend_local, frameon=legend_frame)


        plt.tight_layout()
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight')
        plt.show()


def solver(K, O, T):
    """
    Parameters
    ----------
    K :
    O :
    T :

    Returns
    -------
    d :
    e :
    """
    d = K + 5 * O - T
    e = K + T
    return d, e


if __name__ == '__main__':
    from numpy import pi
#     from CONFORMING.DIII.__PROGRAMS__.P2_Poisson.S1_IGAA2018.v2101_DB_solver import solver
    K = [1,2,3]
    O = [1,2]
    T = [0, pi/4]
    runner1 = ThreeInputsRunner(solver, __file__)
#    runner1.iterate(K, O, T, writeto='test_write1.txt', skips=((1,2),(1,2),(0,)))
    runner1.iterate(K, O, T, writeto='test_write1.txt')

    K = [1,2,3]
    O = [3,]
    T = [0, pi/4]
    runner2 = ThreeInputsRunner(solver, __file__)
    runner2.iterate(K, O, T)

    runner1.writetxt('test_write1')
    runner2.writetxt('test_write2')

    runner1 = ThreeInputsRunner.readtxt('test_write1')
    runner2 = ThreeInputsRunner.readtxt('test_write2')

    rm = ThreeInputsRunner.merge(runner1, runner2)
    rm.matplot('d', 1, 'loglog', usetex=True)
    # rm.matplot('e', 1, 'loglog', usetex=True)

    runner1.saveto('save_1')
    runner2.saveto('save_2')
    runner1 = ThreeInputsRunner.readfrom('save_1.3ir')
    runner2 = ThreeInputsRunner.readfrom('save_2.3ir')

    import os

    os.remove('save_1.3ir')
    os.remove('save_2.3ir')
    os.remove('test_write1.txt')
    os.remove('test_write1')
    os.remove('test_write2')
