# -*- coding: utf-8 -*-
import random
import string
from src.config import RANK, MASTER_RANK
from tools.frozen import Frozen
import psutil
import platform
import socket
import datetime
from tools.miscellaneous.timer import MyTimer
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# noinspection PyBroadException
class IteratorMonitor(Frozen):
    """The monitor class for the Iterator.
    """
    def __init__(self, iterator, monitoring_factor, name):
        assert RANK == MASTER_RANK, "Should only initialize and use monitor in the master core."

        self._iterator_ = iterator

        self._parse_monitoring_factor(monitoring_factor)

        if name is None:
            lettersAndDigits = string.ascii_letters + string.digits
            name = ''.join(random.choice(lettersAndDigits) for _ in range(6))
            name = 'Itr-' + name + '-' + str(id(self))[-5:]

        else:
            assert isinstance(name, str), f"name must be a string."
        self._name = name

        self._times_ = [np.nan]  # cost of all iterations
        self._TIMES_ = [0.]  # total cost after each iteration.

        self._num_iterations = None  # how many iterations in total.
        self._start_time = 0  # the start time of iteration `run`.
        self._str_started_time_ = MyTimer.current_time()
        self._iteration_start_time = 0
        self._report_times = 0

        # ------ machine info ---------------------------------------------
        memory = psutil.virtual_memory()
        self._mem_total_ = str(round(memory.total / 1024 / 1024 / 1024, 2))
        self._cpu_count_ = psutil.cpu_count(logical=False)
        self._logical_cpu_count_ = psutil.cpu_count()
        self._platform_ = platform.platform()
        self._system_ = platform.system()
        self._processor_ = platform.processor()
        self._architecture_ = platform.architecture()
        self._python_version_ = platform.python_version()

        self._freeze()

    def _parse_monitoring_factor(self, monitoring_factor):
        """"""
        assert 0 <= monitoring_factor <= 1, \
            f"monitoring_factor={monitoring_factor} is wrong, must be in [0,1]."
        self._ast = 600 * (1 - monitoring_factor)    # auto_save_time
        self._rpt = 750 * (1 - monitoring_factor)   # report time

        self._last_ast = time() - self._ast  # this is correct, do not delete - self._ast
        self._last_rpt = time() - self._rpt  # this is correct, do not delete - self._rpt

    @property
    def name(self):
        """name"""
        return self._name

    @property
    def _average_each_run_cost_(self):
        used_time = (time() - self._start_time)
        each_time = used_time / (len(self._times_) - 1)
        return each_time

    @property
    def _estimated_remaining_time_(self):
        used_time = (time() - self._start_time)
        each_time = used_time / (len(self._times_) - 1)
        all_time = each_time * self._num_iterations
        return all_time - used_time

    @property
    def _estimated_end_time_(self):
        eet = datetime.datetime.now() + datetime.timedelta(
            seconds=self._estimated_remaining_time_)
        return eet

    @property
    def _computed_steps_(self):
        return len(self._times_) - 1

    def _measure_start(self):
        """"""
        self._iteration_start_time = time()

    def _measure_end(self):
        """"""
        current_time = time()
        self._times_.append(
            current_time - self._iteration_start_time
        )
        self._TIMES_.append(
            current_time - self._start_time
        )

        # auto save
        if current_time - self._last_ast >= self._ast:
            self.save()
            self._last_ast = current_time
        else:
            pass

        # report
        if current_time - self._last_rpt >= self._rpt:
            self.report()
            self._last_rpt = current_time
        else:
            pass

    def save(self, over=False):
        """save to .csv file"""
        if over:
            saving = True
            time_start = time()
            while saving:   # repeat saving till time out or success.
                try:
                    self._iterator_.RDF.to_csv(self.name + '.csv', header=True)
                    saving = False
                except:
                    sleep(5)
                    if time() - time_start > 60:
                        saving = False
                    else:
                        saving = True
        else:
            try:  # in case like PermissionError.
                self._iterator_.RDF.to_csv(self.name + '.csv', header=True)
            except:
                sleep(2)
                try:
                    self._iterator_.RDF.to_csv(self.name + '.csv', header=True)
                except:  # just skip the saving in case like PermissionError.
                    pass

    def report(self, over=False):
        """make graphic report"""
        self._graphic_report(over)
        self._report_times += 1

    def _graphic_report(self, over):
        """"""
        save_time = MyTimer.current_time()[1:-1]
        indices = self._select_reasonable_amount_of_data(300, last_num=75)

        RDF = self._iterator_.RDF.iloc[indices]

        matplotlib.use('Agg')  # make sure we use the right backend.

        plt.rc('text', usetex=False)

        num_subplots = RDF.shape[1] + 6
        # We plot 4 extra: 't iteration', 't accumulation', solver message, and machine load

        colors = matplotlib.colormaps['Dark2']
        r_num_subplots = int(np.ceil(num_subplots/2))
        x_len, y_len = 18, 4.5*r_num_subplots
        fig = plt.figure(figsize=(x_len, y_len))
        plot_keys = [
                        'info0', 'info1', 't iteration', 't accumulation', 'solver message', 'machine load'
                    ] + list(RDF.columns)

        # subplots ...
        for i, di in enumerate(plot_keys):
            ylabel_backgroundcolor = 'paleturquoise'
            sci_format_y = True
            face_color = 'aliceblue'
            ylabel = di.replace('_', '-')
            m = int(i/2)
            n = i % 2
            # noinspection PyUnboundLocalVariable
            ax = plt.subplot2grid((r_num_subplots, 2), (m, n))
            if di == 'machine load':
                mem_percent = psutil.virtual_memory().percent
                cpu_load = self._iterator_._cpu_load
                plt.axis((0, 10, 0, 10))
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                plt.text(0.1, 9.5, 'MACHINE LOAD:',
                         color='deepskyblue', fontsize=18, style='normal',
                         ha='left', va='top', wrap=True)

                TEXT = f"MEM: {mem_percent}% of {self._mem_total_}G used.\n" \
                       f" CPU: {cpu_load}% of {self._cpu_count_} (physical) = " \
                       f"{self._logical_cpu_count_} (logical) processors used.\n\n" \
                       f"SYSTEM: {self._system_}\n" \
                       f"{self._platform_}\n" \
                       f"{self._architecture_}\n" \
                       f"{self._processor_}\n\n" \
                       f"PYTHON version: {self._python_version_}"

                plt.text(0.1, 8, TEXT, color='navy', fontsize=13,
                         ha='left', va='top', wrap=True)

            elif di == 'solver message':
                plt.axis((0, 10, 0, 10))
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                plt.text(0.1, 9.5, 'SOLVER MESSAGE:', color='deepskyblue', fontsize=18, style='normal',
                         ha='left', va='top', wrap=True)
                message = ''
                for M in self._iterator_.message:
                    if len(M) < 70:
                        message += M
                    else:
                        MS = M.split(' ')
                        new_line = 0
                        for ms in MS:
                            if ms == '':
                                # remove extra space.
                                pass
                            else:
                                new_line += len(ms) + 1
                                message += ms + ' '
                                if new_line >= 60:
                                    message += '\n'
                                    new_line = 0
                    message += '\n\n'

                plt.text(0.1, 8.5, message, color='black', fontsize=12,
                         ha='left', va='top', wrap=True)

            elif di == 'info0':
                face_color = 'honeydew'
                plt.axis((0, 10, 0, 10))
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                t1 = f'* {self.name}'
                if len(t1) > 75:
                    t1 = '...' + t1[-72:]
                else:
                    pass
                plt.text(-2, 11.3, t1, color='darkorchid', fontsize=26, style='normal', ha='left',
                         va='top', wrap=True)
                sITC = MyTimer.seconds2hms(self._average_each_run_cost_)
                sTTC = MyTimer.seconds2dhmsm(self._TIMES_[-1]).split('.')[0] + ']'
                sLIC = MyTimer.seconds2hms(self._times_[-1])
                sERT = MyTimer.seconds2dhmsm(self._estimated_remaining_time_).split('.')[0] + ']'
                t2 = 'ITC: ' + sITC + '      LIC: ' + sLIC + '\n'
                t2 += 'TTC: ' + sTTC + "\n"
                t2 += 'ERT: ' + sERT + '\n'
                percentage = int(10000*(self._computed_steps_/self._num_iterations)) / 100
                t2 += f'Iterations done: {self._computed_steps_}/{self._num_iterations} ~ {percentage}%\n'
                plt.text(-0.5, 9.5, t2, color='darkblue', fontsize=22, style='normal', ha='left',
                         va='top', wrap=True)
                t3 = 'Graph saved at:: ' + save_time + '\n'
                t3 += 'Start at :: ' + self._str_started_time_ + '\n'
                if self._computed_steps_ == self._num_iterations:
                    t3 += 'E end at:: NOW'
                else:
                    t3 += 'E end at:: ' + str(self._estimated_end_time_)[:19]
                plt.text(-0.5, 4.25, t3, color='red', fontsize=22, style='normal', ha='left',
                         va='top', wrap=True)
                local_IP = socket.gethostbyname(socket.gethostname())
                local_machine_name = socket.gethostname()
                t4 = "Running on <" + local_machine_name + '@' + local_IP + '>'
                plt.text(-1, 0.25, t4, color='black', fontsize=20, style='normal', ha='left',
                         va='top', wrap=True)

            elif di == 'info1':
                face_color = 'honeydew'
                plt.axis((0, 10, 0, 10))
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.patch.set_alpha(0)
                exit_code_explanation = self._iterator_._exit_code_explanations(
                    self._iterator_.exit_code)
                TEXT = f"exit_code: {self._iterator_.exit_code}\n" \
                       f"{exit_code_explanation}\n"
                plt.text(0.1, 9, TEXT, color='black', fontsize=30,
                         ha='left', va='top', wrap=True)

            elif di == 't iteration':
                ylabel_backgroundcolor = 'greenyellow'
                face_color = 'snow'
                itime = np.array(self._times_)[indices]
                itime = self._filter_extreme_time(itime)  # extreme values are replaced by nan.
                valid_time = itime[~np.isnan(itime)]
                if len(valid_time) > 0:
                    max_time = np.max(valid_time)
                else:
                    max_time = 0.1

                if max_time < 999:
                    ylabel = 't iteration (s)'
                    unit = 's'
                elif max_time < 3600 * 3:
                    itime /= 60
                    ylabel = 't iteration (m)'
                    unit = 'm'
                else:
                    itime /= 3600
                    ylabel = 't iteration (h)'
                    unit = 'h'
                if len(indices) < 25:
                    plt.plot(indices, itime, '-o', color='k', linewidth=0.8, label='ITC')
                else:
                    plt.plot(indices, itime, color='k', linewidth=1.2, label='ITC')

                v10_ratio = None
                if len(valid_time) >= 10 and np.ndim(valid_time) == 1:
                    average = np.sum(valid_time) / len(valid_time)

                    last_10_time = valid_time[-10:]
                    average_last_10 = np.sum(last_10_time) / 10

                    if isinstance(average_last_10, (int, float)) and average_last_10 > 0:
                        if isinstance(average, (int, float)) and average > 0:
                            v10_ratio = average_last_10 / average

                else:
                    average = np.sum(valid_time) / len(valid_time)
                    average_last_10 = None

                if unit == 's':
                    pass
                elif unit == 'm':
                    average /= 60
                    if average_last_10 is not None:
                        average_last_10 /= 60
                elif unit == 'h':
                    average /= 3600
                    if average_last_10 is not None:
                        average_last_10 /= 3600
                else:
                    raise Exception()
                if indices[-1] > 1:
                    plt.plot([1, indices[-1]], [average, average],
                             color='red', linewidth=1.2, label='AVR')

                if average_last_10 is None:
                    pass
                else:
                    plt.plot([1, indices[-10]], [average_last_10, average_last_10], '--',
                             color='blue', linewidth=1.2)
                    # noinspection PyTypeChecker
                    plt.plot([indices[-10], indices[-1]], [average_last_10, average_last_10],
                             color='blue', linewidth=1.2, label='V10')

                if len(indices) <= 5:
                    ax.set_xticks(indices)
                plt.legend(fontsize=16, loc='best', frameon=False)

            elif di == 't accumulation':
                ylabel_backgroundcolor = 'greenyellow'
                face_color = 'snow'
                TIME = np.array(self._TIMES_)[indices]
                MAX_TIME = TIME[-1]
                if MAX_TIME < 60 * 3:
                    ylabel = 'Cumulative t (s)'
                elif MAX_TIME < 3600 * 3:
                    TIME /= 60
                    ylabel = 'Cumulative t (m)'
                elif MAX_TIME < 3600 * 24 * 3:
                    TIME /= 3600
                    ylabel = 'Cumulative t (h)'
                else:
                    TIME /= 3600 * 24
                    ylabel = 'Cumulative t (d)'
                if len(indices) < 25:
                    plt.plot(indices, TIME, '-o', color='k', linewidth=0.8)
                else:
                    plt.plot(indices, TIME, color='k', linewidth=1.2)
                if len(indices) <= 5:
                    ax.set_xticks(indices)

                if not over:

                    sERT = MyTimer.seconds2dhmsm(self._estimated_remaining_time_).split('.')[0] + ']'
                    y_position = float(0.9 * TIME[-1])
                    plt.text(0, y_position, 'ERT: ' + sERT, color='darkblue',
                             fontsize=22, style='normal', ha='left',
                             va='top', wrap=True)
                    # noinspection PyUnboundLocalVariable
                    if v10_ratio is not None:
                        # noinspection PyUnboundLocalVariable
                        v10_ERT_seconds = self._estimated_remaining_time_ * v10_ratio
                        vERT = MyTimer.seconds2dhmsm(v10_ERT_seconds).split('.')[0] + ']'
                        y_position = float(0.75 * TIME[-1])
                        plt.text(0, y_position, 'V10: ' + vERT, color='purple',
                                 fontsize=22, style='normal', ha='left',
                                 va='top', wrap=True)

            else:
                color_index = i - 6
                cmap_index = color_index % 8
                y_data = np.array(RDF[di])

                # --------- auto changing small values into log10 plot ---------------------------------

                if np.all(np.logical_or(y_data > 0, np.isnan(y_data))):
                    small_values = list(y_data < 1e-3)
                    total_ = len(small_values)
                    num_small_values = small_values.count(True)
                    if num_small_values / total_ > 0.75:
                        y_data = np.log10(y_data)
                        ylabel = r"lg10: " + ylabel
                        ylabel_backgroundcolor = 'orchid'
                        sci_format_y = False
                    else:
                        pass
                else:
                    pass

                # ======================================================================================

                if 't' in RDF:
                    if di == 't':
                        if len(indices) <= 10:
                            plt.plot(indices, y_data, '-o', color=colors(cmap_index), linewidth=1.5)
                        else:
                            plt.plot(indices, y_data, color=colors(cmap_index), linewidth=1.5)
                        plt.xlim([min(indices), max(indices)])
                        plt.xlabel('iterations')
                        if len(indices) <= 5:
                            ax.set_xticks(indices)
                    else:
                        if len(indices) <= 10:
                            plt.plot(RDF['t'], y_data, '-o', color=colors(cmap_index), linewidth=1.5)
                        else:
                            plt.plot(RDF['t'], y_data, color=colors(cmap_index), linewidth=1.5)
                        a, b = min(RDF['t']), max(RDF['t'])
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a < b:
                            plt.xlim([a, b])
                else:
                    if len(indices) <= 10:
                        plt.plot(indices, y_data, '-o', color=colors(cmap_index), linewidth=1.5)
                    else:
                        plt.plot(indices, y_data, color=colors(cmap_index), linewidth=1.5)
                    plt.xlim([min(indices), max(indices)])
                    plt.xlabel('iterations')
                    if len(indices) <= 5:
                        ax.set_xticks(indices)

            ax.tick_params(axis="x", direction='in', length=8, labelsize=15)
            ax.tick_params(axis="y", direction='in', length=8, labelsize=15)

            if sci_format_y:
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            else:
                plt.ticklabel_format(style='sci', axis='y', scilimits=(-20, 2))

            tx = ax.yaxis.get_offset_text()
            # ... change the yticks sci-format font size
            tx.set_fontsize(15)
            ax.set_ylabel(ylabel, fontsize=17, backgroundcolor=ylabel_backgroundcolor)

            # ITERATING watermark ...
            if i < 4:  # 0,1,2,3 for regular plots, no ITERATING
                pass
            elif di == 'solver message':  # no ITERATING for solver message subplot
                pass
            elif di == 'machine load':  # no ITERATING for machine load subplot
                pass
            elif not over:  # if not the last iteration, texture it.
                the_text = 'ITERATING'
                text = ax.text(0.1, 0.5, the_text, fontsize=65, color='gray',
                               horizontalalignment='left',
                               verticalalignment='center',
                               transform=ax.transAxes)
                text.set_alpha(.2)
            else:
                pass

            # facecolor ...
            if i < 4:  # regular subplots always have face color
                ax.set_facecolor(face_color)
            elif di == 'solver message':  # solver message subplot always have face color
                ax.set_facecolor(face_color)
            elif di == 'machine load':  # machine load subplot always have face color
                ax.set_facecolor(face_color)
            elif not over:  # only have facecolor if it is not the last iteration.
                ax.set_facecolor(face_color)
            else:
                pass

            # .. subplots done ...
            super_title = "    phyem ITERATOR     \n> {}/{} <".format(
                len(self._times_) - 1, self._num_iterations
            )
            st_fontsize = 100

            alpha = (3.1 * x_len / 17.4) / y_len
            beta = 0.6 + (0.3-0.6)*(r_num_subplots-3)/2
            if over:
                fig.suptitle(super_title, fontsize=st_fontsize, backgroundcolor='seagreen', y=1 + alpha*beta)
            else:
                fig.suptitle(super_title, fontsize=st_fontsize, backgroundcolor='tomato', y=1 + alpha*beta)

            # noinspection PyBroadException

        try:  # in case saving fail, we just skip it as this is not that important.
            plt.savefig(
                f'{self.name}.png',
                dpi=125,
                bbox_inches='tight',
                facecolor='honeydew',
            )
        except:  # in case like PermissionError
            pass

        plt.close(fig)
        matplotlib.use('TkAgg')  # make sure we use the right backend.

    def _select_reasonable_amount_of_data(self, max_num, last_num=1):
        """To report RDF, we do not report all, we make a selection.

        :param max_num: we at most selection this amount of data
        :param last_num: we must select the last ``last_num`` rows of RDF.
        """
        assert max_num >= 10
        assert 1 <= last_num < max_num
        all_data_num = len(self._iterator_.RDF)
        if max_num < all_data_num:
            indices = list(np.linspace(0, all_data_num-last_num, max_num+1-last_num).astype(int))
            indices.extend([all_data_num+1-last_num+i for i in range(last_num-1)])
        else:
            indices = [i for i in range(all_data_num)]
        return indices

    @staticmethod
    def _filter_extreme_time(times):
        """
        We remove some extreme values to make the iteration time plot to be of higher resolution.

        Note that, after we have removed some extreme value, the iteration time plot may look
        very weird. For example, the average iteration time may be greater than all iteration times.
        This is okay, since we may have removed a huge iteration time. This should disappear after
        we have a large amount of iterations.
        """
        valid_time = times[~np.isnan(times)]

        if len(valid_time) > 0:
            avg = np.mean(valid_time)
            mxt = np.max(valid_time)

            if mxt > 2 * avg and len(valid_time[valid_time > 0.9*mxt]) == 1:
                # there is only one huge value. This happens in some, for example, TGV cases.
                TIME = np.nan_to_num(times)
                max_ind = np.argwhere(TIME > 0.9*mxt)[:, 0]
                times[max_ind] = np.nan
            else:
                pass

        else:
            pass

        return times
