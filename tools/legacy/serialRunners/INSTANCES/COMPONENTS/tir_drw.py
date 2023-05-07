# -*- coding: utf-8 -*-
"""
The Data Reader & Writer (drw) for class ThreeInputsRunner (TIR).

Yi Zhang (C)
Created on Sat Dec  1 13:38:55 2018
Aerodynamics, AE
TU Delft
"""
import inspect
from tools.miscellaneous.timer import MyTimer
from tools.frozen import Frozen

class TIR_DRW(Frozen):
    """ """
    def __init__(self, TIR):
        """ """
        self._TIR_ = TIR
        self._readwritefilename_ = None
        self._computed_m_ = []
        self._freeze()
    
    @property
    def readwritefilename(self):
        return self._readwritefilename_
    
    @readwritefilename.setter
    def readwritefilename(self, filename):
        """ """
#        assert isinstance(filename, str) and '.' not in filename, \
#            " <TIR_DRW> : I need a string without extension."
#        self._readwritefilename_ = filename + '.txt'
        assert isinstance(filename, str)
        self._readwritefilename_ = filename
    
    def read(self, filename):
        """ 
        Currently, we can only use this method within an TIR instance, (not as
        a classmethod). We can make another classmethod to use it read a '.txt'
        file and return a TIR instance will be very useful in the sense that
        we then can do tecplot from a '.txt' file not necessarily from a '.3ir'
        file.
        
        """
        self.readwritefilename = filename
        try:
            with open(self.readwritefilename, 'r') as f:
                fstr = f.readlines()
            # if we find the file, we then check it and store info to `self._TIR_.results`.
            total_lines = len(fstr)
            assert fstr[0][:-1] == '<ThreeInputsRunner>', " <TIR_DRW> : I need a <ThreeInputsRunner> file."
            i = fstr.index('<inputs>:\n')
            input_0, I0sequence = fstr[i+1].split(' sequence: ')
            input_1, I1sequence = fstr[i+2].split(' sequence: ')
            input_2, I2sequence = fstr[i+3].split(' sequence: ')
            assert self._TIR_._input_names_ == (input_0, input_1, input_2)
            assert I0sequence[:-1] == str(self._TIR_._I0seq_), " <TIR_DRW> : input[0] wrong."
            assert I1sequence[:-1] == str(self._TIR_._I1seq_), " <TIR_DRW> : input[1] wrong."
            assert I2sequence[:-1] == str(self._TIR_._I2seq_), " <TIR_DRW> : input[2] wrong."
            i = fstr.index('<kwargs>:\n')
            kwargs = fstr[i+1][:-1]
            assert kwargs == str(self._TIR_.___kwargs___), " <TIR_DRW> : kwargs wrong."
            i = fstr.index('<results>:\n')
            i += 1
            stored = fstr[i].split()
            num_stored = len(stored)
            j = stored.index('|')
            outputs = stored[4:j]
            assert tuple(outputs) == self._TIR_._output_names_, " <TIR_DRW> : output names wrong."
            while i < total_lines:
                try:
                    fstr_i_split = list(fstr[i].split())
                    m = int(fstr_i_split[0])
                    if len(fstr_i_split) == num_stored:
                        if fstr[i][-1] != '\n':
                            fstr[i] += '\n'
                        self._computed_m_.append(m)
                        # when this is happening, we stored full values at this line, so we can keep it to `self._TIR_.results`.
                        for k in range(j):
                            fstr_i_split[k] = float(fstr_i_split[k])
                        
                        self._TIR_.___update_results___(*fstr_i_split[1:4], fstr_i_split[4:j], *fstr_i_split[j+1:])
                    else:
                        break
                except ValueError:
                    pass
                i += 1
            with open(self.readwritefilename, 'w') as f:
                f.writelines(fstr[0:i])
        except FileNotFoundError:
            # we do not find the file, so we initialize one.
            self.initialize_writing()
    
    def initialize_writing(self):
        """ """
        with open(self.readwritefilename, 'w') as f:
            f.write('<ThreeInputsRunner>\n{}\n'.format(MyTimer.current_time()))
            f.write('%r\n\n'%self._TIR_._solver_)
            f.write('%r\n\n'%self._TIR_.___file___)
            f.write('<source>:\n')
            if self._TIR_._solver_ is None:
                f.write('None\n')
            else:
                f.write(inspect.getsource(self._TIR_._solver_) + '\n')
            f.write('<inputs>:\n')
            f.write('{} sequence: '.format(self._TIR_._input_names_[0]) + str(self._TIR_._I0seq_) + '\n')
            f.write('{} sequence: '.format(self._TIR_._input_names_[1]) + str(self._TIR_._I1seq_) + '\n')
            f.write('{} sequence: '.format(self._TIR_._input_names_[2]) + str(self._TIR_._I2seq_) + '\n\n')
            f.write('<kwargs>:\n')
            f.write('{}\n\n'.format(self._TIR_.___kwargs___))
            f.write('<results>:\n')
        self.____list_varialbes_write_file____()
    
    def ____list_varialbes_write_file____(self):
        with open(self.readwritefilename, 'a') as f:
            f.write('{: <5}'.format('i'))
            f.writelines(['{: <25}'.format(self._TIR_._input_names_[i]) for i in range(3)])
            f.writelines(['{: <25}'.format(opi) for opi in self._TIR_._output_names_])
            f.write('| ')
            f.write('time_cost(W)   ')
            f.write('total_cost(W)  ')
            f.write('ERT(W)       \n')
            f.write(''+'-----' + '-'*3*25 + '-'*25*len(self._TIR_._output_names_) + 
                    '|-'+'-'*15 + '-'*15 + '-'*13 + '\n')
    
    def write_iteration(self, m):
        """ 
        We write the results of mth iteration to file `self.readwritefilename`.
        
        """
        if self.readwritefilename is None: return
        if m in self._computed_m_: return
        if m % 100 == 0 and m != 0:
            self.____list_varialbes_write_file____()
        with open(self.readwritefilename, 'a') as f:
            f.write('{: <4} '.format(str(m)))
            for i in range(3):
                if isinstance(self._TIR_._results_[self._TIR_._input_names_[i]][m], int):
                    f.write('{: <23}  '.format(self._TIR_._results_[self._TIR_._input_names_[i]][m]))
                else:
                    f.writelines('{: <23}  '.format('%.16e'%self._TIR_._results_[self._TIR_._input_names_[i]][m]))
            f.writelines(['{: <23}  '.format('%.16e'%self._TIR_._results_[self._TIR_._output_names_[i]][m]) 
                for i in range(len(self._TIR_._output_names_))])
            f.write('| ')
            f.write('{: <13}  '.format(self._TIR_._results_['solver_time_cost'][m]))
            f.write('{: <13}  '.format(self._TIR_._results_['total_cost'][m]))
            f.write('{: <13}'.format(self._TIR_._results_['ERT'][m]) + '\n')
