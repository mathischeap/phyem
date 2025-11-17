# -*- coding: utf-8 -*-
"""
A parent class for all serial_runners.

Yi Zhang (C)
Created on Thu Apr 11 10:36:36 2019
Aerodynamics, AE
TU Delft
"""
import pickle
import types
import pandas as pd

from phyem.tools.frozen import Frozen
from phyem.tools.miscellaneous.timer import MyTimer
from phyem.tools.miscellaneous.numpy_styple import NumpyStyleDocstringReader
from phyem.tools.legacy.serialRunners.COMPONENTS.data.body import RunnerData


class Runner(Frozen):
    """Parent for all serial_runners."""
    def __init__(self, solver=None, ___file___=None, task_name=None):
        """
        Parameters
        ----------
        solver ：
            The solver. If solver is set to None, then we probably only use it
            to read data from somewhere, like '.txt' file, and then plot the 
            data.
        ___file___ : str
            The builtin variable `___file___`.
            
        """
        print("\n=<Runner>=" + MyTimer.current_time() + '======', flush=True)
        if solver is not None:
            # noinspection PyTypeChecker
            if isinstance(solver, types.FunctionType):
                pass
            elif isinstance(solver, types.MethodType):
                pass
            elif solver.__class__.__name__ == 'CPUDispatcher':
                pass
            else:
                raise NotImplementedError()
            self._solver_ = solver
            docstring = NumpyStyleDocstringReader(self._solver_)
            self._input_names_ = docstring.Parameters
            self._output_names_ = docstring.Returns
            self._num_inputs_ = len(self._input_names_)
            self._num_outputs_ = len(self._output_names_)
            if  len(self._input_names_) > 3:
                self._input_names_ = self._input_names_[:3]
                self._kw_input_names_ = self._input_names_[3:]
            else:
                self._kw_input_names_ = tuple()

        else:
            self._solver_ = None
            self._input_names_ = None
            self._output_names_ = None
            self._num_inputs_ = 0
            self._num_outputs_ = 0
            self._kw_input_names_ = tuple()

        print("-<dir>-<{}>-".format(___file___), flush=True)
        self.___file___ = ___file___
        self.___kwargs___ = None
        self._results_ = None
        self._rdf_ = None
        self._info_ = None
        self._num_iterations_ = None
        self._computed_pool_ = ()
        self._readwritefilename_ = None
        self._task_name_ = task_name
        self._solver_source_code_ = '    unknown'
        self._solver_dir_ = '    unknown'
        self._criterion_ = None
        
    @property
    def visualize(self):
        """ 
        Access the visualizers. 
        
        This is not good since everytime we call this property, we generate a new 
        `visualize` instance. But this is very convenient and we do not care about that
        the data change when we re-initialize `self._rdf_` or `self._rdf_` is `None`.
        
        """
        return self.data.visualize
    
    @property
    def data(self):
        """ This is not efficient but is convenient, see docstring of `visualize`."""
        return RunnerData(self)
    
    @property
    def solver(self):
        """ Access the solver."""
        return self._solver_
    
    @property
    def results(self):
        """ 
        The result, not necessarily be a `DataFrame`, can be, for example, a dict.
        
        """
        return self._results_
    
    @property
    def task_name(self):
        return self._task_name_
    
    @property
    def input_names(self):
        """ Access the input names."""
        return self._input_names_
    
    @property
    def num_inputs(self):
        """ How many inputs."""
        return self._num_inputs_
    
    @property
    def output_names(self):
        """ Access the output names."""
        return self._output_names_
    
    @property
    def num_outputs(self):
        """ How many inputs."""
        return self._num_outputs_
    
    @property
    def num_iterations(self):
        """ How many iterations."""
        return self._num_iterations_
    
    @property
    def readwritefilename(self):
        """ We will do real time writing to this file."""
        return self._readwritefilename_

    @property
    def solver_source_code(self):
        return self._solver_source_code_

    @property
    def solver_dir(self):
        return self._solver_dir_

    @property
    def rdf(self):
        """ The `results` in `DataFrame` format."""
        return self._rdf_
    
    def ___initialize_dataframe___(self):
        """ 
        Before iterations start, we initialize our attribute `self._rdf_`.
        
        `___step` methods are those methods you almost have to go through when do 
        method `iterate`.
        
        Attributes
        ----------
        self._rdf_ :
            
        """
        self._rdf_ = pd.DataFrame(columns=[*self._input_names_, 
                                           *self._output_names_,
                                           'ITC', 'TTC', 'ERT'])
    
    @classmethod
    def readfile(cls, readfilename):
        """ 
        We generate a Runner instance by reading rdf from files. The instance itself
        will have to been generated from the particular class.
        
        """
        # noinspection PyUnresolvedReferences
        R = cls.___generate_self_empty_copy___()
        R.___readcontents___(readfilename, initializing=True)
        return R
    
    def ___readcontents___(self, readfromfilename, initializing=False):
        """ 
        We use this method to read contents from files.
        
        It works in two ways:
            When `initializing` is True, we actually are reading results from a file to
            a empty runner instance. So we have to read information, like input, output
            names. And we also initialize a `rdf` for this empty runner instance.
            
            When `initializing` is False, then we are actually try to continue an
            iterations, so we check information, like input, output names, and we do 
            not initialize a `rdf`.
        """
        with open(readfromfilename, 'r') as f: 
        # `writeto` is not None, and `writeto` is already there
            contents = f.readlines()
            #_____ Check format ___________________________________________________
            assert contents[0][1:7] == 'Runner', " <Runner> : is not a Runner!"
            assert self.__class__.__name__ in contents[0], \
                " <Runner> : This file is not readable for {}.".format(self.__class__.__name__)
            assert '<RESULTS> : \n' in contents, " <Runner> : No results?"
            for i, ctt in enumerate(contents):

                if ctt[0:13] == '<input names>':
                    if initializing: # if we are initializing it, we get input names from the file
                        ins = ctt[17:-2]
                        self._input_names_ = tuple(ins.replace("'", "").split(', '))
                    else: #otherwise we check the consistence
                        assert ctt == '<input names> : '+str(self._input_names_)+'\n', \
                            " <Runner> : writeto file does not match input names."

                elif ctt[0:17] == '<input criterion>':
                    ipc = ctt.split(' : ')[1][:-1]
                    if initializing: # if we are initializing it, we get input names from the file
                        self._criterion_ = ipc
                    else:

                        assert self._criterion_ == ipc, f"<input criterion> {self._criterion_}!={ipc} do not match."

                elif ctt[0:14] == '<output names>':
                    if initializing: # if we are initializing it, we get output names from the file
                        ons = ctt[18:-2]
                        self._output_names_ = tuple(ons.replace("'", "").split(', '))
                    else: #otherwise we check the consistence
                        assert ctt == '<output names> : '+str(self._output_names_)+'\n', \
                            " <Runner> : writeto file does not match output names."

                elif ctt[0:8] == '<kwargs>':
                    if initializing:  # if we are initializing it, we get output names from the file
                        self.___kwargs___ = ctt.split(' : ')[1][:-1]
                    else:
                        if ctt == '<kwargs>' + ' : ' + str(self.___kwargs___) + '\n':
                            pass
                        else:
                            print("-<Runner>-<Warning> : kwargs do not match......")

                elif ctt[0:9] == '<RESULTS>':
                    i += 3 # we skip the column names row and `-----------` row
                    break

                else:
                    pass
            if initializing: # if we are initializing it, we initialize the rdf.
                self.___initialize_dataframe___()
            else: #otherwise we do nothing
                pass
            #----------------------------------------------------------------------
            while 1:
                try:
                    Di = contents[i].split()
                    assert len(Di) == 1 + len(self.input_names) + len(self.output_names) + 1 + 3
                    # 1: index; +1: '|'; +3: 'ITC', 'TTC', 'ERT'
                    self._computed_pool_ += (self.___read_to_rdf___(Di)[0],)
                    i += 1
                except IndexError:
                    # we reach the end of the file.
                    break
                except AssertionError:
                    # we see not-compete row, so we stop here.
                    break
        return i, contents
        
    def ___prepare_write_file___(self, writeto):
        """
        We check what we have to do with the file `writeto`. If `writeto` is None, then
        nothing happens we just do nothing and return.
        
        If `writeto` is not None, and `writeto` is already there, then we check:
            1). its format is correct a DataFrame?
        If above checks are passed, then we read it into `self._rdf_`.
        
        If `writeto` is not None, and we do not find `writeto`. Then we create one and
        initialize it.
        
        Parameters
        ----------
        writeto : str
            The real-time-writing file name. Will be saved to 
            `self._readwritefilename_`, and can be accessed by property
            `self.readwritefilename`.
            
        Attributes
        ----------
        self._readwritefilename_ :
        self._computed_pool_ :
        
        """
        print("-<{}>-<total iterations> : {}.".format(self.__class__.__name__, self.num_iterations), flush=True)
        self.___initialize_dataframe___()
        if writeto is None:
            return
        self._readwritefilename_ = writeto
        try:
            I, contents = self.___readcontents___(writeto)
            with open(self.readwritefilename, 'w') as f: 
                for j in range(I):
                    f.write(contents[j])
        except FileNotFoundError: # `writeto` is not None, and we do not find `writeto`
            # we do initializing the writing file.
            self.___initial_writing___()
            
    def ___initial_writing___(self):
        """ Initialize the writing process."""
        with open(self.readwritefilename, 'w') as f:
            # we initialize the writing field..........
            f.write('<Runner>-<{}>-<task name: {}>\n'.format(self.__class__.__name__, self.task_name))
            f.write('{}\n'.format(MyTimer.current_time()))
            f.write('<dir> : '+str(self.___file___)+'\n')
            f.write('<solver_dir> :\n' + self.solver_dir + '\n\n')
            f.write('<solver_source_code> :\n' + self.solver_source_code + '\n\n')
            f.write('<input names> : '+str(self._input_names_)+'\n')
            f.write('<input criterion> : '+str(self._criterion_)+'\n')
            f.write('<output names> : '+str(self._output_names_)+'\n')
            f.write('<kwargs> : '+str(self.___kwargs___)+'\n')
            f.write('<INFO> : '+str(self._info_)+'\n\n')
            f.write('<RESULTS> : \n')
            f.write('{: <15}'.format('index'))
            f.writelines(['{: <25}'.format(opi) for opi in self._input_names_])
            f.writelines(['{: <25}'.format(opi) for opi in self._output_names_])
            f.write('| ')
            f.write('ITC(W)         ')
            f.write('TTC(W)         ')
            f.write('ERT(W)       \n')
            f.write(''+'---------------' + '-'*3*25 + '-'*25*len(self._output_names_) + 
                    '|-'+'-'*15 + '-'*15 + '-'*13 + '\n')
            
    def ___update_rdf___(self, index, inputs, outputs, ITC, TTC, ERT, show_progress=True):
        """ Update `self._rdf_` after each iteration."""
        if show_progress:
            print(':::: Index >>> {} <<< added to ResultDataFrame ::::\n'.format(index))
        ndf = pd.DataFrame([[*inputs, *outputs, ITC, TTC, ERT],], index=[index,],
                           columns=[*self._input_names_, *self._output_names_, 'ITC', 'TTC', 'ERT'])

        if len(self._rdf_) == 0:
            self._rdf_ = ndf
        else:
            self._rdf_ = pd.concat([self._rdf_, ndf])
        # self._rdf_ = self._rdf_.append(ndf)
        self._computed_pool_ += ([*inputs],)
        
    def ___write_to___(self, m):
        """ Write the result in the `writeto` file."""
        if self._readwritefilename_ is None:
            return
        with open(self.readwritefilename, 'a') as f:
            f.write('{: <14} '.format(str(m)))
            for i in range(len(self.input_names)):
                if self._rdf_[self._input_names_[i]][m]%1==0:
                    f.write('{: <23}  '.format(int(self._rdf_[self._input_names_[i]][m])))
                else:
                    f.writelines('{: <23}  '.format('%.16e'%self._rdf_[self._input_names_[i]][m]))
            f.writelines(['{: <23}  '.format('%.16e'%self._rdf_[self._output_names_[i]][m]) 
                            for i in range(len(self._output_names_))])
            f.write('| ')
            f.write('{: <13}  '.format(self._rdf_['ITC'][m]))
            f.write('{: <13}  '.format(self._rdf_['TTC'][m]))
            f.write('{: <13}'.format(self._rdf_['ERT'][m]) + '\n')
    
    def ___read_to_rdf___(self, Di):
        """ Used in restarting iterations."""
        Di.remove('|')
        data = [float(i) for i in Di[1:-3]]
        data = [int(i) if i%1==0 else i for i in data]
        ndf = pd.DataFrame([data + Di[-3:],], index=[int(Di[0]),], 
                            columns=[*self._input_names_, *self._output_names_, 
                                     'ITC', 'TTC', 'ERT'])
        if len(self._rdf_) == 0:
            self._rdf_ = ndf
        else:
            self._rdf_ = pd.concat([self._rdf_, ndf])
        # self._rdf_ = self._rdf_.append(ndf)
        return [data[0:len(self._input_names_)], ]
    
    def ___check_inputs_in_computed_pool___(self, inputs):
        """
        We use this method to check in `inputs` is in `self._computed_pool_`.
        
        Parameters
        ----------
        inputs : list or tuple
            The inputs to be checked.
        
        Returns
        -------
        output : bool
        
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)
        if inputs in self._computed_pool_:
            return True
        # ___ I am a little bit worried about this, since when we read
        # from .txt file, we may lose some digits then make this `in`
        # estimation not valid. But so far, there is no problem. I 
        # tested it with pi and e. It works. I believe this is because
        # we write %.16e in to txt file, which is already accurate 
        # enough for float. In the future, if something wrong happend, look at the 
        # commented code below.
#        assert all([isinstance(inputs[i], (int, float)) for i in range(self.num_inputs)]), \
#            " <Runner> : Currently, we only handle int, float inputs"
#        assert all([all([isinstance(icp[i], (int, float)) for i in range(self.num_inputs)]) 
#                        for icp in self._computed_pool_]), \
#                            " <Runner> : Currently, we only handle int, float inputs"
#        for icp in self._computed_pool_:
#            if all([np.abs(icp[i]-inputs[i])<1e-8 for i in range(self.num_inputs)]):
#                return True
#            else:
#                op =()
#                for j in range(self.num_inputs):
#                    if icp[j] == 0:
#                        op += (True,) if np.abs(icp[j]-inputs[j])<1e-8 else (False,)
#                    else:
#                        op += (True,) if np.abs(icp[j]-inputs[j])/np.abs(icp[j])<1e-6 else (False,)
#                op = all(op)
#                if op:
#                    return True
        return False


    def ___deal_with_saveto___(self, writeto, saveto):
        """ 
        Here we figure out how to save the Runner. This method should be called in 
        every `iterate` method.
        
        """
        if isinstance(saveto, str):
            self.saveto(saveto)
        else:
            if writeto is None:
                pass
            else:
                if saveto:
                    saveto = writeto.split('.')[0]
                    self.saveto(saveto)
                else:
                    pass

    def saveto(self, filename, what2save=None, print_info=True):
        """
        We use the method to save this self to a file. Of course, this is done with the
        usage of package pickle.

        Parameters
        ----------
        filename : str
            The directory to save.
        what2save :
        print_info :
        """
        assert isinstance(filename, str) and '.' not in filename, \
            " <SaveRead> : filename need be str without extension."
        # noinspection PyUnresolvedReferences
        filename += self.___file_name_extension___()
        if print_info:
            print('-<S>-<{}>-<{}>------'.format(self.__class__.__name__, filename))
        if what2save is None:
            with open(filename, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        else:
            filename += '_' + what2save
            with open(filename, 'wb') as output:
                pickle.dump(getattr(self, what2save), output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def readfrom(cls, filename):
        """
        We use this classmethod to read a file.

        Parameters
        ----------
        filename : str
            The directory we will read the object from.
        """
        assert isinstance(filename, str), \
            " <SaveRead> : filename wrong! give me a str, now is {}".format(
                filename.__class__.__name__)
        assert '.' in filename, " <SaveRead> : please add extension in filename."

        extension = '.' + filename.split('.')[1]  # we get the extension of the file to be read
        # we check that the starting part of the extension need to fit the class's requirement
        # noinspection PyUnresolvedReferences
        assert extension[:len(cls.___file_name_extension___())] == cls.___file_name_extension___(), \
            " <SaveRead> : file named `{}` is can not read by this class.".format(filename)
        # If we have more characters in extension, then we know we are just read a property of
        # the class. So we first check if the class have such a property.
        # noinspection PyUnresolvedReferences
        if len(extension) != len(cls.___file_name_extension___()):
        # noinspection PyUnresolvedReferences
            assert extension[len(cls.___file_name_extension___()) + 1:] in dir(cls), \
                " <SaveRead> : can not find property:{} in this class.".format(
                    extension[len(cls.___file_name_extension___()) + 1:])
        with open(filename, 'rb') as inputs:
            obj = pickle.load(inputs)
        return obj
