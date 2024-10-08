# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from importlib import import_module
from src.config import SIZE
assert SIZE == 1, "Runners can only be run in single thread."


class RunnerExplorer(Frozen):
    """Runner to collect data or run function on single thread. Then we can do plot."""
    def __init__(self, ID):
        assert ID in self.___defined_runners___(), \
            " <RunnerExplorer> : {} is not coded.".format(ID)
        module_name, cls_name = self.___defined_runners___()[ID].split(' : ')
        cls_path = self.___path___() + module_name
        self._runner_class_ = getattr(import_module(cls_path), cls_name)
        self._freeze()
        
    def __call__(self, *arg, **kwargs):
        return self._runner_class_(*arg, **kwargs)
        
    @classmethod
    def ___defined_runners___(cls):
        """ 
        The following `doctest` is just for testing purpose. So, do not think too much.
        
        doctest
        -------
            >>> R = RunnerExplorer('m3ir')
            >>> R.___defined_runners___()['m3ir']
            'matrix3d_input_runner : Matrix3dInputRunner'
            
        """
        return {'m3ir': 'matrix3d_input_runner : Matrix3dInputRunner',
                'tir': 'three_inputs_runner : ThreeInputsRunner'}
    
    @classmethod
    def ___path___(cls):
        return 'TOOLS.__DEPRECATED__.serial_runners.INSTANCES.'


if __name__ == "__main__":
    R = RunnerExplorer('m3ir')()

    import doctest
    doctest.testmod()
