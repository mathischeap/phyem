# -*- coding: utf-8 -*-
"""
Runner Data. Here we wrap the `_rdf_` (Result DataFrame) into a seperate class in case
that in the future we what to future extend it.

We normalize it with the Parent `DFW` from `accessories.deta_structures`.

Yi Zhang (C)
Created on Tue Apr 16 11:12:11 2019
Aerodynamics, AE
TU Delft
"""
import pandas as pd

from phyem.tools.frozen import Frozen


class DataStructure(Frozen):
    """ """
    def __init_DS__(self):
        """ """


class DFW(DataStructure):
    """
    In the future, maybe from multiple places, we will wrap a pandas DataFrame data
    into some customed class. We will use this template as their parent. So we can
    actually set up some communications between them.

    This is a very special one, itself has no meaning, but in particular class, we can
    inherit this one to generate `DataFrame` like data.

    Because of this, it does not take its component visualization as an attribute here.

    """
    def __init__(self, dfd):
        """ """
        super().__init_DS__()
        assert isinstance(dfd, pd.DataFrame), \
            " <DataFrameWrapper> : I need a DataFrame to initialize."
        self._dfd_ = dfd
        self._freeze()

    @classmethod
    def ___file_name_extension___(cls):
        return '.dfw'

    def __call__(self):
        return self._dfd_


from phyem.tools.legacy.serialRunners.COMPONENTS.data.COMPONENTS.visualize import RunnerDataVisualize


class RunnerData(DFW):
    """ """
    def __init__(self, runner):
        """ """
        super().__init__(runner.rdf)
        self._melt()
        self._runner_ = runner
        self._visualize_ = RunnerDataVisualize(self)
        self._freeze()
    
    @classmethod
    def ___file_name_extension___(cls):
        return ".rd"
    
    @property
    def visualize(self):
        return self._visualize_