# -*- coding: utf-8 -*-
from phyem.tools.frozen import Frozen


class ParallelRunnerBase(Frozen):
    """A template for all parallel runners."""
    def __init__(self):
        """"""
        self.___lock_iterate___ = False

    def iterate(self, *args, **kwargs):
        if self.___lock_iterate___:
            raise Exception('This parallel runner is locked; '
                            'it can not run ``iterate`` function. '
                            'This is probably because it is read from a file. '
                            'So it lacks a solver.')
        else:
            return self.___iterate___(*args, **kwargs)

    @property
    def visualize(self):
        return self.___visualize___

    @property
    def results(self):
        return self.___results___

    # A parallel runner must have the following methods and properties ___
    def ___iterate___(self, *args, **kwargs):
        return NotImplementedError()

    @property
    def ___visualize___(self):
        return NotImplementedError()

    @property
    def ___results___(self):
        return NotImplementedError()
