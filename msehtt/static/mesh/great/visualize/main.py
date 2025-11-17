# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.mesh.great.visualize.matplot import MseHttGreatMeshVisualizeMatplot
from phyem.msehtt.static.mesh.great.visualize.ts import MseHttGreatMeshVisualize_TS_Hierarchy


class MseHttGreatMeshVisualize(Frozen):
    r""""""

    def __init__(self, tgm):
        r""""""
        self._tgm = tgm
        self._matplot = None
        self._ts_ = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        r"""Use the default visualizer to visualize the great mesh."""
        return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        r""""""
        if self._matplot is None:
            self._matplot = MseHttGreatMeshVisualizeMatplot(self._tgm)
        return self._matplot

    @property
    def ts(self):
        r"""To plot the hierarchy of ts meshing!"""
        if self._ts_ is None:
            self._ts_ = MseHttGreatMeshVisualize_TS_Hierarchy(self._tgm)
        return self._ts_

    def _generate_element_outline_data(self, ddf=1, internal_grid=0):
        r""""""
        outline_data = {}
        for i in self._tgm.elements:
            element = self._tgm.elements[i]
            outline_data[i] = element._generate_outline_data(ddf=ddf, internal_grid=internal_grid)
        return outline_data

    def _generate_element_quality_data(self):
        r"""return a dict whose keys are element indices and values are the factor indicating the quality of
        the elements.

        When the factor is 0: the element is worst.
        When the factor is 1: the element is best.

        Returns
        -------

        """
        quality_data = {}
        for i in self._tgm.elements:
            element = self._tgm.elements[i]
            quality_data[i] = element._find_element_quality()
        return quality_data
