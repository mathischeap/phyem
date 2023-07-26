# -*- coding: utf-8 -*-
import numpy as np
from tools.frozen import Frozen
from msepy.mesh.boundary_section.visualize.matplot import Matplot


class MsePyBoundarySectionVisualize(Frozen):
    """"""

    def __init__(self, bs):
        """"""
        self._bs = bs
        self._matplot = Matplot(bs)
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        self._matplot(*args, **kwargs)

    @property
    def matplot(self):
        return self._matplot

    def _generate_element_faces_of_this_boundary_section(self, sampling_factor):
        """"""
        samples = int(25 * sampling_factor)
        if samples <= 3:
            samples = 3
        elif samples >= 100:
            samples = 100
        else:
            pass

        n = self._bs.n

        lin_space = np.linspace(-1, 1, samples)

        lin_spaces = [lin_space for _ in range(n)]

        data = dict()

        for i in self._bs.faces:
            element_face = self._bs.faces[i]
            xyz = element_face.ct.mapping(*lin_spaces)
            data[i] = xyz

        return data
