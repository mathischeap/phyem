
"""
Export to corresponding vtk files.
"""
import numpy as np
from tools.frozen import Frozen
from pyevtk.hl import unstructuredGridToVTK


class MsePyRootFormVisualizeVTK(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._freeze()

    def __call__(self, *args, **kwargs):
        abs_sp = self._f.space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(*args, **kwargs)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(*args, **kwargs)

    def _m3_n3_k0(self):
        """"""
        print(self._mesh.topology.corner_numbering)
        print(self._mesh.topology.edge_numbering)

    def _m3_n3_k1(self, sampling_factor=1):
        """"""
        samples = 100000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples > 50:
            samples = 50
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et_sg = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        xyz, uvw = self._f[t].reconstruct(xi_et_sg, xi_et_sg, xi_et_sg)  # ravel=False by default


    def _m3_n3_k2(self, sampling_factor=1):
        """"""
        samples = 100000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples > 50:
            samples = 50
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et_sg = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        xyz, uvw = self._f[t].reconstruct(xi_et_sg, xi_et_sg, xi_et_sg)  # ravel=False by default


    def _m3_n3_k3(self, sampling_factor=1):
        """"""
        samples = 100000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples > 50:
            samples = 50
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et_sg = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        xyz, u = self._f[t].reconstruct(xi_et_sg, xi_et_sg, xi_et_sg)  # ravel=False by default
