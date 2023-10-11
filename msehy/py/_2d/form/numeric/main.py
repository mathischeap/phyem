# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.dds.region_wise_structured import DDSRegionWiseStructured
from msehy.py._2d.form.numeric.interp import MseHyPy2FormNumericInterp


class MseHyPy2RootFormNumeric(Frozen):
    """"""

    def __init__(self, f):
        self._f = f
        self._freeze()

    def region_wise_reconstruct(self, t, r, s, target='generic', method='linear', density=50):
        """

        Parameters
        ----------
        t
        r :
            2d array, all entries in [0, 1] since the interpolation is mesh-region-wise.
        s :
            2d array, all entries in [0, 1] since the interpolation is mesh-region-wise.
        target
        method
        density

        Returns
        -------

        """
        interp = MseHyPy2FormNumericInterp(self._f, t, target)(method, density=density)
        # mesh-region-wise interp functions.

        if not isinstance(r, np.ndarray):
            r = np.array(r)
        if not isinstance(s, np.ndarray):
            s = np.array(s)
        assert np.ndim(r) == np.ndim(s) == 2, f"r, s must be 2d."
        assert np.shape(r) == np.shape(s), f"r, s shape do not match."
        assert np.min(r) >= 0 and np.max(r) <= 1, f"all entries of r must be in [0,1]."
        assert np.min(s) >= 0 and np.max(s) <= 1, f"all entries of r must be in [0,1]."

        background = self._f.mesh.representative.background
        regions = background.manifold.regions
        Xd = dict()
        Yd = dict()

        results = None
        for region in interp:
            itp_s = interp[region]
            if results is None:
                results = list()
                for _ in itp_s:
                    results.append(
                        dict()
                    )
            else:
                assert len(results) == len(itp_s)

        for region in regions:
            x, y = regions[region]._ct.mapping(r, s)
            Xd[region] = x
            Yd[region] = y
            itp_s = interp[region]
            for j, itp in enumerate(itp_s):
                results[j][region] = itp(x, y)

        return DDSRegionWiseStructured([Xd, Yd], results)

    def region_wise_interp(self, t=None, density=50, saveto=None, method='linear'):
        """Reconstruct the form at time `time` and use the reconstruction results to make interpolation functions
        in each region.

        These functions take (x, y, ...) (physical domain coordinates) are inputs.

        Parameters
        ----------
        t
        density
        saveto
        method

        Returns
        -------

        """
        if t is None:
            t = self._f.generic.cochain.newest
        else:
            pass

        interp = MseHyPy2FormNumericInterp(self._f, t, 'generic')(
            method=method, density=density
        )   # mesh-region-wise interp functions.

        final_interp = dict()
        for region in interp:
            if len(interp[region]) == 1:
                final_interp[region] = interp[region][0]
            else:
                final_interp[region] = interp[region]

        if saveto is None:
            pass
        else:
            from src.config import RANK, MASTER_RANK
            if RANK == MASTER_RANK:
                # we are only calling one thread, so just go ahead with it.
                import pickle
                with open(saveto, 'wb') as output:
                    pickle.dump(final_interp, output, pickle.HIGHEST_PROTOCOL)
                output.close()

            else:
                raise NotImplementedError()

        return final_interp

    def quick_difference(self, t=None, density=50):
        """visualize the difference of form between previous and generic at time t."""
        if t is None:
            t = self._f.generic.cochain.newest
        else:
            pass

        r = np.linspace(0, 1, density)
        s = np.linspace(0, 1, density)
        r, s = np.meshgrid(r, s, indexing='ij')
        dds1 = self.region_wise_reconstruct(t, r, s, target='generic')
        dds2 = self.region_wise_reconstruct(t, r, s, target='previous')
        dds = dds1 - dds2   # 1 - 2
        dds.visualize(magnitude=True)

    def quick_visualize(self, t=None, density=50, saveto=None, **kwargs):
        """A quick visualization of generic cochain @ time t"""
        if t is None:
            t = self._f.generic.cochain.newest
        else:
            pass

        if density < 20:
            density = 20
        else:
            density = int(density)

        r = np.linspace(0, 1, density)
        s = np.linspace(0, 1, density)
        r, s = np.meshgrid(r, s, indexing='ij')
        dds1 = self.region_wise_reconstruct(t, r, s, target='generic', density=int(density/2))
        dds1.visualize(saveto=saveto, **kwargs)
