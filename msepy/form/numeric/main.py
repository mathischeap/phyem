# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.dds.region_wise_structured import DDSRegionWiseStructured
from typing import Dict
from tools.quadrature import Quadrature
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class MsePyRootFormNumeric(Frozen):
    """Numeric methods are approximate; less accurate than, for example, reconstruction."""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def rot(self, *grid_xi_et_sg, t=None):
        """Compute the rot of the form at `t` and save the results in a region-wise structured data set."""
        if t is None:
            time = self._f.cochain.newest
        else:
            time = t
        indicator = self._f.space.abstract.indicator
        if indicator == 'bundle':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k == 0:
                df = self._f.coboundary[time]

                xy, tensor = df.reconstruct(*grid_xi_et_sg)

                x, y = xy

                pv_px = tensor[1][0]
                pu_py = tensor[0][1]

                rot_data = pv_px - pu_py

                x, y, rot_data = self._f.mesh._regionwsie_stack(x, y, rot_data)

                return DDSRegionWiseStructured([x, y], [rot_data, ])

            else:
                raise Exception(f"form of space {space} cannot perform curl.")
        else:
            raise NotImplementedError()

    def divergence(self, *grid_xi_et_sg, t=None, magnitude=False):
        """Compute the divergence of the form at `t` and save the results in a region-wise
        structured data set.
        """
        if t is None:
            time = self._f.cochain.newest
        else:
            time = t
        indicator = self._f.space.abstract.indicator
        if indicator == 'bundle':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k == 0:
                df = self._f.coboundary[time]

                xy, tensor = df.reconstruct(*grid_xi_et_sg)

                x, y = xy

                pu_px = tensor[0][0]
                pv_py = tensor[1][1]

                div_data = pu_px + pv_py
                if magnitude:
                    div_data = np.abs(div_data)
                    div_data[div_data < 1e-16] = 1e-16
                    div_data = np.log10(div_data)
                else:
                    pass

                x, y, div_data = self._f.mesh._regionwsie_stack(x, y, div_data)

                return DDSRegionWiseStructured([x, y], [div_data, ])

            else:
                raise Exception(f"form of space {space} cannot perform curl.")
        else:
            raise NotImplementedError()

    def _make_interp(self, t=None, density=30):
        """"""
        if t is None:
            time = self._f.cochain.newest
        else:
            time = t
        indicator = self._f.space.abstract.indicator
        if indicator == 'Lambda':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k in (0, 2):
                shape = [1]     # scalar in 2d
                ndim = 2        # scalar in 2d

            elif m == n == 2 and k == 1:
                shape = [2]     # vector in 2d
                ndim = 2        # vector in 2d

            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        nodes = Quadrature(density, category='Gauss').quad_nodes
        xyz, v = self._f[time].reconstruct(nodes, nodes, ravel=True)

        regions = self._f.mesh.manifold.regions

        if ndim == 2:            # 2d
            X = dict()
            Y = dict()
            for region in regions:
                X[region] = list()
                Y[region] = list()
            x, y = xyz

            for region in regions:
                elements = self._f.mesh.elements._elements_in_region(region)
                for ele in range(*elements):
                    X[region].extend(x[ele])
                    Y[region].extend(y[ele])

            if shape == [1]:       # scalar in 2d
                v = v[0]
                V = dict()
                interp: Dict = dict()
                for region in regions:
                    V[region] = []
                    interp[region] = None

                for region in regions:
                    elements = self._f.mesh.elements._elements_in_region(region)
                    for ele in range(*elements):
                        V[region].extend(v[ele])

                    region_xy = list(zip(X[region], Y[region]))
                    interp[region] = NearestNDInterpolator(
                        region_xy, V[region]
                    )
                return interp

            elif shape == [2]:       # vector in 2d
                u, v = v
                U = dict()
                V = dict()
                interp_u: Dict = dict()
                interp_v: Dict = dict()
                for region in regions:
                    U[region] = []
                    V[region] = []
                    interp_u[region] = None
                    interp_v[region] = None

                for region in regions:
                    elements = self._f.mesh.elements._elements_in_region(region)
                    for ele in range(*elements):
                        U[region].extend(u[ele])
                        V[region].extend(v[ele])

                    region_xy = list(zip(X[region], Y[region]))
                    interp_u[region] = NearestNDInterpolator(
                        region_xy, U[region]
                    )
                    interp_v[region] = NearestNDInterpolator(
                        region_xy, V[region]
                    )

                return interp_u, interp_v

            else:
                raise NotImplementedError()
        else:   # other dimensions
            raise NotImplementedError()

    def region_wise_interp(self, t=None, density=30, saveto=None):
        """Reconstruct the form at time `t` and use the reconstruction results to
        make interpolation functions in each region.

        These functions take (x, y, ...) (physical domain coordinates) as inputs.

        Parameters
        ----------
        t
        density
        saveto

        Returns
        -------
        final_interp :  dict
            A dictionary of interpolations. Keys are regions, values are the
            interpolations in the regions.

        """
        indicator = self._f.space.abstract.indicator
        if indicator == 'Lambda':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k in (0, 2):
                shape = [1]   # scalar in 2d
                ndim = 2      # scalar in 2d

            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        regions = self._f.mesh.manifold.regions
        interp = self._make_interp(t=t, density=density)

        if ndim == 2:         # 2d
            r = s = np.linspace(0, 1, 3 * density)
            r, s = np.meshgrid(r, s, indexing='ij')
            r = r.ravel('F')
            s = s.ravel('F')
            final_interp: Dict = dict()

            if shape == [1]:   # scalar in 2d
                for region in interp:
                    x, y = regions[region]._ct.mapping(r, s)
                    xy = np.vstack([x, y]).T
                    v = interp[region](x, y)
                    final_itp = LinearNDInterpolator(xy, v)
                    final_interp[region] = final_itp

            else:
                raise NotImplementedError()
        else:  # other dimensions
            raise NotImplementedError()

        if saveto is None:
            pass
        else:
            import pickle
            from src.config import SIZE
            if SIZE == 1:
                # we are only calling one thread, so just go ahead with it.
                with open(saveto, 'wb') as output:
                    pickle.dump(final_interp, output, pickle.HIGHEST_PROTOCOL)
                output.close()

            else:
                raise NotImplementedError()

        return final_interp

    def interp(self, t=None, density=30):
        """Reconstruct the form at time `t` and use the reconstruction results to
        make interpolation functions all over the domain

        These functions take (x, y, ...) (physical domain coordinates) as inputs.

        Parameters
        ----------
        t
        density

        Returns
        -------
        final_interp :
            A interpolation for the whole domain.

        """
        indicator = self._f.space.abstract.indicator
        if indicator == 'Lambda':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k in (0, 2):
                shape = [1]   # scalar in 2d
                ndim = 2      # scalar in 2d

            elif m == n == 2 and k == 1:
                shape = [2]   # vector in 2d
                ndim = 2      # vector in 2d

            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        regions = self._f.mesh.manifold.regions
        interp = self._make_interp(t=t, density=density)

        if ndim == 2:         # 2d
            r = s = np.linspace(0, 1, 3 * density)
            r, s = np.meshgrid(r, s, indexing='ij')
            r = r.ravel('F')
            s = s.ravel('F')

            X = list()
            Y = list()

            if shape == [1]:   # scalar in 2d
                V0 = list()
                for region in interp:
                    x, y = regions[region]._ct.mapping(r, s)
                    v = interp[region](x, y)
                    X.extend(x)
                    Y.extend(y)
                    V0.extend(v)
                final_itp = LinearNDInterpolator(list(zip(X, Y)), V0)
                return final_itp

            elif shape == [2]:  # vector in 2d
                U, V = list(), list()
                interp_u, interp_v = interp
                for region in regions:
                    x, y = regions[region]._ct.mapping(r, s)
                    u = interp_u[region](x, y)
                    v = interp_v[region](x, y)
                    X.extend(x)
                    Y.extend(y)
                    U.extend(u)
                    V.extend(v)
                xy = list(zip(X, Y))
                final_itp_u = LinearNDInterpolator(xy, U)
                final_itp_v = LinearNDInterpolator(xy, V)
                return final_itp_u, final_itp_v

            else:
                raise NotImplementedError()
        
        else:  # other dimensions
            raise NotImplementedError()
