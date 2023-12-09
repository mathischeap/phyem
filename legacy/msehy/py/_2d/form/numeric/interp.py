# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from typing import Dict
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class MseHyPy2FormNumericInterp(Frozen):
    """"""

    def __init__(self, f, t, target):
        """"""
        self._f = f
        self._t = t
        assert target in ('generic', 'previous'), f"target = {target} is wrong."
        self._target = target
        self._freeze()

    def __call__(self, method='linear', density=50):
        """"""
        nodes = Quadrature(density, category='Gauss').quad_nodes
        if self._target == 'generic':
            xy, v = self._f.generic[self._t].reconstruct(nodes, nodes, ravel=True)
        elif self._target == 'previous':
            xy, v = self._f.previous[self._t].reconstruct(nodes, nodes, ravel=True)
        else:
            raise Exception()

        x, y = xy

        if len(v) == 1 and isinstance(v[0], dict):                    # v represents a scalar
            v_list = v
        elif len(v) == 2 and all([isinstance(_, dict) for _ in v]):   # v represents a vector
            v_list = v
        # - if tensor, for example in 2d, plut value in a list of four entries:
        # v_list = [v00, v01, v10, v11]
        # Do not use a 2d list.
        else:
            raise NotImplementedError()

        if self._target == 'generic':
            representative = self._f.mesh.representative
        elif self._target == 'previous':
            representative = self._f.mesh.previous
        else:
            raise Exception()

        regions = representative.background.manifold.regions

        X = dict()
        Y = dict()
        V = dict()
        interp: Dict = dict()
        final_interp: Dict = dict()
        for region in regions:
            X[region] = list()
            Y[region] = list()
            V_ = list()
            itp_list = list()
            f_itp_list = list()
            for _ in v_list:
                V_.append([])
                itp_list.append(None)
                f_itp_list.append(None)
            V[region] = tuple(V_)
            interp[region] = itp_list
            final_interp[region] = f_itp_list

        for index in x:
            if isinstance(index, str):
                num_level = index.count('-')
                fc = representative.levels[num_level].triangles[index]
            else:
                fc = representative.background.elements[index]
            region = fc.region
            X[region].extend(x[index])
            Y[region].extend(y[index])
            for j, v in enumerate(v_list):
                V[region][j].extend(v[index])

        for region in X:
            for j, v in enumerate(v_list):
                interp[region][j] = NearestNDInterpolator(
                    list(zip(X[region], Y[region])), V[region][j]
                )

        r = s = np.linspace(0, 1, density)
        r, s = np.meshgrid(r, s, indexing='ij')
        r = r.ravel('F')
        s = s.ravel('F')

        for region in interp:
            x, y = regions[region]._ct.mapping(r, s)
            xy = np.vstack([x, y]).T

            local_itp_s = interp[region]
            for j, itp in enumerate(local_itp_s):
                v = itp(x, y)
                final_itp = LinearNDInterpolator(xy, v)
                final_interp[region][j] = final_itp

        return final_interp
