# -*- coding: utf-8 -*-
r"""

The multi-crazy ones are like stacking multiple :ref:`GALLERY-msepy-domains-and-meshes=crazy`
(blocks) together. The parameters are same to those of the crazy
mesh except there is one additional, ``Ns``, see:


.. autofunction:: msepy.manifold.predefined.crazy_multi.crazy_multi

.. note::

    Multi-crazy domains and meshes, like the crazy ones, are also mainly used for testing purposes. As each block will
    be treated as a region, with them, we can test our codes with meshes of multiple (orthogonal or curvilinear)
    regions.


Examples
--------

.. testsetup:: *

    None_or_custom_path = './source/gallery/msepy_domains_and_meshes/msepy/crazy_multi.png'
    import __init__ as ph

.. testcleanup::

    pass


We now generate a multi-crazy mesh in domain :math:`\Omega:=(x,y,z)\in[-1,1]\times[0,2]\subset\mathbb{R}^2` of
:math:`2 * 3` crazy blocks at :math:`c=0.3`. In each crazy block, we make :math:`5 * 3` elements. The codes are

>>> ph.config.set_embedding_space_dim(2)
>>> manifold = ph.manifold(2)
>>> mesh = ph.mesh(manifold)
>>> msepy, obj = ph.fem.apply('msepy', locals())
>>> manifold = obj['manifold']
>>> mesh = obj['mesh']
>>> msepy.config(manifold)('crazy_multi', c=0.3, periodic=False, Ns=[2, 3], bounds=[[-1, 1], [0, 2]])
>>> msepy.config(mesh)([5, 3])
>>> mesh.visualize(saveto=None_or_custom_path)  # doctest: +ELLIPSIS
<Figure size ...


The multi-crazy mesh is visualized as

.. figure:: crazy_multi.png
    :height: 400

    The crazy mesh in :math:`\Omega=[-1,1]\times[0,2]\subset\mathbb{R}^2` of :math:`2 * 3` blocks
    at deformation factor :math:`c=0.3`. In each block, we have :math:`5 * 3` elements.


|

↩️  Back to :ref:`GALLERY-msepy-domains-and-meshes`.
"""

import numpy as np
from tools.frozen import Frozen
from msepy.manifold.predefined.crazy import _MesPyRegionCrazyMapping


def crazy_multi(bounds=None, c=0, Ns=None, periodic=False):
    r"""

    Parameters
    ----------
    bounds : list, tuple, ``None``, default=None
        The bounds of the domain along each axis. When it is ``None``, the code will automatically analyze the
        manifold and set the ``bounds`` to be :math:`[0,1]^n` where :math:`n` is the dimensions of the space.

        For example, the unit cube is of ``bounds = ([0, 1], [0, 1], [0, 1])``.

    c : float, default=0.
        The deformation factor. ``c`` must be in :math:`[0, 0.3]`. When ``c = 0``, the domain is orthogonal, and
        when ``c > 0``, the space in the domain is distorted.

    Ns : list, None, default=None

        ``Ns`` should be a list of :math:`n` (the dimensions of the manifold) positive integers. It means along
        each axis, we will stack how many crazy meshes.

        When it is ``None``, the code will automatically analyze the manifold and then set ``Ns`` to be
        :math:`[\underbrace{2, 2, \cdots}_{n}]`.

    periodic : bool, default=False
        It indicates whether the domain is periodic. When it is ``True``, the domain is fully periodic along all
        axes. And when it is ``False``, the domain is not periodic at all.

    """
    raise Exception(bounds, c, Ns, periodic)


def _crazy_multi(mf, bounds=None, c=0, Ns=None, periodic=False):
    """"""
    assert mf.m == mf.ndim, f"crazy mesh only works for manifold.ndim == embedding space dimensions."
    m = mf.m
    if bounds is None:
        bounds = [(0, 1) for _ in range(m)]
    else:
        assert len(bounds) == m, f"bounds={bounds} dimensions wrong."

    region_mappings = _MesPyRegionCrazyMultiMapping(bounds, c, m, Ns=Ns)
    Ns = region_mappings._Ns
    num_regions = np.prod(Ns)
    region_map_nd = np.arange(num_regions).reshape(Ns, order='F')

    region_map = dict()
    if m == 1:
        if periodic:
            N = Ns[0]

            for n in range(N):

                left, right = n-1, n+1

                if left < 0:
                    left = region_map_nd[-1]
                else:
                    left = region_map_nd[left]

                if right >= N:
                    right = region_map_nd[0]
                else:
                    right = region_map_nd[right]

                region_map[n] = [left, right]

        else:
            N = Ns[0]

            for n in range(N):

                left, right = n-1, n+1

                if left < 0:
                    left = None
                else:
                    left = region_map_nd[left]

                if right >= N:
                    right = None
                else:
                    right = region_map_nd[right]

                region_map[n] = [left, right]
    elif m == 2:
        if periodic:
            Nx, Ny = Ns
            for ny in range(Ny):
                for nx in range(Nx):
                    n = nx + ny * Nx

                    x_m, x_p = nx-1, nx+1
                    y_m, y_p = ny-1, ny+1

                    if x_m < 0:
                        x_m = region_map_nd[-1, ny]
                    else:
                        x_m = region_map_nd[x_m, ny]

                    if x_p >= Nx:
                        x_p = region_map_nd[0, ny]
                    else:
                        x_p = region_map_nd[x_p, ny]

                    if y_m < 0:
                        y_m = region_map_nd[nx, -1]
                    else:
                        y_m = region_map_nd[nx, y_m]

                    if y_p >= Ny:
                        y_p = region_map_nd[nx, 0]
                    else:
                        y_p = region_map_nd[nx, y_p]

                    region_map[n] = [x_m, x_p, y_m, y_p]
        else:
            Nx, Ny = Ns
            for ny in range(Ny):
                for nx in range(Nx):
                    n = nx + ny * Nx

                    x_m, x_p = nx-1, nx+1
                    y_m, y_p = ny-1, ny+1

                    if x_m < 0:
                        x_m = None
                    else:
                        x_m = region_map_nd[x_m, ny]

                    if x_p >= Nx:
                        x_p = None
                    else:
                        x_p = region_map_nd[x_p, ny]

                    if y_m < 0:
                        y_m = None
                    else:
                        y_m = region_map_nd[nx, y_m]

                    if y_p >= Ny:
                        y_p = None
                    else:
                        y_p = region_map_nd[nx, y_p]

                    region_map[n] = [x_m, x_p, y_m, y_p]
    elif m == 3:
        if periodic:
            Nx, Ny, Nz = Ns
            for nz in range(Nz):
                for ny in range(Ny):
                    for nx in range(Nx):
                        n = nx + ny * Nx + nz * Nx * Ny

                        x_m, x_p = nx-1, nx+1
                        y_m, y_p = ny-1, ny+1
                        z_m, z_p = nz-1, nz+1

                        if x_m < 0:
                            x_m = region_map_nd[-1, ny, nz]
                        else:
                            x_m = region_map_nd[x_m, ny, nz]
                        if x_p >= Nx:
                            x_p = region_map_nd[0, ny, nz]
                        else:
                            x_p = region_map_nd[x_p, ny, nz]

                        if y_m < 0:
                            y_m = region_map_nd[nx, -1, nz]
                        else:
                            y_m = region_map_nd[nx, y_m, nz]
                        if y_p >= Ny:
                            y_p = region_map_nd[nx, 0, nz]
                        else:
                            y_p = region_map_nd[nx, y_p, nz]

                        if z_m < 0:
                            z_m = region_map_nd[nx, ny, -1]
                        else:
                            z_m = region_map_nd[nx, ny, z_m]
                        if z_p >= Nz:
                            z_p = region_map_nd[nx, ny, 0]
                        else:
                            z_p = region_map_nd[nx, ny, z_p]

                        region_map[n] = [x_m, x_p, y_m, y_p, z_m, z_p]
        else:
            Nx, Ny, Nz = Ns
            for nz in range(Nz):
                for ny in range(Ny):
                    for nx in range(Nx):
                        n = nx + ny * Nx + nz * Nx * Ny

                        x_m, x_p = nx-1, nx+1
                        y_m, y_p = ny-1, ny+1
                        z_m, z_p = nz-1, nz+1

                        if x_m < 0:
                            x_m = None
                        else:
                            x_m = region_map_nd[x_m, ny, nz]
                        if x_p >= Nx:
                            x_p = None
                        else:
                            x_p = region_map_nd[x_p, ny, nz]

                        if y_m < 0:
                            y_m = None
                        else:
                            y_m = region_map_nd[nx, y_m, nz]
                        if y_p >= Ny:
                            y_p = None
                        else:
                            y_p = region_map_nd[nx, y_p, nz]

                        if z_m < 0:
                            z_m = None
                        else:
                            z_m = region_map_nd[nx, ny, z_m]
                        if z_p >= Nz:
                            z_p = None
                        else:
                            z_p = region_map_nd[nx, ny, z_p]

                        region_map[n] = [x_m, x_p, y_m, y_p, z_m, z_p]
    else:
        raise NotImplementedError()

    mapping_dict = dict()
    Jacobian_matrix_dict = dict()
    mtype_dict = dict()

    for r in range(num_regions):
        mapping_dict[r] = region_mappings._regions[r].mapping
        Jacobian_matrix_dict[r] = region_mappings._regions[r].Jacobian_matrix
        if c == 0:
            mtype = {'indicator': 'Linear', 'parameters': []}
            for i, lb_ub in enumerate(region_mappings._regions[r]._bounds):
                xyz = 'xyz'[i]
                lb, ub = lb_ub
                d = str(round(ub - lb, 5))  # do this to round off the truncation error.
                mtype['parameters'].append(xyz + d)
        else:
            mtype = None  # this is a unique region. Its metric does not like any other.

        mtype_dict[r] = mtype

    return region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict, None


class _MesPyRegionCrazyMultiMapping(Frozen):
    """"""

    def __init__(self, bounds, c, m, Ns=None):
        """

        Parameters
        ----------
        bounds
        c
        m
        Ns :
            The domain is divided into np.prod(Ns) regions equally.
        """
        for i, bs in enumerate(bounds):
            assert len(bs) == 2 and all([isinstance(_, (int, float)) for _ in bs]), f"bounds[{i}]={bs} is illegal."
            lb, up = bs
            assert lb < up, f"bounds[{i}]={bs} is illegal. low bound is larger than (or equal to) higher bound."
        assert isinstance(c, (int, float)), f"={c} is illegal, need to be a int or float. Ideally in [0, 0.3]."

        if Ns is None:
            Ns = 2
        else:
            pass

        if isinstance(Ns, int):
            Ns = [Ns for _ in range(m)]
        else:
            assert len(Ns) == m and all([isinstance(ns, int) for ns in Ns]) and all([ns > 0 for ns in Ns]), \
                f"Ns={Ns} is wrong."

        num_regions = np.prod(Ns)

        axis_indices = [np.arange(Ns[_]) for _ in range(m)]
        axis_indices = np.meshgrid(*axis_indices, indexing='ij')
        axis_indices = [ai.ravel('F') for ai in axis_indices]

        regions = dict()

        self._indices = dict()
        for i in range(num_regions):
            indices = [ai[i] for ai in axis_indices]
            self._indices[i] = indices
            region_bounds = list()
            for j in range(m):
                bds = bounds[j]
                lb, up = bds
                N = Ns[j]
                delta = (up - lb) / N
                region_bounds.append([lb + delta * indices[j], lb + delta * (indices[j]+1)])
            regions[i] = _MesPyRegionCrazyMapping(region_bounds, c, m)
        self._regions = regions
        self._Ns = Ns
        self._freeze()
