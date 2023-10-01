# -*- coding: utf-8 -*-
r"""
In :math:`\mathbb{R}^n` :math:`(n\in\left\lbrace1,2,3\right\rbrace)`, the crazy domain is
(for ``msepy`` implementation only) defined in
:math:`\underbrace{[a, b]\times[e, f]\times\cdots}_{n}`,
and :math:`[a, b]`, :math:`[e, f]`, :math:`\cdots` :math:`(a, b, e, f, \cdots \in \mathbb{R})` are the bounds.
The parameters of a crazy domain are

.. autofunction:: msepy.manifold.predefined.crazy.crazy

.. note::

    You may ask that in such a regularly shaped domain why would one use a deformation factor :math:`c>0` to make
    life more difficulty. You are absolutely right. When you surely know what you are doing, you probably will only
    use :math:`c=0`. The meshes of :math:`c>0` are normally for testing, for example, the convergence rate of your
    scheme in bad grids.

The mapping transforming the domain
-----------------------------------

We use :math:`\mathbb{R}^3` as an example. Assume the crazy domain is
:math:`\Omega:=(x,y,z)\in[a,b]\times[e,f]\times[g,h]`,
i.e., ``bounds = ([a, b], [e, f], [g, h])``. And let :math:`\mathring{\Omega}:=(r, s, t)\in [0,1]^3` be an orthogonal
domain. The mapping :math:`\Phi` that transforms :math:`\mathring{\Omega}` into :math:`\Omega` is

.. math::
    \begin{pmatrix}
        x\\y\\z
    \end{pmatrix} = \Phi(r,s,t) =
    \begin{pmatrix}
    (b-a)\left(r + \frac{1}{2}c \sin(2\pi r)\sin(2\pi s)\sin(2\pi t)\right) + a\\
    (f-e)\left(s + \frac{1}{2}c \sin(2\pi r)\sin(2\pi s)\sin(2\pi t)\right) + e\\
    (g-h)\left(t + \frac{1}{2}c \sin(2\pi r)\sin(2\pi s)\sin(2\pi t)\right) + h
    \end{pmatrix},


Examples
--------

.. testsetup:: *

    None_or_custom_path_3 = './source/gallery/msepy_domains_and_meshes/msepy/crazy_3d_c.png'
    None_or_custom_path_2 = './source/gallery/msepy_domains_and_meshes/msepy/crazy_2d_c.png'
    import __init__ as ph

.. testcleanup::

    pass

Below codes generate a crazy domain in :math:`\Omega:=(x,y,z)\in[-1,1]\times[0,2]\times[0,2]\subset\mathbb{R}^3` of
:math:`c=0.15`. A mesh
of :math:`5 * 5 * 5` elements are then generated in the domain ans is shown the following figure.

>>> ph.config.set_embedding_space_dim(3)
>>> manifold = ph.manifold(3)
>>> mesh = ph.mesh(manifold)
>>> msepy, obj = ph.fem.apply('msepy', locals())
>>> manifold = obj['manifold']
>>> mesh = obj['mesh']
>>> msepy.config(manifold)('crazy', c=0.15, periodic=False, bounds=[[-1, 1], [0, 2], [0, 2]])
>>> msepy.config(mesh)([5, 5, 5])
>>> mesh.visualize(saveto=None_or_custom_path_3)  # doctest: +ELLIPSIS
<Figure size ...

.. figure:: crazy_3d_c.png
    :height: 400

    The crazy mesh in :math:`\Omega=[-1,1]\times[0,2]\times[0,2]` of :math:`5 * 5 * 5` elements
    at deformation factor :math:`c=0.15`.

And, if we want to generate a crazy mesh in domain :math:`\Omega:=(x,y,z)\in[-1,1]\times[0,2]\subset\mathbb{R}^2` of
:math:`7 * 7` elements at :math:`c=0.3`, we can do

>>> ph.config.set_embedding_space_dim(2)
>>> manifold = ph.manifold(2)
>>> mesh = ph.mesh(manifold)
>>> msepy, obj = ph.fem.apply('msepy', locals())
>>> manifold = obj['manifold']
>>> mesh = obj['mesh']
>>> msepy.config(manifold)('crazy', c=0.3, periodic=False, bounds=[[-1, 1], [0, 2]])
>>> msepy.config(mesh)([7, 7])
>>> mesh.visualize(saveto=None_or_custom_path_2)  # doctest: +ELLIPSIS
<Figure size ...

.. figure:: crazy_2d_c.png
    :height: 400

    The crazy mesh in :math:`\Omega=[-1,1]\times[0,2]\subset\mathbb{R}^2` of :math:`7 * 7` elements
    at deformation factor :math:`c=0.3`.


|

↩️  Back to :ref:`GALLERY-msepy-domains-and-meshes`.

"""

from numpy import sin, pi, cos, ones_like

import warnings
from tools.frozen import Frozen


class CrazyMeshCurvatureWarning(UserWarning):
    pass


def crazy(bounds=None, c=0, periodic=False):
    """
    Parameters
    ----------
    bounds : list, tuple, None, default=None
        The bounds of the domain along each axis. When it is ``None``, the code will automatically analyze the
        manifold and set the ``bounds`` to be :math:`[0,1]^n` where :math:`n` is the dimensions of the space.

        For example, the unit cube is of ``bounds = ([0, 1], [0, 1], [0, 1])``.

    c : float, default=0.
        The deformation factor. ``c`` must be in :math:`[0, 0.3]`. When ``c = 0``, the domain is orthogonal, and
        when ``c > 0``, the space in the domain is distorted.

    periodic : bool, default=False
        It indicates whether the domain is periodic. When it is ``True``, the domain is fully periodic along all
        axes. And when it is ``False``, the domain is not periodic at all.
    """
    raise Exception(bounds, c, periodic)


def _crazy(mf, bounds=None, c=0, periodic=False):
    """"""
    assert mf.esd == mf.ndim, f"crazy mesh only works for manifold.ndim == embedding space dimensions."
    esd = mf.esd
    if bounds is None:
        bounds = [(0, 1) for _ in range(esd)]
    else:
        assert len(bounds) == esd, f"bounds={bounds} dimensions wrong."

    rm0 = _MesPyRegionCrazyMapping(bounds, c, esd)

    if periodic:
        region_map = {
            0: [0 for _ in range(2 * esd)],    # region #0
        }
    else:
        region_map = {
            0: [None for _ in range(2 * esd)],    # region #0
        }

    mapping_dict = {
        0: rm0.mapping,  # region #0
    }

    Jacobian_matrix_dict = {
        0: rm0.Jacobian_matrix
    }

    if c == 0:
        mtype = {'indicator': 'Linear', 'parameters': []}
        for i, lb_ub in enumerate(bounds):
            xyz = 'xyz'[i]
            lb, ub = lb_ub
            d = str(round(ub - lb, 5))  # do this to round off the truncation error.
            mtype['parameters'].append(xyz + d)
    else:
        mtype = None  # this is a unique region. Its metric does not like any other.

    mtype_dict = {
        0: mtype
    }

    return region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict, None


class _MesPyRegionCrazyMapping(Frozen):

    def __init__(self, bounds, c, esd):
        for i, bs in enumerate(bounds):
            assert len(bs) == 2 and all([isinstance(_, (int, float)) for _ in bs]), f"bounds[{i}]={bs} is illegal."
            lb, up = bs
            assert lb < up, f"bounds[{i}]={bs} is illegal."
        assert isinstance(c, (int, float)), f"={c} is illegal, need to be a int or float. Ideally in [0, 0.3]."

        if not (0 <= c <= 0.3):
            warnings.warn(f"c={c} is not good. Ideally, c in [0, 0.3].", CrazyMeshCurvatureWarning)

        self._bounds = bounds
        self._c = c
        self._esd = esd
        self._freeze()

    def mapping(self, *rst):
        """ `*rst` be in [0, 1]. """
        assert len(rst) == self._esd, f"amount of inputs wrong."

        if self._esd == 1:

            r = rst[0]
            a, b = self._bounds[0]
            x = (b - a) * (r + 0.5 * self._c * sin(2 * pi * r)) + a
            return [x]

        elif self._esd == 2:

            r, s = rst
            a, b = self._bounds[0]
            c, d = self._bounds[1]
            if self._c == 0:
                x = (b - a) * r + a
                y = (d - c) * s + c
            else:
                x = (b - a) * (r + 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s)) + a
                y = (d - c) * (s + 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s)) + c
            return x, y

        elif self._esd == 3:

            r, s, t = rst
            a, b = self._bounds[0]
            c, d = self._bounds[1]
            e, f = self._bounds[2]

            if self._c == 0:
                x = (b - a) * r + a
                y = (d - c) * s + c
                z = (f - e) * t + e

            else:
                x = (b - a) * (r + 0.5 * self._c *
                               sin(2 * pi * r) *
                               sin(2 * pi * s) *
                               sin(2 * pi * t)) + a
                y = (d - c) * (s + 0.5 * self._c *
                               sin(2 * pi * r) *
                               sin(2 * pi * s) *
                               sin(2 * pi * t)) + c
                z = (f - e) * (t + 0.5 * self._c *
                               sin(2 * pi * r) *
                               sin(2 * pi * s) *
                               sin(2 * pi * t)) + e

            return x, y, z

        else:
            raise NotImplementedError()

    def Jacobian_matrix(self, *rst):
        """ r, s, t be in [0, 1]. """
        assert len(rst) == self._esd, f"amount of inputs wrong."

        if self._esd == 1:
            r = rst[0]
            a, b = self._bounds[0]
            if self._c == 0:
                xr = (b - a) * ones_like(r)
            else:
                xr = (b - a) + (b - a) * 2 * pi * 0.5 * self._c * cos(2 * pi * r)
            return [[xr]]

        elif self._esd == 2:
            r, s = rst
            
            a, b = self._bounds[0]
            c, d = self._bounds[1]
            if self._c == 0:
                xr = (b - a) * ones_like(r)
                xs = 0
                yr = 0
                ys = (d - c) * ones_like(r)

            else:
                xr = (b - a) + (b - a) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s)
                xs = (b - a) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s)
                yr = (d - c) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s)
                ys = (d - c) + (d - c) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s)
            return ((xr, xs),
                    (yr, ys))

        elif self._esd == 3:

            r, s, t = rst
            a, b = self._bounds[0]
            c, d = self._bounds[1]
            e, f = self._bounds[2]

            if self._c == 0:
                xr = (b - a) * ones_like(r)
                xs = 0  # np.zeros_like(r)
                xt = 0
    
                yr = 0
                ys = (d - c) * ones_like(r)
                yt = 0
    
                zr = 0
                zs = 0
                zt = (f - e) * ones_like(r)

            else:
                xr = (b - a) + (b - a) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s) * sin(
                    2 * pi * t)
                xs = (b - a) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s) * sin(
                    2 * pi * t)
                xt = (b - a) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s) * cos(
                    2 * pi * t)
    
                yr = (d - c) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s) * sin(
                    2 * pi * t)
                ys = (d - c) + (d - c) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s) * sin(
                    2 * pi * t)
                yt = (d - c) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s) * cos(
                    2 * pi * t)
    
                zr = (f - e) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s) * sin(
                    2 * pi * t)
                zs = (f - e) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s) * sin(
                    2 * pi * t)
                zt = (f - e) + (f - e) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s) * cos(
                    2 * pi * t)
    
            return [(xr, xs, xt),
                    (yr, ys, yt),
                    (zr, zs, zt)]
