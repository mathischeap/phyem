# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences,PyRedeclaration
r"""
.. testsetup:: *

    None_or_custom_path_2 = './source/gallery/msepy_domains_and_meshes/msepy/backward_step2.png'
    None_or_custom_path_3 = './source/gallery/msepy_domains_and_meshes/msepy/backward_step3.png'

    import phyem as ph
    from phyem.msepy.manifold.predefined.backward_step import _make_an_illustration
    _make_an_illustration(
        './source/gallery/msepy_domains_and_meshes/msepy/backward_step_illustration.png'
    )

.. testcleanup::

    pass


The backward step is a mesh (or domain) in :math:`\mathbb{R}^n`, :math:`n\in\left\lbrace2,3\right\rbrace`. The domain is
illustrated in the following figure.

.. figure:: backward_step_illustration.png
    :width: 100%

    The illustration of the backward step domain.

When :math:`n=3`, The domain is extended to the thrid axis, :math:`z`-axis, perpendicular to the plane.
The parameters are

.. autofunction:: phyem.msepy.manifold.predefined.backward_step.backward_step


Boundary units
==============

The domain is divided into three regions,

+---------------+--------------------------------------------------------------------------+
| region ``0``  |  bottom-right region, :math:`[x_1, (x_1+x_2)]\times[0, y_1]`             |
+---------------+--------------------------------------------------------------------------+
| region ``1``  |  top-right region , :math:`[x_1, (x_1+x_2)]\times[y_1, (y_1+y_2)]`       |
+---------------+--------------------------------------------------------------------------+
| region ``2``  |  top-left region, :math:`[0, x_1]\times[y_1, (y_1+y_2)]`                 |
+---------------+--------------------------------------------------------------------------+

Thus, for a 2-dimensional domain, it has 8 boundary units, i.e.,

>>> boundary_units_set = {
...     0: [1, 1, 1, 0],
...     1: [0, 1, 0, 1],
...     2: [1, 0, 1, 1],
... }

For example, for the bottom-right region ``0``, its left (:math:`x^-`), right (:math:`x^+`) and bottom
(:math:`y^-`) faces are boundary units, while its north (:math:`y^+`) face is not. So we have
``0: [1, 1, 1, 0]`` in the set.

And a 3-dimensional domain, it has 8 + 6
(:math:`2\times3` :math:`z`-direction) boundary units, i.e.,

>>> boundary_units_set = {
...     0: [1, 1, 1, 0, 1, 1],
...     1: [0, 1, 0, 1, 1, 1],
...     2: [1, 0, 1, 1, 1, 1],
... }

Examples
========

3d
--

Below codes will lead to a three-dimensional backward step mesh of :math:`3*5*5*5` elements (because the domain
is splited into 3 regions).

>>> ph.config.set_embedding_space_dim(3)
>>> manifold = ph.manifold(3)
>>> mesh = ph.mesh(manifold)
>>> msepy, obj = ph.fem.apply('msepy', locals())
>>> manifold = obj['manifold']
>>> mesh = obj['mesh']
>>> msepy.config(manifold)('backward_step')
>>> msepy.config(mesh)([5, 5, 5])
>>> mesh.visualize(saveto=None_or_custom_path_3)  # doctest: +ELLIPSIS
<Figure size ...

.. figure:: backward_step3.png
    :width: 60%

    The three-dimensional backward step mesh of :math:`3*5*5*5` elements.

2d
--

To make a two-dimensional backward step mesh of :math:`3*24*6` elements, just do

>>> ph.config.set_embedding_space_dim(2)
>>> manifold = ph.manifold(2)
>>> mesh = ph.mesh(manifold)
>>> msepy, obj = ph.fem.apply('msepy', locals())
>>> manifold = obj['manifold']
>>> mesh = obj['mesh']
>>> msepy.config(manifold)('backward_step')
>>> msepy.config(mesh)([24, 6])
>>> mesh.visualize(saveto=None_or_custom_path_2)  # doctest: +ELLIPSIS
<Figure size ...

.. figure:: backward_step2.png
    :width: 100%

    The two-dimensional backward step mesh of :math:`3*24*6` elements.


|

↩️  Back to :ref:`GALLERY-msepy-domains-and-meshes`.
"""
import matplotlib.pyplot as plt
from phyem.msepy.manifold.predefined._helpers import _LinearTransformation


def _make_an_illustration(saveto, x1=1, x2=1, y1=0.25, y2=0.25):
    """"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(
        axis='both',
        which='both',
        left=False,
        bottom=False,
        labelbottom=False,
        labelleft=False
    )

    x = [x1, x1+x2, x1+x2, 0,     0, x1, x1]
    y = [0,  0,     y1+y2, y1+y2, y1, y1, 0]

    plt.plot(x, y, '-k', linewidth=0.8)
    plt.text(x1 + x2/2, 0.03, r'$x2$', ha='center')
    plt.text(x1/2, y1 + 0.03, r'$x1$', ha='center')
    plt.text(- 0.07, y1 + y2/2, r'$y2$', va='center')
    plt.text(x1 - 0.07, y1/2, r'$y1$', va='center')
    plt.text(x1, y1, r'$(x1, y1)$', va='bottom', ha='left')

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig(saveto, bbox_inches='tight', dpi=200)
    plt.close()


def backward_step(x1=1, x2=1, y1=0.25, y2=0.25, z=None, periodic=False):
    """

    Parameters
    ----------
    x1 : float, default=1
        See the illustration.
    x2 : float, default=1
        See the illustration.
    y1 : float, default=0.25
        See the illustration.
    y2 : float, default=0.25
        See the illustration.
    z : float, None, default=None
        When it is ``None``, it gives a two-dimensional domain. Otherwise, :math:`(z>0)`, it gives a three-dimensional
        one.
    periodic : bool, default=False
        When the domain is 3d, whether it is periodic along the ``z``-axis?

    """
    raise Exception(x1, x2, y1, y2, z, periodic)


def _backward_step(mf, x1=1, x2=1, y1=0.25, y2=0.25, z=None, x0=0, y0=0, periodic=False):
    r"""
    ^ y
    |
    |              x1                      x2
     __________________________________________________
    ||                    |                           |
    ||                    |                           |
    ||         r2         |           r1              |    y2
    ||                    |                           |
    ||____________________|___________________________|
    |            (x1, y1) |                           |
    |                     |                           |
    |                     |           r0              |    y1
    |                     |                           |
    | (x0,y0)             |___________________________|
    .--------------------------------------------------------------> x
    z

    Parameters
    ----------
    mf
    x1
    x2
    y1
    y2
    z

    Returns
    -------

    """
    assert mf.esd == mf.ndim, f"backward_step mesh only works for manifold.ndim == embedding space dimensions."
    assert mf.esd in (2, 3), f"backward_step mesh only works in 2-, 3-dimensions."
    esd = mf.esd
    if z is None:
        if esd == 2:
            z = 0
        else:
            z = 0.25
    else:
        pass
    if esd == 2:
        assert z == 0, f"for 2-d backward_step mesh, z must be 0."
    elif esd == 3:
        assert z > 0, f"for 3-d backward_step mesh, z must be greater than 0."
    else:
        raise NotImplementedError()

    if esd == 2:
        rm0 = _LinearTransformation(x0+x1, x0+x1+x2, y0+0,  y0+y1)
        rm1 = _LinearTransformation(x0+x1, x0+x1+x2, y0+y1, y0+y1+y2)
        rm2 = _LinearTransformation(x0+0,  x0+x1,    y0+y1, y0+y1+y2)
    elif esd == 3:
        rm0 = _LinearTransformation(x0+x1, x0+x1+x2, y0+0,  y0+y1,    0, z)
        rm1 = _LinearTransformation(x0+x1, x0+x1+x2, y0+y1, y0+y1+y2, 0, z)
        rm2 = _LinearTransformation(x0+0,  x0+x1,    y0+y1, y0+y1+y2, 0, z)
    else:
        raise Exception()

    if esd == 2:
        region_map = {
            0: [None, None, None, 1],
            1: [2,    None, 0,    None],
            2: [None, 1,    None, None],
        }
    elif esd == 3:
        if periodic:
            region_map = {
                0: [None, None, None, 1,    0, 0],
                1: [2,    None, 0,    None, 1, 1],
                2: [None, 1,    None, None, 2, 2],
            }
        else:
            region_map = {
                0: [None, None, None, 1,    None, None],
                1: [2,    None, 0,    None, None, None],
                2: [None, 1,    None, None, None, None],
            }
    else:
        raise Exception()

    mapping_dict = {
        0: rm0.mapping,  # region #0
        1: rm1.mapping,  # region #1
        2: rm2.mapping,  # region #2
    }

    Jacobian_matrix_dict = {
        0: rm0.Jacobian_matrix,  # region #1
        1: rm1.Jacobian_matrix,  # region #2
        2: rm2.Jacobian_matrix,  # region #3
    }

    if esd == 2:
        mtype_dict = {
            0: {'indicator': 'Linear', 'parameters': [f'x{x2}', f'y{y1}']},
            1: {'indicator': 'Linear', 'parameters': [f'x{x2}', f'y{y2}']},
            2: {'indicator': 'Linear', 'parameters': [f'x{x1}', f'y{y2}']},
        }
    elif esd == 3:
        mtype_dict = {
            0: {'indicator': 'Linear', 'parameters': [f'x{x2}', f'y{y1}', f'z{z}']},
            1: {'indicator': 'Linear', 'parameters': [f'x{x2}', f'y{y2}', f'z{z}']},
            2: {'indicator': 'Linear', 'parameters': [f'x{x1}', f'y{y2}', f'z{z}']}
        }
    else:
        raise Exception()

    return region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict, None
