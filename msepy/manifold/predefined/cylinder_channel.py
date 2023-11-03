# -*- coding: utf-8 -*-
r"""
.. testsetup:: *

    import __init__ as ph
    from msepy.manifold.predefined.cylinder_channel import _make_an_illustration
    _make_an_illustration(
        './source/gallery/msepy_domains_and_meshes/msepy/cylinder_channel_2d.png'
    )
    None_or_custom_path = './source/gallery/msepy_domains_and_meshes/msepy/cylinder_channel_example.png'


.. testcleanup::

    pass

The cylinder channel is a mesh (or domain) in :math:`\mathbb{R}^n`, :math:`n\in\left\lbrace2,3\right\rbrace`.
The 2d domain is illustrated in the following figure.

.. figure:: cylinder_channel_2d.png
    :width: 100%

    The illustration of the 2d cylinder channel domain.


.. autofunction:: msepy.manifold.predefined.cylinder_channel.cylinder_channel


Examples
========

2d
--

>>> ph.config.set_embedding_space_dim(2)
>>> manifold = ph.manifold(2)
>>> mesh = ph.mesh(manifold)
>>> msepy, obj = ph.fem.apply('msepy', locals())
>>> manifold = obj['manifold']
>>> mesh = obj['mesh']
>>> msepy.config(manifold)('cylinder_channel')
>>> msepy.config(mesh)(3)
>>> mesh.visualize(saveto=None_or_custom_path)  # doctest: +ELLIPSIS
<Figure size ...

.. figure:: cylinder_channel_example.png
    :width: 100%

    The cylinder_channel mesh of element factor 3.

"""

import sys

if './' not in sys.path:
    sys.path.append('./')

import numpy as np
from msepy.manifold.predefined._helpers import _LinearTransformation, _Transfinite2

import matplotlib.pyplot as plt


def _make_an_illustration(saveto, r=1, dl=10, dr=25, h=6):
    """Make a picture illustrating the domain."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(
        axis='both',
        which='both',
        left=False,
        bottom=False,
        labelbottom=False,
        labelleft=False
    )

    x = [-dl, dr, dr, -dl, -dl]
    y = [-h/2, -h/2, h/2, h/2, -h/2]

    angle = np.linspace(0, 2*np.pi, 100)
    rx = r * np.cos(angle)
    ry = r * np.sin(angle)

    plt.plot(x, y, '-k', linewidth=0.8)
    plt.plot(rx, ry, '-k', linewidth=0.8)
    _r = 1.1*dr
    plt.plot([0, _r], [0, 0], '-', linewidth=0.8, color='lightgray')
    _ = 0.05 * h
    plt.plot([_r - _, _r, _r - _], [_, 0, -_], '-', linewidth=0.8, color='lightgray')
    plt.text(_r, 0, r"$x$", c='gray', va='bottom', ha='left')
    _y = 1.4 * h / 2
    plt.plot([0, 0], [0, _y], '-', linewidth=0.8, color='lightgray')
    plt.plot([-_, 0, _], [_y-_, _y, _y-_], '-', linewidth=0.8, color='lightgray')
    plt.text(0, _y, r"$y$", c='gray', va='bottom', ha='left')
    plt.text(0, 0, r"$r$", c='k', va='bottom', ha='left')
    plt.plot([0, r], [0, 0], c='k', linewidth=0.8)
    plt.plot([0, 0], [0, -h/2], c='lightgray', linewidth=0.8)
    plt.text(-dl, 0, r"$h$", va='center', ha='left')
    plt.text(-dl/2, -h/2, r"$d_l$", va='bottom', ha='center')
    plt.text(dr/2, -h/2, r"$d_r$", va='bottom', ha='center')
    plt.savefig(saveto, bbox_inches='tight', dpi=200)
    plt.close()


def cylinder_channel(r=1, dl=8, dr=25, h=6, w=0, periodic=True):
    r"""

    Parameters
    ----------
    r : float, default=1
        The radius of the cylinder.
    dl : float, default=8
        The :math:`x` distance from the left boundary to the cylinder center.
    dr : float, default=25
        The :math:`x` distance from the right boundary to the cylinder center.
    h : float, default=6
        The height (along :math:`y`-direction, :math:`[-h/2, h/2]`) of the channel; must have
        :math:`h/2 > r`.
    w : float, default=0
        The width (along :math:`z`-direction, :math:`[-w/2, w/2]`) of the channel.
    periodic : bool, default=True
        When the domain is 3d, whether it is periodic along the :math:`z`-axis? It has no affect
        when ``w=0`` (the domain is 2d).

    """
    raise Exception(r, dl, dr, h, w, periodic)


# noinspection PyPep8Naming
class _CylinderChannel(object):
    r"""          ^  y
                  |
    ______________|______________________________________
    |                                                   |
    |            ___                                    |
    |           /  r\                                   |
    |h         |  .--|                                  |----------> x
    |           \_|_/                                   |
    |             |                                     |
    |______dl_____|__________________dr_________________|


    Regions are distributed as:

                 ^ y
                 |
    __________________________________________
    |   5   |     6    |          7          |
    |_______|__________|_____________________|
    |   3   /          \          4          |  --->x
    |_______\__________/_____________________|
    |   0   |     1    |          2          |
    |_______|__________|_____________________|

    """
    def __init__(self, mf, r=1, dl=8, dr=25, h=6, w=0, periodic=True):
        """
        Parameters
        ----------
        mf
        r
        dl
        dr
        h
        w
        periodic

        Returns
        -------

        """
        self._mf = mf
        self._r = r
        self._dl = dl
        self._dr = dr
        self._h = h
        self._w = w
        self._periodic = periodic

        assert dl > 2 * r and (h/2) > 2 * r and dr > 5 * r, f"shape wrong!"
        assert mf.esd == mf.ndim, f"_cylinder_channel mesh only works for manifold.ndim == embedding space dimensions."
        assert mf.esd in (2, 3), f"_cylinder_channel mesh only works in 2-, 3-dimensions."
        esd = mf.esd
        self._esd = esd

        if w == 0:
            assert esd == 2, f"w==0, space must be 2d"
        else:
            assert w > 0 and esd == 3, f"w>0, space must be 3d"

        if esd == 2:
            region_map = {
                0: [None, 1, None, 3],
                1: [0, 2, None, None],
                2: [1, None, None, 4],
                3: [None, None, 0, 5],
                4: [None, None, 2, 7],
                5: [None, 6, 3, None],
                6: [5, 7, None, None],
                7: [6, None, 4, None],
            }
        elif esd == 3:
            if periodic:
                region_map = {
                    0: [None, 1, None, 3, 0, 0],
                    1: [0, 2, None, None, 1, 1],
                    2: [1, None, None, 4, 2, 2],
                    3: [None, None, 0, 5, 3, 3],
                    4: [None, None, 2, 7, 4, 4],
                    5: [None, 6, 3, None, 5, 5],
                    6: [5, 7, None, None, 6, 6],
                    7: [6, None, 4, None, 7, 7],
                }
            else:
                region_map = {
                    0: [None, 1, None, 3, None, None],
                    1: [0, 2, None, None, None, None],
                    2: [1, None, None, 4, None, None],
                    3: [None, None, 0, 5, None, None],
                    4: [None, None, 2, 7, None, None],
                    5: [None, 6, 3, None, None, None],
                    6: [5, 7, None, None, None, None],
                    7: [6, None, 4, None, None, None],
                }
        else:
            raise Exception()

        hr = 0.5 * r * np.sqrt(2)

        if esd == 2:
            tf1 = _Transfinite2(
                ['straight line', [(-hr, -h/2), (-hr, -hr)]],
                ['straight line', [(hr, -h/2), (hr, -hr)]],
                ['straight line', [(-hr, -h/2), (hr, -h/2)]],
                ['anticlockwise arc', [(0, 0), (-hr, -hr), (hr, -hr)]],
            )
            tf3 = _Transfinite2(
                ['straight line', [(-dl, -hr), (-dl, hr)]],
                ['clockwise arc', [(0, 0), (-hr, -hr), (-hr, hr)]],
                ['straight line', [(-dl, -hr), (-hr, -hr)]],
                ['straight line', [(-dl, hr), (-hr, hr)]],
            )
            tf4 = _Transfinite2(
                ['anticlockwise arc', [(0, 0), (hr, -hr), (hr, hr)]],
                ['straight line', [(dr, -hr), (dr, hr)]],
                ['straight line', [(hr, -hr), (dr, -hr)]],
                ['straight line', [(hr, hr), (dr, hr)]],
            )
            tf6 = _Transfinite2(
                ['straight line', [(-hr, hr), (-hr, h/2)]],
                ['straight line', [(hr, hr), (hr, h/2)]],
                ['clockwise arc', [(0, 0), (-hr, hr), (hr, hr)]],
                ['straight line', [(-hr, h/2), (hr, h/2)]],
            )
            rm0 = _LinearTransformation(-dl, -hr, -h/2, -hr)
            rm1 = tf1
            rm2 = _LinearTransformation(hr, dr, -h/2, -hr)
            rm3 = tf3
            rm4 = tf4
            rm5 = _LinearTransformation(-dl, -hr, hr, h/2)
            rm6 = tf6
            rm7 = _LinearTransformation(hr, dr, hr, h/2)
        elif esd == 3:
            raise NotImplementedError()
        else:
            raise Exception()

        mapping_dict = {
            0: rm0.mapping,
            1: rm1.mapping,
            2: rm2.mapping,
            3: rm3.mapping,
            4: rm4.mapping,
            5: rm5.mapping,
            6: rm6.mapping,
            7: rm7.mapping,
        }

        Jacobian_matrix_dict = {
            0: rm0.Jacobian_matrix,
            1: rm1.Jacobian_matrix,
            2: rm2.Jacobian_matrix,
            3: rm3.Jacobian_matrix,
            4: rm4.Jacobian_matrix,
            5: rm5.Jacobian_matrix,
            6: rm6.Jacobian_matrix,
            7: rm7.Jacobian_matrix,
        }

        if esd == 2:
            mtype_dict = {
                0: rm0.mtype,
                1: rm1.mtype,  # unique region
                2: rm2.mtype,
                3: rm3.mtype,  # unique region
                4: rm4.mtype,  # unique region
                5: rm5.mtype,
                6: rm6.mtype,  # unique region
                7: rm7.mtype,
            }
        elif esd == 3:
            raise NotImplementedError()
        else:
            raise Exception()

        default_element_layout = self._cylinder_channel_default_element_layout

        self._para = (
            region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict, default_element_layout
        )

    def __call__(self, *args, **kwargs):
        return self._para

    def _cylinder_channel_default_element_layout(self, characteristic_element_number):
        """default_element_layout_maker must return a dict indicating the element layouts in all regions.

        When we config the mesh, if only one number argument is provided, this method will be called to
        make a default element layout with the only argument being the `characteristic_element_number`.

        For this mesh, ``characteristic_element_number`` indicating the element number of
        1/4 of the cylinder. Thus, around the cylinder, there will be 4 * ``characteristic_element_number``
        elements in total.

        """
        assert characteristic_element_number > 0 and characteristic_element_number % 1 == 0, \
            f"characteristic_element_number = {characteristic_element_number} is wrong, must be positive integer."

        arc_length = 2 * np.pi * self._r * 0.25
        hr = 0.5 * self._r * np.sqrt(2)
        c_elements = characteristic_element_number

        # x-direction of #0, 3, 5
        left_elements = int(((self._dl - hr) / arc_length) * characteristic_element_number) + 1
        # x-direction of #2, 4, 7
        right_elements = int(((self._dr - hr) / arc_length) * characteristic_element_number) + 1
        # y-direction of #0, 1, 2, 5, 6, 7
        height_elements = int(((self._h/2 - hr) / arc_length) * characteristic_element_number) + 1

        if self._esd == 2:
            pass
        else:
            raise NotImplementedError(f"compute z-direction elements.")

        element_layout = dict()
        if self._esd == 2:
            element_layout[0] = [left_elements, height_elements]
            element_layout[1] = [c_elements, height_elements]
            element_layout[2] = [right_elements, height_elements]
            element_layout[3] = [left_elements, c_elements]
            element_layout[4] = [right_elements, c_elements]
            element_layout[5] = [left_elements, height_elements]
            element_layout[6] = [c_elements, height_elements]
            element_layout[7] = [right_elements, height_elements]
        else:
            raise NotImplementedError()

        return element_layout


if __name__ == '__main__':
    # python msepy/manifold/predefined/cylinder_channel.py
    # _make_an_illustration('cylinder_channel.png')
    pass
