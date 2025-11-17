# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from numpy import ones_like, sin, pi, cos
from random import randint

from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK


___A___ = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 0]
])

___invA___ = np.linalg.inv(___A___)


def chaotic(bounds=([0, 1], [0, 1]), c=0, periodic=False):
    r"""A msehtt mesh generator.

    Mainly for test purpose.

    A msehtt mesh generator must return REGIONS, region_map, periodic_setting.

    Returns
    -------
    REGIONS : dict
        A dict whose keys are region indices (names) and values are region instances.

        A region instance is an object that has two properties (ndim and etype) and
        two methods (mapping and Jacobian_matrix).

    region_map :
        `region_map` is None or a dictionary.
        When it is a dictionary, its keys are the same to those of `regions` and values
        are list of region corners. For example, in 2d
            region_map = {
                0: [0, 1, 2, 3],
                1: [1, 4, 5, 2],
                2: ....
            }

        means the numbering for four corners of region #0 are 0, 1, 2, 3, and
        the numbering for four corners of region #1 are 1, 4, 5, 3, and so on. Recall the following
        topology of a reference msehtt region.

          ^ s
          |
          |
          |  node 3        face 2         node 2
          |     -----------------------------
          |     |                           |
          |     |                           |
          |     |                           |
          |     | face 3                    | face 1
          |     |                           |
          |     |                           |
          |     |                           |
          |     |                           |
          |     -----------------------------
          |   node 0         face 0       node 1
          |
          |
          ------------------------------------------> r

    periodic_setting :
        The indicator of periodic region faces.

    """
    assert RANK == MASTER_RANK, f"only initialize chaotic mesh in the master rank"

    if len(bounds) == 2:
        return crazy2d(bounds=bounds, c=c, periodic=periodic)
    elif len(bounds) == 3:
        raise NotImplementedError()
    else:
        raise Exception()


# ============ 2d =====================================================================


def crazy2d(bounds=([0, 1], [0, 1]), c=0, periodic=False, shifting=True):
    r""""""
    if periodic:
        REGIONS, region_map, _ = crazy2d(bounds=bounds, c=c, periodic=False, shifting=False)

        periodic_setting = {
            (0, (0, 1)): (6, (3, 2)),
            # above line  says the node0->node1 face of region 0 is periodic to node3->node2 face of region 6.
            (1, (0, 1)): (7, (3, 2)),
            (2, (0, 1)): (8, (3, 2)),
            (0, (0, 3)): (2, (1, 2)),
            (3, (0, 3)): (5, (1, 2)),
            (6, (0, 3)): (8, (1, 2)),
        }
        return REGIONS, region_map, periodic_setting
    else:
        pass

    # ----- there will be 9 regions --------------------------------------------
    low_x, upp_x = bounds[0]
    low_y, upp_y = bounds[1]

    total_mapping = TotalMapping2D(low_x, upp_x, low_y, upp_y, c)

    assert low_x < upp_x and low_y < upp_y, f"bounds = {bounds} wrong."

    X = np.linspace(low_x, upp_x, 4)
    Y = np.linspace(low_y, upp_y, 4)

    regions = {}
    for m in range(9):
        regions[m] = {
            'x': [],
            'y': [],
        }

    shift = [randint(0, 3) for _ in range(9)]
    # shift = [0 for _ in range(9)]

    for j in range(3):
        for i in range(3):
            m = i + j * 3
            xx = [X[i], X[i+1], X[i+1], X[i]]
            yy = [Y[j], Y[j], Y[j+1], Y[j+1]]

            if shifting:
                sft = shift[m]
                if sft == 0:
                    pass
                elif sft == 1:
                    xx = [xx[1], xx[2], xx[3], xx[0]]
                    yy = [yy[1], yy[2], yy[3], yy[0]]
                elif sft == 2:
                    xx = [xx[2], xx[3], xx[0], xx[1]]
                    yy = [yy[2], yy[3], yy[0], yy[1]]
                elif sft == 3:
                    xx = [xx[3], xx[0], xx[1], xx[2]]
                    yy = [yy[3], yy[0], yy[1], yy[2]]
                else:
                    raise Exception()
            else:
                pass

            regions[m]['x'] = xx
            regions[m]['y'] = yy

    REGIONS = {}
    for m in range(9):
        A = ___invA___ @ np.array(regions[m]['x'])
        B = ___invA___ @ np.array(regions[m]['y'])
        REGIONS[m] = _Single_Map_(A, B, total_mapping)

    region_map = None        # the config method will parse the region map.
    periodic_setting = None

    return REGIONS, region_map, periodic_setting


class _Single_Map_(Frozen):
    r""""""
    def __init__(self, A, B, total_mapping):
        """
        It first maps [0, 1]^2 into a quad region (q, w). q, w can be computed by affine
        quad mapping. See `mapping`.

        Then the (q, w) region is mapping to a physical region with the total crazy mapping.

        Parameters
        ----------
        A
        B
        total_mapping
        """
        self._a1, self._a2, self._a3, self._a4 = A
        self._b1, self._b2, self._b3, self._b4 = B
        self._tm = total_mapping
        self._freeze()

    @property
    def ndim(self):
        """This is a 2d region."""
        return 2

    @property
    def etype(self):
        r"""The element made in this region can only be of this type."""
        if self._tm._c == 0:
            return 9
        else:
            return 'unique curvilinear quad'

    def mapping(self, r, s):
        """"""
        q = self._a1 + self._a2 * r + self._a3 * s + self._a4 * r * s
        w = self._b1 + self._b2 * r + self._b3 * s + self._b4 * r * s

        x, y = self._tm.mapping(q, w)

        return x, y

    def Jacobian_matrix(self, r, s):
        """Remember: not all element types will call this method. When c=0, element type=9, and this
        method will not be called at all. In that case we can leave it empty.
        """
        qr = self._a2 + self._a4 * s
        qs = self._a3 + self._a4 * r

        wr = self._b2 + self._b4 * s
        ws = self._b3 + self._b4 * r

        q = self._a1 + self._a2 * r + self._a3 * s + self._a4 * r * s
        w = self._b1 + self._b2 * r + self._b3 * s + self._b4 * r * s

        JM = self._tm.Jacobian_matrix(q, w)

        xq, xw = JM[0]
        yq, yw = JM[1]

        xr = xq * qr + xw * wr
        xs = xq * qs + xw * ws

        yr = yq * qr + yw * wr
        ys = yq * qs + yw * ws

        return (
            [xr, xs],
            [yr, ys],
        )


class TotalMapping2D(Frozen):
    r""""""

    def __init__(self, a, b, c, d, deformation_factor):
        """mapping for [a, b] * [c, d]."""
        self._abcd_ = (a, b, c, d)
        self._c = deformation_factor
        self._freeze()

    def mapping(self, p, q):
        r"""p in [a, b], q in [c, d]"""
        if self._c == 0:
            x = p
            y = q
        else:
            a, b, c, d = self._abcd_
            r = (p - a) / (b - a)
            s = (q - c) / (d - c)
            x = r + 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s)
            y = s + 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s)
            x += a
            c += c
        return x, y

    def Jacobian_matrix(self, p, q):
        """ r, s, t be in [0, 1]. """
        if self._c == 0:
            xp = ones_like(p)
            xq = 0
            yp = 0
            yq = ones_like(q)

        else:
            a, b, c, d = self._abcd_

            r = (p - a) / (b - a)
            s = (q - c) / (d - c)

            xr = 1 + 0.5 * self._c * 2 * pi * cos(2 * pi * r) * sin(2 * pi * s)
            xs = 0.5 * self._c * sin(2 * pi * r) * 2 * pi * cos(2 * pi * s)
            yr = 0.5 * self._c * 2 * pi * cos(2 * pi * r) * sin(2 * pi * s)
            ys = 1 + 0.5 * self._c * sin(2 * pi * r) * 2 * pi * cos(2 * pi * s)

            xp = xr / (b - a)
            xq = xs / (d - c)
            yp = yr / (b - a)
            yq = ys / (d - c)

        return ((xp, xq),
                (yp, yq))
