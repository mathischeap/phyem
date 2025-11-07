r"""
On different lines (1d geometries in 2d space), we can define different scalars. This is
useful for example when defining (pressure, temperature) boundary conditions.

"""
from tools.miscellaneous.geometries.m2n2 import StraightSegment2, Curve2
from tools.miscellaneous.geometries.m2n2 import whether_point_on_straight_segment
from tools.miscellaneous.geometries.m2n2 import whether_point_on_curve

from tools.functions.time_space._2d.wrappers.scalar import T2dScalar

from tools.frozen import Frozen
import numpy as np


from tools.functions.time_space._2d.constant import cfg_t


def scalar_on_lines(*scalar_on_lines_pairs):
    r"""

    Parameters
    ----------
    scalar_on_lines_pairs :

        For example, vector_on_lines_pairs = (
            (a, f1),
            (b, f2),
            (c, t2ds),
            ...
        )
        where a, b, c, d are StraightSegment2 or Curve2 instances.

        where f1, f2 are functions takes (t, x, y) are return one value. That is saying, we can
        for example use f1 or f2 to define a T2dScalar instance.

        And t2ds is a T2dScalar instance.

    Returns
    -------
    sol:
        Scalar On Lines. A T2dScalar instance. On the defined lines, we use the corresponding T2dScalar instance
        to compute the scalar. Other wise, it returns as a 0-valued function (scalar).

    """
    gen_func = _GenFunction_SOL_()
    for pair in scalar_on_lines_pairs:
        cond, func = pair
        assert cond.__class__ in (StraightSegment2, Curve2), f"conditions must be StraightSegment2 or Curve2 object."
        if func.__class__ is T2dScalar:
            func = func._s_
        elif isinstance(func, (int, float)):
            func = cfg_t(func)
        else:
            pass

        gen_func.add(func, cond)

    return T2dScalar(gen_func)


class _GenFunction_SOL_(Frozen):
    r""""""
    def __init__(self):
        r""""""
        self._func = []
        self._cond = []
        self._freeze()

    def add(self, func, condition):
        r""""""
        assert condition.__class__ in (StraightSegment2, Curve2), \
            f"A condition must be a StraightSegment2 or Curve2 object"
        self._func.append(func)
        self._cond.append(condition)

    def find_point_satisfy_which_condition(self, x, y):
        r""""""
        for i, cond in enumerate(self._cond):
            if cond.__class__ is StraightSegment2:
                if whether_point_on_straight_segment((x, y), cond):
                    return i
                else:
                    pass
            elif cond.__class__ is Curve2:
                if whether_point_on_curve((x, y), cond):
                    return i
                else:
                    pass
            else:
                raise NotImplementedError()
        return None

    def __call__(self, t, x, y):
        r""""""
        if isinstance(x, (int, float)):
            assert isinstance(y, (int, float)), f"if x is a number, y must also be a number."
            cond_i = self.find_point_satisfy_which_condition(x, y)
            if cond_i is None:
                return 0
            else:
                return self._func[cond_i](t, x, y)

        elif isinstance(y, (int, float)):
            raise Exception(f"if y is a number, x must also be a number.")
        else:
            pass

        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), f"x, y must be numpy array."
        assert x.shape == y.shape, f"x, y must be array of the same shape."

        U = np.zeros_like(x)
        x_ndim = np.ndim(x)
        if x_ndim == 1:
            for i, xi in enumerate(x):
                yi = y[i]
                cond = self.find_point_satisfy_which_condition(xi, yi)
                if cond is None:
                    pass
                else:
                    U[i] = self._func[cond](t, xi, yi)

        elif x_ndim == 2:
            for i, xi in enumerate(x):
                yi = y[i]
                for j, xij in enumerate(xi):
                    yij = yi[j]
                    cond = self.find_point_satisfy_which_condition(xij, yij)
                    if cond is None:
                        pass
                    else:
                        U[i, j] = self._func[cond](t, xij, yij)

        else:
            raise NotImplementedError()

        return U
