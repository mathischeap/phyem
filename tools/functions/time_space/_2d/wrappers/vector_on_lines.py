r"""
On different lines (1d geometries in 2d space), we can define different vector functions. This is
useful for example when defining boundary conditions.

"""
from tools.miscellaneous.geometries.m2n2 import StraightSegment2, Curve2
from tools.miscellaneous.geometries.m2n2 import whether_point_on_straight_segment
from tools.miscellaneous.geometries.m2n2 import whether_point_on_curve

from tools.functions.time_space._2d.wrappers.vector import T2dVector
from tools.functions.time_space._2d.constant import cfg_t

from tools.frozen import Frozen
import numpy as np


def vector_on_lines(*vector_on_lines_pairs):
    r"""

    Parameters
    ----------
    vector_on_lines_pairs :

        For example, vector_on_lines_pairs = (
            (a, (f1x, f1y)),
            (b, (f2x, f2y)),
            (c, t2dv),
            ...
        )
        where a, b, c, d are StraightSegment2 or Curve2 instances.

        where f1x, f1y, f2x, f2y are functions takes (t, x, y) are return one value. That is saying, we can
        for example use (f1x, f1y) to define a T2dVector instance or use f1x to define a T2dScalar.

        And t2dv is a T2dVector instance.

    Returns
    -------
    vol:
        Vector On Lines. A T2dVector instance. On the defined lines, we use the corresponding T2dVector instance
        to compute the vector. Other wise, it returns (0, 0) vector.

    """
    gen_func = _GenFunction_VOL_()
    for pair in vector_on_lines_pairs:
        cond, func = pair
        assert cond.__class__ in (StraightSegment2, Curve2), f"conditions must be StraightSegment2 or Curve2 object."
        if func.__class__ is T2dVector:
            fun_x, fun_y = func._v0_, func._v1_
        else:
            assert len(func) == 2, f"I must receive two functions."
            fun_x, fun_y = func
            if isinstance(fun_x, (int, float)):
                fun_x = cfg_t(fun_x)
            if isinstance(fun_y, (int, float)):
                fun_y = cfg_t(fun_y)

        gen_func.add(fun_x, fun_y, cond)

    return T2dVector(gen_func._call_x_, gen_func._call_y_, v0v1_linked_object=gen_func)


class _GenFunction_VOL_(Frozen):
    r""""""
    def __init__(self):
        r""""""
        self._funX = []
        self._funY = []
        self._cond = []
        self._freeze()

    def add(self, funX, funY, condition):
        r""""""
        assert condition.__class__ in (StraightSegment2, Curve2), \
            f"A condition must be a StraightSegment2 or Curve2 object"
        self._funX.append(funX)
        self._funY.append(funY)
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

    def _call_x_(self, t, x, y):
        r"""compute the x component only."""
        return self(t, x, y)[0]

    def _call_y_(self, t, x, y):
        r"""compute the y component only."""
        return self(t, x, y)[1]

    def __call__(self, t, x, y):
        r""""""
        if isinstance(x, (int, float)):
            assert isinstance(y, (int, float)), f"if x is a number, y must also be a number."
            cond = self.find_point_satisfy_which_condition(x, y)
            if cond is None:
                return 0, 0
            else:
                return self._funX[cond](t, x, y), self._funY[cond](t, x, y)

        elif isinstance(y, (int, float)):
            raise Exception(f"if y is a number, x must also be a number.")
        else:
            pass

        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), f"x, y must be numpy array."
        assert x.shape == y.shape, f"x, y must be array of the same shape."

        U = np.zeros_like(x)
        V = np.zeros_like(y)
        x_ndim = np.ndim(x)
        if x_ndim == 1:
            for i, xi in enumerate(x):
                yi = y[i]
                cond = self.find_point_satisfy_which_condition(xi, yi)
                if cond is None:
                    pass
                else:
                    U[i] = self._funX[cond](t, xi, yi)
                    V[i] = self._funY[cond](t, xi, yi)

        elif x_ndim == 2:
            for i, xi in enumerate(x):
                yi = y[i]
                for j, xij in enumerate(xi):
                    yij = yi[j]
                    cond = self.find_point_satisfy_which_condition(xij, yij)
                    if cond is None:
                        pass
                    else:
                        U[i, j] = self._funX[cond](t, xij, yij)
                        V[i, j] = self._funY[cond](t, xij, yij)

        else:
            raise NotImplementedError()

        return U, V
