# -*- coding: utf-8 -*-
"""
Taken from:
https://numpy-discussion.10968.n7.nabble.com/Any-interest-in-a-generalized-piecewise-function-td38808.html

By Per.Brodtkorb
Published on Oct 10, 2014; 11:33am

"""

import numpy as np
from collections.abc import Callable


def _zeros(*args):
    """"""
    return np.zeros_like(args[-1])


def genpiecewise(xi, condlist, funclist, fillvalue=0, args=(), **kw):
    """Evaluate a piecewise-defined function.

    Given a set of conditions and corresponding functions, evaluate each
    function on the input data wherever its condition is true.

    Parameters
    ----------
    xi : tuple, list
        input arguments to the functions in funclist, i.e., (x0, x1,...., xn)
    condlist : list of bool arrays
        Each boolean array corresponds to a function in `funclist`.  Wherever
        `condlist[i]` is True, `funclist[i](x0,x1,...,xn)` is used as the
        output value. Each boolean array in `condlist` selects a piece of `xi`,
        and should therefore be of the same shape as `xi`.

        The length of `condlist` must correspond to that of `funclist`.
        If one extra function is given, i.e. if
        ``len(funclist) - len(condlist) == 1``, then that extra function
        is the default value, used wherever all conditions are false.
    funclist : list of callables, f(*(xi + args), **kw), or scalars
        Each function is evaluated over `x` wherever its corresponding
        condition is True.  It should take an array as input and give an array
        or a scalar value as output.  If, instead of a callable,
        a scalar is provided then a constant function (``lambda x: scalar``) is
        assumed.
    fillvalue : scalar
        fillvalue for out of range values. Default 0.
    args : tuple, optional
        Any further arguments given here passed to the functions
        upon execution, i.e., if called ``piecewise(..., ..., args=(1, 'a'))``,
        then each function is called as ``f(x0, x1,..., xn, 1, 'a')``.
    kw : dict, optional
        Keyword arguments used in calling `piecewise` are passed to the
        functions upon execution, i.e., if called
        ``piecewise(..., ..., lambda=1)``, then each function is called as
        ``f(x0, x1,..., xn, lambda=1)``.

    Returns
    -------
    out : ndarray
        The output is the same shape and type as x and is found by
        calling the functions in `funclist` on the appropriate portions of `x`,
        as defined by the boolean arrays in `condlist`.  Portions not covered
        by any condition have undefined values.

    See Also
    --------
    choose, select, where

    Notes
    -----
    This is similar to choose or select, except that functions are
    evaluated on elements of `xi` that satisfy the corresponding condition from
    `condlist`.

    The result is::
            |--
            |funclist[0](x0[condlist[0]],x1[condlist[0]],...,xn[condlist[0]])
      out = |funclist[1](x0[condlist[1]],x1[condlist[1]],...,xn[condlist[1]])
            |...
            |funclist[n2](x0[condlist[n2]],x1[condlist[n2]],...,xn[condlist[n2]])
            |--

    Examples
    --------
    Define the sigma function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.
    >>> import numpy as np
    >>> x = np.linspace(-2.5, 2.5, 6)
    >>> genpiecewise(x, [x < 0, x >= 0], [-1, 1])
    array([-1., -1., -1.,  1.,  1.,  1.])

    Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for ``x >= 0``.
    >>> genpiecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    array([2.5, 1.5, 0.5, 0.5, 1.5, 2.5])

    An example with two parameters.
    >>> def Yp1(p, q): return -np.pi*5*0.3*np.cos(np.pi*5*p)*(1-(-5*p+2.5)/2.5)
    >>> def Yp2(p, q): return -np.pi*5*0.3*np.cos(np.pi*5*p)*(1-(5*p-2.5)/2.5)
    >>> p = np.linspace(0,1,10)
    >>> q = np.linspace(0,1,10)
    >>> Yp = genpiecewise((p,q), [p<0.5, p>=0.5], [Yp1, Yp2])
    >>> Yp[:]
    array([-0.        ,  0.18184395,  1.96808762, -1.57079633, -3.20879946,
            3.20879946,  1.57079633, -1.96808762, -0.18184395,  0.        ])

    """
    # noinspection PyTypeChecker
    nc, nf = len(condlist), len(funclist)
    if nc not in [nf - 1, nf]:
        raise ValueError("function list and condition list" +
                         " must be the same length")

    if not isinstance(xi, (list, tuple)):
        xi = (xi,)
    condlist = np.broadcast_arrays(*condlist)
    # noinspection PyTypeChecker
    if len(condlist) == len(funclist) - 1:
        # noinspection PyUnresolvedReferences
        condlist.append(~np.logical_or.reduce(condlist, axis=0))

    arrays = np.broadcast_arrays(*xi)
    dtype = np.result_type(*arrays)

    # noinspection PyUnresolvedReferences
    out = np.ones(condlist[0].shape, dtype=bool) * fillvalue
    if dtype is not None:
        out = out.astype(dtype)
    if not isinstance(out, np.ndarray):
        out = np.asarray(out)

    for cond, func in zip(condlist, funclist):
        if isinstance(func, Callable):
            temp = tuple(np.extract(cond, arr) for arr in arrays) + args
            np.place(out, cond, func(*temp, **kw))
        else:  # func is a scalar value
            np.place(out, cond, func)

    return out


if __name__ == '__main__':
    # python tools/gen_piece_wise.py
    import doctest

    doctest.testmod()

    def Yp0(p, q): return 0 * p * q

    def Yp1(p, q): return 1 + 0 * p * q

    def Yp2(p, q): return 2 + 0 * p * q

    p = np.linspace(0, 1, 10)
    q = np.linspace(0, 1, 10)
    Yp = genpiecewise([p, q], [p < 0.2, np.logical_and(p >= 0.2, p < 0.5), p >= 0.5], [Yp0, Yp1, Yp2])

    print(Yp)
