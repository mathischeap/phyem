# -*- coding: utf-8 -*-
r"""
"""


def derivative(f, x, method='central', h=0.00001):
    r"""Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    x : number
        Compute derivative at x
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: (f(a+h) - f(a-h))/2h
            forward: (f(a+h) - f(a))/h
            backward: (f(a) - f(a-h))/h
    """
    if method == 'central':
        return (f(x + h) - f(x - h))/(2 * h)
    elif method == 'forward':
        return (f(x + h) - f(x))/h
    elif method == 'backward':
        return (f(x) - f(x - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
