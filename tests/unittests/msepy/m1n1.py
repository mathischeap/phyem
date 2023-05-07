# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 6:59 PM on 5/2/2023

$ python .\tests\unittests\msepy\m1n1.py
"""
import numpy as np
import sys
import os
if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph

from tools.run.reader import ParallelMatrix3dInputRunner, RunnerDataReader


def fx(t, x):
    """"""
    return np.sin(2*np.pi*x) + t


space_dim = 1
ph.config.set_embedding_space_dim(space_dim)

manifold = ph.manifold(space_dim)
mesh = ph.mesh(manifold)

L0 = ph.space.new('Lambda', 0)
L1 = ph.space.new('Lambda', 1)


def test_function(K, N, c):
    """

    Parameters
    ----------
    K :
        elements
    N :
        Polynomial degree
    c :
        Mesh deformation factor.

    Returns
    -------
    f0_error :
    f1_error :
    df_error :

    """
    ph.clear_forms()

    f0 = L0.make_form('f^0', '0-f')
    f1 = L1.make_form('f^1', '1-f')

    ph.space.finite(N)

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = msepy.base['manifolds'][r"\mathcal{M}"]
    mesh = msepy.base['meshes'][r'\mathfrak{M}']
    f0 = obj['f0']
    f1 = obj['f1']

    msepy.config(manifold)(
        'crazy_multi', c=c, bounds=[[0, 2] for _ in range(space_dim)]
    )

    msepy.config(mesh)(K)

    scalar = ph.vc.scalar(fx)

    f1.cf = scalar
    f1[2].reduce()
    f0.cf = scalar
    f0[2].reduce()

    f0_error = f0[2].error()  # by default, we will compute the L^2 error.
    f1_error = f1[2].error()  # by default, we will compute the L^2 error.

    df0 = f0.coboundary[2]()
    df_error = df0[2].error()  # by default, we will compute the L^2 error.

    return f0_error, f1_error, df_error


current_dir = os.path.dirname(__file__)


Ns = [[1, 1, 1, 1],
      [3, 3, 3, 3]]
Ks = [[2, 4, 6, 8],
      [2, 4, 6, 8]]
cs = [0, 0.1]

pr = ParallelMatrix3dInputRunner(test_function)


if os.path.isfile(current_dir + '/WTP.txt'):
    os.remove(current_dir + '/WTP.txt')

pr.iterate(
    Ks, Ns, cs,
    writeto=current_dir + '/WTP.txt',
    show_progress=False,  # turn off iterator progress.
)

PR = RunnerDataReader(current_dir + '/WTP.txt')
PR.visualize.quick('N', y='f0_error', saveto=current_dir + '/images/f0_error_quick.png')
PR.visualize.quick('K', y='f1_error', saveto=current_dir + '/images/f1_error_quick.png')

orders = PR.visualize(
    'loglog', 'N', 'f0_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$',
            '$N=1,c=0.1$', '$N=3, c=0.1$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| f^0_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 2},
                         1: {'tp': (0.02, 0.2), 'order': 4}},
    saveto=current_dir + '/images/f0_error.png',
    return_order=True,
)
o1, o2, o3, o4 = orders
assert abs(o1-2) < 0.1
assert abs(o2-4) < 0.1
assert abs(o3-2) < 0.1
assert abs(o4-4) < 0.3
orders = PR.visualize(
    'loglog', 'N', 'f1_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$',
            '$N=1,c=0.1$', '$N=3, c=0.1$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| f^1_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 1},
                         1: {'tp': (0.02, 0.2), 'order': 3}},
    saveto=current_dir + '/images/f1_error.png',
    return_order=True,
)
o1, o2, o3, o4 = orders
assert abs(o1-1) < 0.1
assert abs(o2-3) < 0.1
assert abs(o3-1) < 0.5
assert abs(o4-3) < 0.5

os.remove(current_dir + '/WTP.txt')


if __name__ == '__main__':
    # python .\tests\unittests\msepy\m1n1.py
    test_function(3, 2, 0)
