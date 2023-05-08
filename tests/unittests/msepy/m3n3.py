# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:27 PM on 5/8/2023

$ python tests/unittests/msepy/m3n3.py
"""
import numpy as np
import sys
import os
if './' not in sys.path:
    sys.path.append('./')
import __init__ as ph
assert ph.config.SIZE == 1, f"msepy does not work with multiple ranks."

from tools.runner import ParallelMatrix3dInputRunner, RunnerDataReader

print(f"[MsePy] >>> <m3n3> ...")


def fx(t, x, y, z):
    return np.cos(2*np.pi*x) * np.cos(np.pi*y) * np.cos(np.pi*z) + t


def ux(t, x, y, z):
    return np.sin(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) + t


def uy(t, x, y, z):
    return np.cos(2*np.pi*x) * np.sin(2*np.pi*y) * np.cos(2*np.pi*z) + t


def uz(t, x, y, z):
    return np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + t


space_dim = 3
ph.config.set_embedding_space_dim(space_dim)

manifold = ph.manifold(space_dim)
mesh = ph.mesh(manifold)

L0 = ph.space.new('Lambda', 0)
L1 = ph.space.new('Lambda', 1)
L2 = ph.space.new('Lambda', 2)
L3 = ph.space.new('Lambda', 3)


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
    f2_error :
    f3_error :
    df0_error :
    df1_error :
    df2_error :

    """
    if isinstance(K, int):
        K = (K, K, K)
    else:
        pass
    if isinstance(N, int):
        N = (N, N, N)
    else:
        pass

    ph.clear_forms()

    f0 = L0.make_form('f^0', '0-f')
    f1 = L1.make_form('f^1', '1-f')
    f2 = L2.make_form('f^2', '2-f')
    f3 = L3.make_form('f^3', '3-f')

    ph.space.finite(N)

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = msepy.base['manifolds'][r"\mathcal{M}"]
    mesh = msepy.base['meshes'][r'\mathfrak{M}']

    f0 = obj['f0']
    f1 = obj['f1']
    f2 = obj['f2']
    f3 = obj['f3']

    msepy.config(manifold)(
        'crazy_multi', c=c, bounds=[[0, 2] for _ in range(space_dim)]
    )

    msepy.config(mesh)(K)

    scalar = ph.vc.scalar(fx)
    vector = ph.vc.vector(ux, uy, uz)

    f0.cf = scalar
    f0[2].reduce()
    f1.cf = vector
    f1[2].reduce()
    f2.cf = vector
    f2[2].reduce()
    f3.cf = scalar
    f3[2].reduce()

    f0_error = f0[2].error()  # by default, we will compute the L^2 error.
    f1_error = f1[2].error()  # by default, we will compute the L^2 error.
    f2_error = f2[2].error()  # by default, we will compute the L^2 error.
    f3_error = f3[2].error()  # by default, we will compute the L^2 error.

    df0 = f0.coboundary[2]()
    df1 = f1.coboundary[2]()
    df2 = f2.coboundary[2]()

    df0_error = df0[2].error()  # by default, we will compute the L^2 error.
    df1_error = df1[2].error()  # by default, we will compute the L^2 error.
    df2_error = df2[2].error()  # by default, we will compute the L^2 error.

    return f0_error, f1_error, f2_error, f3_error, df0_error, df1_error, df2_error


current_dir = os.path.dirname(__file__)

Ns = [[1, 1, 1],
      [3, 3]]
Ks = [[4, 6, 8],
      [4, 6]]
cs = [0, ]

pr = ParallelMatrix3dInputRunner(test_function)


if os.path.isfile(current_dir + '/WTP3.txt'):
    os.remove(current_dir + '/WTP3.txt')

pr.iterate(
    Ks, Ns, cs,
    writeto=current_dir + '/WTP3.txt',
    show_progress=False,  # turn off iterator progress.
)


PR = RunnerDataReader(current_dir + '/WTP3.txt')
PR.visualize.quick('K', y='f3_error', saveto=current_dir + '/images/f3_error_quick.png')

orders = PR.visualize(
    'loglog', 'N', 'f0_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| f^0_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 2},
                         1: {'tp': (0.02, 0.2), 'order': 4}},
    saveto=current_dir + '/images/f0_3d_error.png',
    return_order=True,
)
o1, o2 = orders
assert abs(o1-2) < 0.1
assert abs(o2-4) < 0.1

orders = PR.visualize(
    'loglog', 'N', 'f1_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| f^0_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 2},
                         1: {'tp': (0.02, 0.2), 'order': 4}},
    saveto=current_dir + '/images/f1_3d_error.png',
    return_order=True,
)
o1, o2 = orders
assert abs(o1-1) < 0.2
assert abs(o2-3) < 0.1

orders = PR.visualize(
    'loglog', 'N', 'f2_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| f^0_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 2},
                         1: {'tp': (0.02, 0.2), 'order': 4}},
    saveto=current_dir + '/images/f2_3d_error.png',
    return_order=True,
)
o1, o2 = orders
assert abs(o1-1) < 0.1
assert abs(o2-3) < 0.1

orders = PR.visualize(
    'loglog', 'N', 'f3_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| f^0_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 2},
                         1: {'tp': (0.02, 0.2), 'order': 4}},
    saveto=current_dir + '/images/f3_3d_error.png',
    return_order=True,
)
o1, o2 = orders
assert abs(o1-1) < 0.1
assert abs(o2-3) < 0.1

os.remove(current_dir + '/WTP3.txt')

errors = test_function(([3, 3, 2, 3, 2], [2, 3, 3, 2], [1, 1, 1, 1, 1.5]), (4, 5, 4), c=0)
assert sum(errors) < 0.03


if __name__ == '__main__':
    # python tests/unittests/msepy/m3n3.py
    pass
