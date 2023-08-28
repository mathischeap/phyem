# -*- coding: utf-8 -*-
r"""
python tests/msepy/m2n2.py
"""
import numpy as np
import sys
import os
if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph
assert ph.config.SIZE == 1, f"msepy does not work with multiple ranks."

print(f"[MsePy] >>> <m2n2> ...")


def fx(t, x, y):
    return np.sin(2*np.pi*x) * np.sin(np.pi*y) + t


def ux(t, x, y):
    return np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + t


def uy(t, x, y):
    return np.cos(2*np.pi*x) * np.sin(np.pi*y) + t


space_dim = 2


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
    f0i_error :
    f0o_error :
    f2_error :
    f1i_error :
    f1o_error :
    df0i_error :
    df0o_error :
    df1i_error :
    df1o_error :

    """
    if isinstance(K, int):
        K = (K, K)
    else:
        pass
    if isinstance(N, int):
        N = (N, N)
    else:
        pass

    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)

    L0i = ph.space.new('Lambda', 0, orientation='inner')
    L0o = ph.space.new('Lambda', 0, orientation='outer')
    L1i = ph.space.new('Lambda', 1, orientation='inner')
    L1o = ph.space.new('Lambda', 1, orientation='outer')
    L2 = ph.space.new('Lambda', 2)

    f0i = L0i.make_form('f_i^0', '0-f-i')
    f0o = L0o.make_form('f_o^0', '0-f-o')
    f1i = L1i.make_form('f_i^1', '1-f-i')
    f1o = L1o.make_form('f_o^1', '1-f-o')
    f2 = L2.make_form('f^2', '2-f')

    ph.space.finite(N)

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = msepy.base['manifolds'][r"\mathcal{M}"]
    mesh = msepy.base['meshes'][r'\mathfrak{M}']
    f0i = obj['f0i']
    f0o = obj['f0o']
    f1i = obj['f1i']
    f1o = obj['f1o']
    f2 = obj['f2']

    msepy.config(manifold)(
        'crazy_multi', c=c, bounds=[[0, 2] for _ in range(space_dim)]
    )

    msepy.config(mesh)(K)

    scalar = ph.vc.scalar(fx)
    vector = ph.vc.vector(ux, uy)

    f2.cf = scalar
    f2[2].reduce()
    f0i.cf = scalar
    f0i[2].reduce()
    f0o.cf = scalar
    f0o[2].reduce()

    f0i_error = f0i[2].error()  # by default, we will compute the L^2 error.
    f0o_error = f0o[2].error()  # by default, we will compute the L^2 error.
    f2_error = f2[2].error()  # by default, we will compute the L^2 error.

    f1i.cf = vector
    f1i[2].reduce()
    f1o.cf = vector
    f1o[2].reduce()

    f1i_error = f1i[2].error()  # by default, we will compute the L^2 error.
    f1o_error = f1o[2].error()  # by default, we will compute the L^2 error.

    df0i = f0i.coboundary[2]
    df0i_error = df0i.error()  # by default, we will compute the L^2 error.
    df0o = f0o.coboundary[2]
    df0o_error = df0o.error()  # by default, we will compute the L^2 error.

    df1i = f1i.coboundary[2]
    df1i_error = df1i.error()  # by default, we will compute the L^2 error.
    df1o = f1o.coboundary[2]
    df1o_error = df1o.error()  # by default, we will compute the L^2 error.

    return f0i_error, f0o_error, f2_error, \
        f1i_error, f1o_error, \
        df0i_error, df0o_error, \
        df1i_error, df1o_error


current_dir = os.path.dirname(__file__)

Ns = [[1, 1, 1, 1],
      [3, 3, 3, 3]]
Ks = [[2, 4, 6, 8],
      [2, 4, 6, 8]]
cs = [0, 0.1]

pr = ph.run(test_function)


if os.path.isfile(current_dir + '/WTP2.txt'):
    os.remove(current_dir + '/WTP2.txt')

pr.iterate(
    Ks, Ns, cs,
    writeto=current_dir + '/WTP2.txt',
    show_progress=False,  # turn off iterator progress.
)

PR = ph.rdr(current_dir + '/WTP2.txt')
PR.visualize.quick('K', y='f2_error', saveto=current_dir + '/images/f2_error_quick.png')

orders = PR.visualize(
    'loglog', 'N', 'f0i_error', prime='input2', hcp=1, usetex=True,
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
    saveto=current_dir + '/images/f0i_error.png',
    return_order=True,
)
o1, o2, o3, o4 = orders
# print(orders)
assert abs(o1-2) < 0.1
assert abs(o2-4) < 0.1
assert abs(o3-2) < 0.2
assert abs(o4-4) < 0.1

orders = PR.visualize(
    'loglog', 'N', 'f2_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$',
            '$N=1,c=0.1$', '$N=3, c=0.1$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| f^0_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 1},
                         1: {'tp': (0.02, 0.2), 'order': 3}},
    saveto=current_dir + '/images/f2_error.png',
    return_order=True,
)
o1, o2, o3, o4 = orders
# print(orders)
assert abs(o1-1) < 0.1
assert abs(o2-3) < 0.1
assert abs(o3-1) < 0.1
assert abs(o4-3) < 0.2

orders = PR.visualize(
    'loglog', 'N', 'df0i_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$',
            '$N=1,c=0.1$', '$N=3, c=0.1$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| \mathrm{d}f^0_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 1},
                         1: {'tp': (0.02, 0.2), 'order': 3}},
    saveto=current_dir + '/images/df0i_error.png',
    return_order=True,
)
o1, o2, o3, o4 = orders
assert abs(o1-1) < 0.1
assert abs(o2-3) < 0.1
assert abs(o3-1) < 0.1
assert abs(o4-3) < 0.1

orders = PR.visualize(
    'loglog', 'N', 'df0o_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$',
            '$N=1,c=0.1$', '$N=3, c=0.1$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| \mathrm{d}f^0_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 1},
                         1: {'tp': (0.02, 0.2), 'order': 3}},
    saveto=current_dir + '/images/df0o_error.png',
    return_order=True,
)
o1, o2, o3, o4 = orders
assert abs(o1-1) < 0.1
assert abs(o2-3) < 0.1
assert abs(o3-1) < 0.1
assert abs(o4-3) < 0.1

orders = PR.visualize(
    'loglog', 'N', 'df1i_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$',
            '$N=1,c=0.1$', '$N=3, c=0.1$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| \mathrm{d}f^1_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 1},
                         1: {'tp': (0.02, 0.2), 'order': 3}},
    saveto=current_dir + '/images/df1i_error.png',
    return_order=True,
)
o1, o2, o3, o4 = orders
assert abs(o1-1) < 0.1
assert abs(o2-3) < 0.1
assert abs(o3-1) < 0.1
assert abs(o4-3) < 0.3

orders = PR.visualize(
    'loglog', 'N', 'df1o_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=1,c=0$', '$N=3, c=0$',
            '$N=1,c=0.1$', '$N=3, c=0.1$'],
    styles=["-s", "-v"],
    colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$1/K$',
    ylabel=r"$\left\| \mathrm{d}f^1_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    plot_order_triangle={0: {'tp': (0.02, -0.5), 'order': 1},
                         1: {'tp': (0.02, 0.2), 'order': 3}},
    saveto=current_dir + '/images/df1o_error.png',
    return_order=True,
)
o1, o2, o3, o4 = orders
assert abs(o1-1) < 0.1
assert abs(o2-3) < 0.1
assert abs(o3-1) < 0.1
assert abs(o4-3) < 0.2

os.remove(current_dir + '/WTP2.txt')

errors = test_function((15, [3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4]), (4, 5), c=0)
assert sum(errors) < 2e-4


if __name__ == '__main__':
    # python tests/msepy/m2n2.py
    pass
