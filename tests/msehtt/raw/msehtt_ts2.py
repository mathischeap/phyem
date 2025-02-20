# -*- coding: utf-8 -*-
"""
Test the reduction and reconstruction for msehtt mesh built upon msepy 2d meshes.

mpiexec -n 4 python tests/msehtt/raw/msehtt_ts2.py
"""

import sys

import numpy as np

ph_dir = './'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import __init__ as ph


def ph_test(N, K, ts=1):
    r""""""
    ph.config.set_embedding_space_dim(2)
    ph.config.set_high_accuracy(True)
    ph.config.set_pr_cache(False)

    manifold = ph.manifold(2)
    mesh = ph.mesh(manifold)

    Out0 = ph.space.new('Lambda', 0, orientation='outer')
    Out1 = ph.space.new('Lambda', 1, orientation='outer')
    Out2 = ph.space.new('Lambda', 2, orientation='outer')

    Inn0 = ph.space.new('Lambda', 0, orientation='inner')
    Inn1 = ph.space.new('Lambda', 1, orientation='inner')
    Inn2 = ph.space.new('Lambda', 2, orientation='inner')

    o0 = Out0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
    o1 = Out1.make_form(r'\tilde{\omega}^1', 'outer-form-1')
    o2 = Out2.make_form(r'\tilde{\omega}^2', 'outer-form-2')

    i0 = Inn0.make_form(r'{\omega}^0', 'inner-form-0')
    i1 = Inn1.make_form(r'{\omega}^1', 'inner-form-1')
    i2 = Inn2.make_form(r'{\omega}^2', 'inner-form-2')

    ph.space.finite(N)

    # ------------- implementation ---------------------------------------------------
    msehtt, obj = ph.fem.apply('msehtt-s', locals())
    tgm = msehtt.tgm()
    msehtt.config(tgm)('crazy', element_layout=K, c=0, trf=1, ts=ts)
    # msehtt.config(tgm)('chaotic', element_layout=K, c=c, ts=True)
    # msehtt.config(tgm)('quad', element_layout=K, B=(np.sqrt(3),0), C=(np.sqrt(3), 1), D=(np.sqrt(3)/2, 3/2))
    # msehtt.config(tgm)('quad', element_layout=K, B=(1,0), C=(1, 1), D=(0, 1))
    # tgm.visualize(internal_grid=0)

    msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
    msehtt.config(msehtt_mesh)(tgm, including='all')

    fi0 = obj['i0']
    fi1 = obj['i1']
    fi2 = obj['i2']

    fo0 = obj['o0']
    fo1 = obj['o1']
    fo2 = obj['o2']


    def fx(t, x, y):
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(t)


    def fy(t, x, y):
        return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.exp(t)


    def fw(t, x, y):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.exp(t)


    vector = ph.vc.vector(fx, fy)
    scalar = ph.vc.scalar(fw)

    fo0.cf = scalar
    fo0[0].reduce()
    fo1.cf = vector
    fo1[0].reduce()
    fo2.cf = scalar
    fo2[0].reduce()

    # fo2[0].visualize.matplot()

    o_err0 = fo0[0].error()
    o_err1 = fo1[0].error()
    o_err2 = fo2[0].error()
    dofs0 = fo0.cochain.num_global_dofs
    dofs1 = fo1.cochain.num_global_dofs
    dofs2 = fo2.cochain.num_global_dofs

    # fi0.cf = scalar
    # fi0[0].reduce()
    # fi1.cf = vector
    # fi1[0].reduce()
    # fi2.cf = scalar
    # fi2[0].reduce()
    #
    # i_err0 = fi0[0].error()
    # i_err1 = fi1[0].error()
    # i_err2 = fi2[0].error()

    return [dofs0, dofs1, dofs2], [o_err0, o_err1, o_err2]  # i_err0, i_err1, i_err2


if __name__ == '__main__':
    # mpiexec -n 4 python tests/msehtt/raw/msehtt_ts2.py

    N = 1
    K = 4

    Dofs0, e0 = ph_test(N, K, ts=1)
    Dofs1, e1 = ph_test(N, K, ts=3)
    order = list()
    for i, _ in enumerate(e0):
        dofs0 = np.sqrt(Dofs0[i])
        dofs1 = np.sqrt(Dofs1[i])
        order.append(float((np.log10(e0[i]) - np.log10(e1[i])) / (np.log10(1/dofs0) - np.log10(1/dofs1))))

    if ph.config.RANK == 0:
        print(order)
        # print([float(_) for _ in e0])
        # print([float(_) for _ in e1])
    else:
        pass
