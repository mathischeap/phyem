# -*- coding: utf-8 -*-
r"""
python tests/msepy/codifferential_test.py
"""
import numpy as np

import phyem as ph


def codifferential_test(n, k, orientation):
    r""""""
    print(f"codifferential tests for n={n}, k={k}, orientation={orientation}.\n")

    ph.config.set_embedding_space_dim(n)
    ph.config.set_high_accuracy(False)
    manifold = ph.manifold(n, periodic=True)
    mesh = ph.mesh(manifold)

    Lambda_n = ph.space.new('Lambda', k, orientation=orientation)
    Lambda_nm1 = ph.space.new('Lambda', k-1, orientation=orientation)

    phi = Lambda_n.make_form(rf'\varphi^{k}', 'potential')
    u = Lambda_nm1.make_form(rf'u^{k-1}', 'velocity')
    ds_phi = phi.codifferential()

    expression = [
        'u = ds_phi',
    ]
    pde = ph.pde(expression, locals())
    pde._indi_dict = None  # clear this local expression
    pde.unknowns = [u]

    wf = pde.test_with([Lambda_nm1], sym_repr=[rf'v^{n-1}'])
    wf = wf.derive.integration_by_parts('0-1')

    ph.space.finite(3)
    mp = wf.mp()
    ls = mp.ls()

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = obj['manifold']
    mesh = obj['mesh']
    msepy.config(manifold)('crazy_multi', c=0., periodic=True)
    msepy.config(mesh)(5)

    phi = msepy.base['forms'][r'potential']
    u = msepy.base['forms'][r'velocity']
    ls = obj['ls'].apply()

    # ----- 1d -------------
    def phi_func(t, x):
        """"""
        return np.sin(2 * np.pi * x) + t * 0

    phi_func_1d = ph.vc.scalar(phi_func)

    # ----- 2d -------------
    def phi_func(t, x, y):
        """"""
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + t * 0

    phi_func_2ds = ph.vc.scalar(phi_func)

    def phi_fx(t, x, y):
        """"""
        return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) + t * 0

    def phi_fy(t, x, y):
        """"""
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + t * 0

    phi_func_2dv = ph.vc.vector(phi_fx, phi_fy)

    # ----- 3d -------------
    def phi_func(t, x, y, z):
        """"""
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) + t * 0

    phi_func_3ds = ph.vc.scalar(phi_func)

    def phi_fx(t, x, y, z):
        """"""
        return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) + t * 0

    def phi_fy(t, x, y, z):
        """"""
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * z) + t * 0

    def phi_fz(t, x, y, z):
        """"""
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * z) + t * 0

    phi_func_3dv = ph.vc.vector(phi_fx, phi_fy, phi_fz)

    # -----------------------------
    if n == 1:
        phi_func = phi_func_1d

    elif n == 2:
        if k == 2:
            phi_func = phi_func_2ds
        elif k == 1:
            phi_func = phi_func_2dv
        else:
            raise Exception()

    elif n == 3:
        if k == 3:
            phi_func = phi_func_3ds
        elif k == 2:
            phi_func = phi_func_3dv
        elif k == 1:
            phi_func = phi_func_3dv
        else:
            raise Exception()

    else:
        raise Exception()

    phi.cf = phi_func
    u.cf = phi.cf.codifferential()
    phi[0].reduce()

    ls0 = ls(0)
    als = ls0.assemble()
    results = als.solve()
    ls0.x.update(results[0])

    ph.config.set_high_accuracy(True)

    return u[0].error()


if __name__ == '__main__':
    # python tests/msepy/codifferential_test.py 1 1 outer
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    orientation = sys.argv[3]
    error = codifferential_test(n, k, orientation=orientation)
    assert error < 0.005
    print('codifferential test >>> n:', n, ' k:', k, " orientation:", orientation)
