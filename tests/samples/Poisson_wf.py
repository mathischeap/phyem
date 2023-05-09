# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 3:47 PM on 5/9/2023
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph


def wf_Poisson(n=3, degree=2, orientation='outer', periodic=False):
    """Generate wf representations of the Poisson problem.

    """
    if orientation == 'outer' and periodic:
        # f = - d d^{\ast} phi
        return _outer_periodic_Poisson(n, degree)
    else:
        raise NotImplementedError()


def _outer_periodic_Poisson(n, degree):
    """
    f = - d d^{\ast} phi

    In the mixed formulation, it is

    u = d^{\ast} phi
    f = - d u

    Parameters
    ----------
    n

    Returns
    -------

    """
    ph.config.set_embedding_space_dim(n)
    manifold = ph.manifold(n, is_periodic=True)

    mesh = ph.mesh(manifold)

    Lambda_n = ph.space.new('Lambda', n, orientation='outer')
    Lambda_nm1 = ph.space.new('Lambda', n-1, orientation='outer')

    phi = Lambda_n.make_form(rf'\varphi^{n}', 'potential')
    u = Lambda_nm1.make_form(rf'u^{n-1}', 'velocity')
    f = Lambda_n.make_form(rf'f^{n}', 'source')

    d_u = u.exterior_derivative()
    ds_phi = phi.codifferential()

    expression = [
        'u = ds_phi',
        'f = - d_u',
    ]
    pde = ph.pde(expression, locals())
    pde._indi_dict = None  # clear this local expression
    pde.unknowns = [u, phi]

    wf = pde.test_with([Lambda_nm1, Lambda_n], sym_repr=[rf'v^{n-1}', rf'q^{n}'])
    wf = wf.derive.integration_by_parts('0-1')

    wf = wf.derive.rearrange(
        {
            0: '0, 1 = ',
            1: '1 = 0',
        }
    )
    wf = wf.derive.switch_sign(1)
    ph.space.finite(degree)
    mp = wf.mp()
    ls = mp.ls()
    return ls


if __name__ == '__main__':
    # python 
    pass
