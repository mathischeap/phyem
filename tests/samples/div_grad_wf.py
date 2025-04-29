# -*- coding: utf-8 -*-
r"""
"""
import __init__ as ph
from src.form.operators import trace, Hodge


def wf_div_grad(n=3, degree=2, orientation='outer', periodic=False):
    """Generate wf representations of the Poisson problem.

    """
    if orientation == 'outer':
        # f = - d d^{\ast} phi
        if periodic:
            return _outer_periodic_Poisson(n, degree)
        else:
            return _outer_Poisson(n, degree)
    elif orientation == 'inner':
        # f = - d d^{\ast} phi
        if periodic:
            raise NotImplementedError()
        else:
            return _inner_Poisson(n, degree)

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
    manifold = ph.manifold(n, periodic=True)

    mesh = ph.mesh(manifold)

    Lambda_n = ph.space.new('Lambda', n, orientation='outer')
    Lambda_nm1 = ph.space.new('Lambda', n-1, orientation='outer')

    phi = Lambda_n.make_form(rf'\varphi^{n}', 'potential')
    u = Lambda_nm1.make_form(rf'u^{n-1}', 'velocity')
    f = Lambda_n.make_form(rf'f^{n}', 'source')

    d_u = u.exterior_derivative()
    ds_phi = phi.codifferential()

    expression = [
        'u = - ds_phi',
        'f = - d_u',
    ]
    pde = ph.pde(expression, locals())
    pde._indi_dict = None  # clear this local expression
    pde.unknowns = [u, phi]
    # pde.pr()

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
    return ls, mp


def _outer_Poisson(n, degree):
    """
    f = - div grad phi

    In the mixed formulation, it is

    u = grad phi
    f = - div u

    Parameters
    ----------
    n

    Returns
    -------

    """
    ph.config.set_embedding_space_dim(n)
    manifold = ph.manifold(n, periodic=False)

    mesh = ph.mesh(manifold)

    Lambda_0 = ph.space.new('Lambda', 0, orientation='outer')
    Lambda_n = ph.space.new('Lambda', n, orientation='outer')
    Lambda_nm1 = ph.space.new('Lambda', n-1, orientation='outer')

    phi = Lambda_n.make_form(rf'\varphi^{n}', 'potential')
    u = Lambda_nm1.make_form(rf'u^{n-1}', 'velocity')
    f = Lambda_n.make_form(rf'f^{n}', 'source')

    p0 = Lambda_0.make_form(f"p^0", 'helper0')
    p1 = Lambda_nm1.make_form(f"p^1", 'helper1')

    d_u = u.exterior_derivative()
    ds_phi = phi.codifferential()

    expression = [
        'u = - ds_phi',
        'f = - d_u',
    ]
    pde = ph.pde(expression, locals())
    pde._indi_dict = None  # clear this local expression
    pde.unknowns = [u, phi]
    # pde.pr(vc=True)

    pde.bc.partition(r"\Gamma_\phi", r"\Gamma_u")
    pde.bc.define_bc(
        {
            r"\Gamma_u": trace(u),
            r"\Gamma_\phi": trace(Hodge(phi)),
        }
    )

    wf = pde.test_with([Lambda_nm1, Lambda_n], sym_repr=[rf'v^{n-1}', rf'q^{n}'])
    wf = wf.derive.integration_by_parts('0-1')

    wf = wf.derive.rearrange(
        {
            0: '0, 1 = 2',
            1: '1 = 0',
        }
    )

    # wf.pr()

    wf = wf.derive.switch_sign(1)
    # wf.pr()
    ph.space.finite(degree)
    mp = wf.mp()
    # mp.pr()
    ls = mp.ls()
    # ls.pr()
    return ls, mp


def _inner_Poisson(n, degree):
    """
    f = - div grad phi

    In the mixed formulation, it is

    u = grad phi
    f = - div u
    """
    ph.config.set_embedding_space_dim(n)
    manifold = ph.manifold(n, periodic=False)

    mesh = ph.mesh(manifold)

    Lambda_0 = ph.space.new('Lambda', 0, orientation='inner')
    Lambda_1 = ph.space.new('Lambda', 1, orientation='inner')

    phi = Lambda_0.make_form(rf'\varphi^{0}', 'potential')
    u = Lambda_1.make_form(r'u^{1}', 'velocity')
    f = Lambda_0.make_form(rf'f^{0}', 'source')

    ds_u = u.codifferential()
    d_phi = phi.exterior_derivative()

    expression = [
        'u = d_phi',
        'f = ds_u',
    ]
    pde = ph.pde(expression, locals())
    pde._indi_dict = None  # clear this local expression
    pde.unknowns = [u, phi]
    # pde.pr(vc=True)

    pde.bc.partition(r"\Gamma_\phi", r"\Gamma_u")
    pde.bc.define_bc(
        {
            r"\Gamma_u": trace(Hodge(u)),  # natural
            r"\Gamma_\phi": trace(phi),    # essential
        }
    )

    wf = pde.test_with([Lambda_1, Lambda_0], sym_repr=[rf'v^{1}', rf'q^{0}'])
    wf = wf.derive.integration_by_parts('1-1')

    wf = wf.derive.rearrange(
        {
            0: '0, 1 = ',
            1: '1 = 0, 2',
        }
    )

    # wf.pr()

    wf = wf.derive.switch_sign(1)
    # wf.pr()
    ph.space.finite(degree)
    mp = wf.mp()
    # mp.pr()
    ls = mp.ls()
    # ls.pr()
    return ls, mp
