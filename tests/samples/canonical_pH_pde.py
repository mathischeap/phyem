# -*- coding: utf-8 -*-
"""
created at: 3/14/2023 12:13 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from src.config import set_embedding_space_dim
from src.manifold import manifold
from src.mesh import mesh
from src.pde import pde
import src.spaces.main as space
from src.form.operators import trace, Hodge


def pde_canonical_pH(n=3, p=3, periodic=False):
    """Generate pde representations of the canonical port-Hamiltonian systems."""
    set_embedding_space_dim(n)
    q = n + 1 - p

    if periodic:
        m = manifold(n, periodic=True)
    else:
        m = manifold(n)
    mesh(m)

    omega_p = space.new('Lambda', p, orientation='outer')
    omega_pm1 = space.new('Lambda', p-1, orientation='outer')

    ap = omega_p.make_form(r'\widehat{\alpha}^' + rf'{p}', 'a-p')
    bpm1 = omega_pm1.make_form(r'\widehat{\beta}^' + rf'{p-1}', 'b-pm1')

    sign1 = '+' if (-1) ** p == 1 else '-'
    sign2 = '+' if (-1) ** (p+1) == 1 else '-'

    pt_ap = ap.time_derivative()
    pt_bpm1 = bpm1.time_derivative()
    d_bpm1 = bpm1.exterior_derivative()
    ds_ap = ap.codifferential()

    outer_expression = [
        f'pt_ap = {sign1} d_bpm1',
        f'pt_bpm1 = {sign2} ds_ap'
    ]

    outer_pde = pde(outer_expression, locals())
    outer_pde._indi_dict = None  # clear this local expression
    outer_pde.unknowns = [ap, bpm1]

    # outer_pde.pr()

    if periodic:
        pass
    else:
        outer_pde.bc.partition(r"\Gamma_\alpha", r"\Gamma_\beta")
        alpha, beta = outer_pde.unknowns
        outer_pde.bc.define_bc(
            {
                r"\Gamma_\alpha": trace(Hodge(alpha)),
                r"\Gamma_\beta": trace(beta),
            }
        )

    inner_pde = None   # not implemented yet
    return outer_pde, inner_pde


if __name__ == '__main__':
    # python tests/samples/canonical_pH_pde.py

    pde_canonical_pH()
