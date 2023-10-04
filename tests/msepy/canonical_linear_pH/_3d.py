# -*- coding: utf-8 -*-
r"""
Here we demonstrate how to use PHYEM to solve the canonical linear port Hamiltonian problem in three dimensions.

In :math:`\mathbb{R}^3`, a canonical form the linear port-Hamiltonian is written as

.. math::
    \begin{equation}\left\lbrace
    \begin{aligned}
        \partial_t \alpha^3 & = - \mathrm{d}\beta^2\\
        \partial_t \beta^2 & = \mathrm{d}^{\ast} \alpha^3
    \end{aligned}\right.
    \end{equation}

It is solved with manufactured solutions. The program is

.. autofunction:: tests.msepy.canonical_linear_pH._3d.canonical_linear_pH_3d_periodic_manufactured_test

It solves the canonical linear port Hamiltonian problem in a periodic unit cube, :math:`\Omega:=[0,1]^3`. The time
step is :math:`\Delta t = 0.01`.

--------
Examples
--------

.. testsetup:: *

    from _sdt import canonical_linear_pH_3d_periodic_manufactured_test

.. testcleanup::

    pass

If we solve the problem with mimetic spectral element of degree 1 on a mesh of :math:`3*3*3` elements for only
one time step, we can do

>>> errors1 = canonical_linear_pH_3d_periodic_manufactured_test(1, 3, 1)   # doctest: +ELLIPSIS
<BLANKLINE>

To check the :math:`L^2`-error of solution :math:`\alpha_h^3` (at :math:`t=0.01` since we only solve for one
time step), do

>>> errors1[0]  # doctest: +ELLIPSIS
5.2...

You may do :math:`ph` refinements to decrease the error or run more time-steps to check the error after longer
iterations.

"""

import sys

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph
from tests.msepy.canonical_linear_pH._3d_eigen_solution import Eigen1


def canonical_linear_pH_3d_periodic_manufactured_test(degree, K, num_steps):
    r"""

    Parameters
    ----------
    degree : int
        The degree of mimetic spectral elements.
    K : int
        We use :math:`K*K*K` uniform elements.
    num_steps : int
        How many steps we want to run? Each time step is :math:`\Delta t = 0.01`. Maximum ``num_steps`` is
        100.

    Returns
    -------
    a3_L2_error : float
        :math:`L^2`-error of solution :math:`\alpha_h^3` at the last time step.

    b2_L2_error : float
        :math:`L^2`-error of solution :math:`\beta_h^2` at the last time step.

    """

    samples = ph.samples

    periodic = True
    oph = samples.pde_canonical_pH(n=3, p=3, periodic=periodic)[0]
    a3, b2 = oph.unknowns
    # oph.pr()

    wf = oph.test_with([a3, b2], sym_repr=[r'v^3', r'u^2'])

    wf = wf.derive.integration_by_parts('1-1')

    td = wf.td
    td.set_time_sequence()  # initialize a time sequence

    td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
    td.differentiate('0-0', 'k-1', 'k')
    td.average('0-1', b2, ['k-1', 'k'])

    td.differentiate('1-0', 'k-1', 'k')
    td.average('1-1', a3, ['k-1', 'k'])
    dt = td.time_sequence.make_time_interval('k-1', 'k')

    wf = td()

    ts = td.ts

    # wf.pr()

    wf.unknowns = [
        a3 @ td.time_sequence['k'],
        b2 @ td.time_sequence['k'],
        ]

    wf = wf.derive.split(
        '0-0', 'f0',
        [a3 @ td.ts['k'], a3 @ td.ts['k-1']],
        ['+', '-'],
        factors=[1/dt, 1/dt],
    )

    wf = wf.derive.split(
        '0-2', 'f0',
        [ph.d(b2 @ td.ts['k-1']), ph.d(b2 @ td.ts['k'])],
        ['+', '+'],
        factors=[1/2, 1/2],
    )

    wf = wf.derive.split(
        '1-0', 'f0',
        [b2 @ td.ts['k'], b2 @ td.ts['k-1']],
        ['+', '-'],
        factors=[1/dt, 1/dt]
    )

    wf = wf.derive.split(
        '1-2', 'f0',
        [a3 @ td.ts['k-1'], a3 @ td.ts['k']],
        ['+', '+'],
        factors=[1/2, 1/2],
    )

    wf = wf.derive.rearrange(
        {
            0: '0, 3 = 2, 1',
            1: '3, 0 = 2, 1',
        }
    )

    ph.space.finite(degree)

    mp = wf.mp()
    mp.parse([
        a3 @ td.time_sequence['k-1'],
        b2 @ td.time_sequence['k-1']]
    )
    ls = mp.ls()

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = msepy.base['manifolds'][r"\mathcal{M}"]
    mesh = msepy.base['meshes'][r'\mathfrak{M}']

    msepy.config(manifold)(
        'crazy', c=0., bounds=[[0, 1], [0, 1], [0, 1]], periodic=True,
    )
    msepy.config(mesh)([K, K, K])

    a3 = obj['a3']
    b2 = obj['b2']

    # ls.pr()
    ls = obj['ls'].apply()
    # ls.pr()

    ts.specify('constant', [0, 1, 100], 1)

    eigen = Eigen1()
    a_scalar = ph.vc.scalar(eigen.p)
    b_vector = ph.vc.vector(eigen.u, eigen.v, eigen.w)

    a3.cf = a_scalar
    a3[0].reduce()
    # a3[0].visualize()

    b2.cf = b_vector
    b2[0].reduce()
    # b2[0].visualize()

    def solver(k):
        """
        Parameters
        ----------
        k

        Returns
        -------
        exit_code :
        message :
        t :
        a3_L2_error :
        b2_L2_error :
        """
        static_ls = ls(k=k)
        als = static_ls.assemble()
        x = als.solve()[0]
        static_ls.x.update(x)
        a3_L2_error = a3.error(None)
        b2_L2_error = b2.error(None)

        return 0, als.solve.message, a3.cochain.newest, a3_L2_error, b2_L2_error

    iterator = ph.iterator(solver, [0, a3.error(0), a3.error(0)], name='gallery_pH_test')

    iterator.run(range(1, num_steps+1))

    last_results = iterator.RDF.to_numpy()[-1, :]

    import os
    os.remove('gallery_pH_test.png')
    os.remove('gallery_pH_test.csv')

    return last_results[1], last_results[2]


if __name__ == '__main__':
    # python tests/msepy/canonical_linear_pH/_3d.py
    import doctest
    doctest.testmod()

    canonical_linear_pH_3d_periodic_manufactured_test(1, 3, 1)
