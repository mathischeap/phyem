
.. testsetup:: *

    import phyem as ph
    ph.config.set_embedding_space_dim(2)
    ph.config.set_high_accuracy(True)
    ph.config.set_pr_cache(False)
    manifold = ph.manifold(2)
    mesh = ph.mesh(manifold)
    Out1 = ph.space.new('Lambda', 1, orientation='outer')
    Out2 = ph.space.new('Lambda', 2, orientation='outer')
    a = Out2.make_form(r'\tilde{\alpha}', 'variable1')
    b = Out1.make_form(r'\tilde{\beta}', 'variable2')
    da_dt = a.time_derivative()
    db_dt = b.time_derivative()
    cd_a = a.codifferential()
    d_b = b.exterior_derivative()
    expression = ['da_dt = + d_b', 'db_dt = - cd_a']
    pde = ph.pde(expression, locals())
    pde.unknowns = [a, b]
    pde.bc.partition(r"\Gamma_{\alpha}", r"\Gamma_{\beta}")
    pde.bc.define_bc(
        {
            r"\Gamma_{\alpha}": ph.trace(ph.Hodge(a)),   # natural
            r"\Gamma_{\beta}": ph.trace(b),              # essential
        }
    )
    wf = pde.test_with([Out2, Out1], sym_repr=['p', 'q'])
    wf = wf.derive.integration_by_parts('1-1')
    wf = wf.derive.rearrange(
        {
           0: '0, 1 = ',    # do nothing to the first equations; can be removed
           1: '0, 1 = 2',   # rearrange the second equations
        }
    )
    ts = ph.time_sequence()
    dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')
    td = wf.td
    td.set_time_sequence(ts)
    td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
    td.differentiate('0-0', 'k-1', 'k')
    td.average('0-1', b, ['k-1', 'k'])
    td.differentiate('1-0', 'k-1', 'k')
    td.average('1-1', a, ['k-1', 'k'])
    td.average('1-2', a, ['k-1/2'])
    wf = td()
    wf.unknowns = [
        a @ ts['k'],
        b @ ts['k']
    ]
    wf = wf.derive.split(
        '0-0', 'f0',
        [a @ ts['k'], a @ ts['k-1']],
        ['+', '-'],
        factors=[1/dt, 1/dt],
    )

    wf = wf.derive.split(
        '1-0', 'f0',
        [b @ ts['k'], b @ ts['k-1']],
        ['+', '-'],
        factors=[1/dt, 1/dt],
    )

    wf = wf.derive.split(
        '0-2', 'f0',
        [(b @ ts['k']).exterior_derivative(), (b @ ts['k-1']).exterior_derivative()],
        ['+', '+'],
        factors=[1/2, 1/2],
    )

    wf = wf.derive.split(
        '1-2', 'f0',
        [(a @ ts['k']), (a @ ts['k-1'])],
        ['+', '+'],
        factors=[1/2, 1/2],
    )
    wf = wf.derive.rearrange(
        {
            0: '0, 2 = 1, 3',
            1: '2, 0 = 3, 1, 4'
        }
    )
    ph.space.finite(3)

.. testcleanup::

    pass


.. _docs-ap:

===============
Algebraic proxy
===============


.. topic:: Estimated reading time

    ⏱️ 5 minutes


.. automodule:: phyem.src.wf.mp.main
    :undoc-members:

|

.. topic:: Python script of this section

    :download:`ap.py <py/ap.py>`

↩️  Back to :ref:`Documentation`.
