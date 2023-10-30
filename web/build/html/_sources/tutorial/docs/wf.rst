
.. testsetup:: *

    import __init__ as ph
    ph.config.set_embedding_space_dim(2)
    ph.config.set_high_accuracy(True)
    ph.config.set_pr_cache(True)
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
    _wf = pde.test_with([Out2, Out1], sym_repr=['{p}', '{q}'])
    _wf.pr(saveto='./source/tutorial/docs/images/docs_raw_wf.png')
    del _wf

.. testcleanup::

    pass


.. _docs-wf-and-discretization:

=================================
Weak formulation & discretization
=================================

.. topic:: Estimated reading time

    ⏱️ 20 minutes

.. automodule:: src.wf.main
    :undoc-members:

|

.. topic:: Python script of this section

    :download:`wf.py <py/wf.py>`

↩️  Back to :ref:`Documentation`.
