
.. testsetup:: *

    import phyem as ph
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

.. testcleanup::

    pass


.. _docs-pde:

===
PDE
===

.. topic:: Estimated reading time

    ⏱️ 15 minutes


.. automodule:: phyem.src.pde
    :undoc-members:


|

.. topic:: Python script of this section

    :download:`pde.py <py/pde.py>`

↩️  Back to :ref:`Documentation`.
