# -*- coding: utf-8 -*-
r"""The .py file of Section PDE.
"""

import sys
ph_path = ...  # customize the path to the dir that contains phyem.
sys.path.append(ph_path)

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

manifold = ph.manifold(2)
mesh = ph.mesh(manifold)

ph.space.set_mesh(mesh)
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')
ph.list_spaces()

a = Out2.make_form(r'\tilde{\alpha}', 'variable1')
b = Out1.make_form(r'\tilde{\beta}', 'variable2')
ph.list_forms()

da_dt = a.time_derivative()
db_dt = b.time_derivative()
cd_a = a.codifferential()
d_b = b.exterior_derivative()
ph.list_forms()

expression = [
    'da_dt = + d_b',
    'db_dt = - cd_a'
]
# interpreter = {
#     'da_dt': da_dt,
#     'd_b': d_b,
#     'db_dt': db_dt,
#     'cd_a': cd_a
# }
interpreter = locals()
pde = ph.pde(expression, interpreter)
pde.unknowns = [a, b]

pde.pr()
pde.pr(vc=True)

pde.bc.partition(r"\Gamma_{\alpha}", r"\Gamma_{\beta}")
pde.bc.define_bc(
   {
       r"\Gamma_{\alpha}": ph.trace(ph.Hodge(a)),   # natural boundary condition
       r"\Gamma_{\beta}": ph.trace(b),              # essential boundary condition
   }
)
pde.pr()
