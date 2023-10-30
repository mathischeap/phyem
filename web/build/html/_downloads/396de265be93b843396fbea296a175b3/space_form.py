# -*- coding: utf-8 -*-
r"""The .py file of Section Space & form.
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
