# -*- coding: utf-8 -*-
r"""The .py file of Section Weak formulation.
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

wf = pde.test_with([Out2, Out1], sym_repr=['p', 'q'])
wf.pr()

wf = wf.derive.integration_by_parts('1-1')  # integrate the term '1-1' by parts
wf.pr()

wf = wf.derive.rearrange(
    {
        0: '0, 1 = ',    # do nothing to the first equations; can be removed
        1: '0, 1 = 2',   # rearrange the second equations
    }
)
wf.pr()

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
wf.pr()

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
wf.pr()

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
        1: '2, 0 = 3, 1, 4',
     }
)
wf.pr()

ph.space.finite(3)
wf.pr()
