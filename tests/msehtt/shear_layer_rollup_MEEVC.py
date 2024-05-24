# -*- coding: utf-8 -*-
"""
mpiexec -n 4 python tests/msehtt/shear_layer_rollup_MEEVC.py
"""
from numpy import pi
import sys

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph
ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 3
K = 24

computation_time = 8
steps_per_second = 100
total_steps = int(steps_per_second * computation_time)

manifold = ph.manifold(2, periodic=True)
mesh = ph.mesh(manifold)

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')

w = Out0.make_form(r'\tilde{\omega}', 'outer-vorticity')
u = Out1.make_form(r'\tilde{u}', 'outer-velocity')
P = Out2.make_form(r'\tilde{P}', 'outer-pressure')

w_x_u = w.cross_product(u)

cd_P = P.codifferential()
cd_u = u.codifferential()
d_u = u.exterior_derivative()

du_dt = u.time_derivative()

expression_outer = [
    'du_dt + w_x_u - cd_P = 0',
    'w - cd_u = 0',
    'd_u = 0',
]

pde = ph.pde(expression_outer, locals())
pde.unknowns = [u, w, P]
# pde.pr(vc=True)

wf = pde.test_with([Out1, Out0, Out2], sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}'])
wf = wf.derive.integration_by_parts('0-2')
wf = wf.derive.integration_by_parts('1-1')

# wf.pr()

ts = ph.time_sequence()  # initialize a time sequence
dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')

td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
td.differentiate('0-0', 'k-1', 'k')
td.average('0-1', w, ['k-1', 'k'])
td.average('0-1', u, ['k-1', 'k'])
td.average('0-2', P, ['k-1/2', ])
td.average('1-0', w, ['k', ])
td.average('1-1', u, ['k', ])
td.average('2-0', u, ['k', ])
wf = td()

wf.unknowns = [
    u @ ts['k'],
    w @ ts['k'],
    P @ ts['k-1/2'],
    ]
wf = wf.derive.split(
    '0-0', 'f0',
    [u @ ts['k'], u @ ts['k-1']],
    ['+', '-'],
    factors=[1/dt, 1/dt],
)

wf = wf.derive.split(
    '0-2', 'f0',
    [
        (w @ ts['k-1']).cross_product(u @ ts['k-1']),
        (w @ ts['k-1']).cross_product(u @ ts['k']),
        (w @ ts['k']).cross_product(u @ ts['k-1']),
        (w @ ts['k']).cross_product(u @ ts['k'])
    ],
    ['+', '+', '+', '+'],
    factors=[1/4, 1/4, 1/4, 1/4],
)


wf = wf.derive.rearrange(
    {
        0: '0, 3, 4, 5, 6 = 1, 2',
    }
)

term = wf.terms['0-1']
term.add_extra_info(
    {'known-forms': w @ ts['k-1']}
)

term = wf.terms['0-2']
term.add_extra_info(
    {'known-forms': u @ ts['k-1']}
)
term = wf.terms['0-6']
term.add_extra_info(
    {'known-forms': [w @ ts['k-1'], u @ ts['k-1']]}
)

# wf.pr()

ph.space.finite(N)

# ap = wf.ap()
# ap.pr()
mp = wf.mp()
# mp.pr()
nls = mp.nls()
# nls.pr()

# ----- implementation ---------------------------------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)('crazy', element_layout=K, c=0, bounds=[[0., 2*pi], [0, 2*pi]], periodic=True)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')

ts.specify('constant', [0, computation_time, total_steps*2], 2)
initial_condition = ph.samples.InitialConditionShearLayerRollUp()

w = obj['w']
u = obj['u']
P = obj['P']

w.cf = initial_condition.vorticity
u.cf = initial_condition.velocity
w[0].reduce()
u[0].reduce()

# u[0].visualize.quick()
results_dir = './__phcache__/msehtt_shear_layer_rollup/'

msehtt_nls = obj['nls'].apply()
# msehtt_nls.pr()

for step in range(1, total_steps+1):
    system = msehtt_nls(k=step)
    system.customize.linear.set_local_dof(2, 0, 0, 0)
    system.solve(
        [u, w, P],
        atol=1e-6,
        inner_solver_scheme='lgmres',
        inner_solver_kwargs={'inner_m': 150, 'outer_k': 25, 'atol': 1e-6}
    )
    msehtt.info(rf"N={N}", system.solve.message)

    if step % 100 == 0:
        w[None].visualize.quick(saveto=results_dir + f'w{step}.png')
    else:
        pass
