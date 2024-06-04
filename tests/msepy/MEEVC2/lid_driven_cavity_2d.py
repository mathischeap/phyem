# -*- coding: utf-8 -*-
r"""
$ python tests/msepy/MEEVC2/lid_driven_cavity_2d.py
"""
import sys
import numpy as np

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(True)

N = 2
K = 16
assert K >= 8, f"K={K} wrong, must be larger than 8."
element_layout = [
    [1, 2, 4, 8] + [16] * (K-8) + [8, 4, 2, 1],
    [1, 2, 4, 8] + [16] * (K-8) + [8, 4, 2, 1],
]
t_max = 100
steps_per_second = 100
steps = steps_per_second * t_max
_dt_ = 1 / steps_per_second
Re = 1000


manifold = ph.manifold(2)
mesh = ph.mesh(manifold)

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')

w = Out0.make_form(r'\tilde{\omega}', 'outer-vorticity')
u = Out1.make_form(r'\tilde{u}', 'outer-velocity')
P = Out2.make_form(r'\tilde{P}', 'outer-pressure')

Rn = ph.constant_scalar(r'\frac{1}{\mathrm{Re}}', "Re")
Rn2 = ph.constant_scalar(r'\frac{1}{2\mathrm{Re}}', "Re2")

w_x_u = w.cross_product(u)

d_w = Rn * w.exterior_derivative()
cd_P = P.codifferential()
cd_u = u.codifferential()
d_u = u.exterior_derivative()

du_dt = u.time_derivative()

expression_outer = [
    'du_dt + w_x_u + d_w - cd_P = 0',
    'w - cd_u = 0',
    'd_u = 0',
]

pde = ph.pde(expression_outer, locals())
pde.unknowns = [u, w, P]

pde.bc.define_bc(
    {
        r"\partial\mathcal{M}": ph.trace(u),          # essential

    }
)

pde.bc.partition(r"\Gamma_l", r"\Gamma_w")
pde.bc.define_bc(
    {
        r"\Gamma_l": ph.trace(ph.Hodge(u)),           # natural

    }
)
# pde.pr(vc=True)

wf = pde.test_with([Out1, Out0, Out2], sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}'])
wf = wf.derive.integration_by_parts('0-3')
wf = wf.derive.integration_by_parts('1-1')
wf = wf.derive.delete('0-4')
# wf.pr()

ts = ph.time_sequence()  # initialize a time sequence
dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')

td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
td.differentiate('0-0', 'k-1', 'k')
td.average('0-1', w, ['k-1', 'k'])
td.average('0-1', u, ['k-1', 'k'])
td.average('0-2', w, ['k-1', 'k'])
td.average('0-3', P, ['k-1/2', ])
td.average('1-0', w, ['k', ])
td.average('1-1', u, ['k', ])
td.average('1-2', u, ['k', ])
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

wf = wf.derive.split(
    '0-6', 'f0',
    [(w @ ts['k']).exterior_derivative(), (w @ ts['k-1']).exterior_derivative()],
    ['+', '+'],
    factors=[Rn2, Rn2],
)

wf.pr()

wf = wf.derive.rearrange(
    {
        0: '0, 3, 4, 5, 6, 8 = 1, 2, 7',
        1: '0, 1 = 2',
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
term = wf.terms['0-7']
term.add_extra_info(
    {'known-forms': [w @ ts['k-1'], u @ ts['k-1']]}
)

wf.pr()

ph.space.finite(N)

# ap = wf.ap()
# ap.pr()
mp = wf.mp()
mp.pr()
nls = mp.nls()
nls.pr()

# ------------- implementation ---------------------------------------------------
msepy, obj = ph.fem.apply('msepy', locals())
manifold = obj['manifold']
msepy.config(manifold)(
    'crazy', bounds=([0, 1], [0, 1])
)
boundary = msepy.base['manifolds'][r"\partial\mathcal{M}"]

Gamma_l = msepy.base['manifolds'][r"\Gamma_l"]

msepy.config(Gamma_l)(
    manifold, {
        0: [0, 0, 0, 1],
    }
)

imp_mesh = obj['mesh']
msepy.config(imp_mesh)(element_layout)

results_dir = './__phcache__/lid_driven_cavity_2d/'

import os
if os.path.isdir(results_dir):
    pass
else:
    os.mkdir(results_dir)

# imp_mesh.visualize(
#     xlim=[0, 1], ylim=[0, 1],
#     color='b',
#     linewidth=0.75,
#     title=False,
#     labelsize=18,
#     ticksize=18,
#     saveto=results_dir + f'/mesh.pdf',
# )

for msh in msepy.base['meshes']:
    msh = msepy.base['meshes'][msh]
    msh.visualize()

ts.specify('constant', [0, t_max, steps*2], 2)

msepy.info()

conditions = ph.samples.ConditionsLidDrivenCavity2()
Rn2.value = 1 / (2 * Re)

w = obj['w']
u = obj['u']
P = obj['P']

w.cf = conditions.vorticity_initial_condition
u.cf = conditions.velocity_initial_condition
w[0].reduce()
u[0].reduce()

nls = obj['nls'].apply()
# ts.pr(nls)
# nls.pr()

nls.bc.config(boundary)(conditions.velocity_initial_condition)             # essential
nls.bc.config(Gamma_l)(conditions.velocity_boundary_condition_tangential)  # natural

e6_save = True

for step in range(1, steps+1):

    s_nls = nls(k=step)
    # s_nls.pr()
    s_nls.customize.set_no_evaluation(-1)
    x, message, info = s_nls.solve(
        [u, w, P],
        atol=1e-7,
        inner_solver_scheme='spsolve',
    )  # nonlinear solver will update cochain automatically.

    norm_diff = u.norm(t=u.cochain.newest, diff_to_t=u.cochain.newest-_dt_)
    msepy.info(rf"N={N}", s_nls.solve.message, f"velocity norm residual: {norm_diff}")

    w[None].visualize(
        plot_type='contourf',
        levels=np.linspace(-10, 10, 50),
        saveto=results_dir + f'/omega_{step}.png',
        title=False,
    )
