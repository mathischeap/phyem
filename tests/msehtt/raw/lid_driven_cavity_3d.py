# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/lid_driven_cavity_3d.py
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph

ph.config.set_embedding_space_dim(3)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 2
K = 9
assert K >= 8, f"K={K} wrong, must be larger than 8."
element_layout = [
    [1, 2, 4] + [8] * (K-6) + [4, 2, 1],
    [8] * (K-3) + [4, 2, 1],
    [1, 2, 4] + [8] * (K-6) + [4, 2, 1],
]
t_max = 10
steps_per_second = 50
total_steps = steps_per_second * t_max
Re = 100

manifold = ph.manifold(3)
mesh = ph.mesh(manifold)

Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')
Out3 = ph.space.new('Lambda', 3, orientation='outer')

w = Out1.make_form(r'\tilde{\omega}', 'outer-vorticity')
u = Out2.make_form(r'\tilde{u}', 'outer-velocity')
P = Out3.make_form(r'\tilde{P}', 'outer-pressure')

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


wf = pde.test_with([Out2, Out1, Out3], sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}'])
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

# wf.pr()

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

# wf.pr()

ph.space.finite(N)

# ap = wf.ap()
# ap.pr()
mp = wf.mp()
# mp.pr()
nls = mp.nls()
# nls.pr()

# print(nls.pr())

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=element_layout,
    c=0,
    bounds=[[0, 1], [0, 1], [0, 1]],
    periodic=False
)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize()

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)

boundary_lid = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_l\right)"]

msehtt.config(boundary_lid)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'ounv': ([0, 1, 0],)    # outward unit norm vector.
    }
)
# boundary_lid.visualize()


ts.specify('constant', [0, t_max, total_steps*2], 2)

Rn2.value = 1 / (2 * Re)

w = obj['w']
u = obj['u']
P = obj['P']

M3 = msehtt.array('mass matrix', P)[0]  # manually make an array.
invM3 = M3.inv()

conditions = ph.samples.ConditionsLidDrivenCavity3()

u.cf = conditions.velocity_initial_condition
w.cf = conditions.vorticity_initial_condition

u['0'].reduce()     # t_0
w['0'].reduce()     # t_{0.5}

msehtt.info()

msehtt_nls = obj['nls'].apply()
# msehtt_nls.pr()

msehtt_nls.config(
    ('essential bc', 1),
    total_boundary,
    conditions.velocity_initial_condition,
    root_form=u
)  # essential bc

msehtt_nls.config(
    ['natural bc', 1],
    boundary_lid,
    conditions.velocity_boundary_condition_tangential,
    root_form=u
)

results_dir = './__phcache__/msehtt_3d_cavity/'

for step in range(1, total_steps+1):
    system = msehtt_nls(k=step)
    system.customize.linear.left_matmul_A_block(2, 0, invM3)
    # system.linear.spy(0)
    system.customize.linear.set_local_dof(2, 0, 0, 0)

    system.solve(
        [u, w, P],
        threshold=1e-8,
        # atol=1e-6,
        # maxiter=10,
        # inner_solver_scheme='lgmres',
        # inner_solver_kwargs={'inner_m': 500, 'outer_k': 25, 'atol': 1e-6, 'maxiter': 30}
    )
    msehtt.info(rf"N={N}", system.solve.message)
#
#     # ph.vtk(results_dir + rf'step_{step}', u[None], w[None], P[None])
#
    if step % (steps_per_second/5) == 0:
        ph.vtk(results_dir + rf'step1_{step}', u[None], w[None], P[None])
    else:
        pass
