# -*- coding: utf-8 -*-
r"""A *phyem* implementation of the normal dipole collision test case in Section 5.3 of
`[MEEVC, Zhang et al., arXiv, <https://arxiv.org/abs/2307.08166>]`_.

By `Yi Zhang <https://mathischeap.com/>`_.

mpiexec -n 4 python tests/msehtt/normal_dipole_collision.py

"""
import sys

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 2
KDx = [1 for _ in range(5)] + [2 for _ in range(5)] + [3 for _ in range(15)]  # element distribution along x-axis
KDy = [3 for _ in range(5)] + [2 for _ in range(5)] + [1 for _ in range(15)]  # element distribution along y-axis

element_layout = [
    KDx + KDx[::-1],
    KDy + KDy[::-1],
]

t_max = 1
steps_per_second = 200
total_steps = t_max * steps_per_second
Re = 625

manifold = ph.manifold(2, periodic=False)
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
        r"\partial\mathcal{M}": ph.trace(u),   # essential
    }
)
# pde.pr(vc=True)

wf = pde.test_with([Out1, Out0, Out2], sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}'])
wf = wf.derive.integration_by_parts('0-3')
wf = wf.derive.integration_by_parts('1-1')
wf = wf.derive.delete('0-4')
wf = wf.derive.delete('1-2')
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
mp = wf.mp()
nls = mp.nls()   # nonlinear system
# nls.pr()

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)('crazy', element_layout=element_layout, c=0, bounds=[[-1., 1], [-1, 1]], periodic=False)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)

# for msh in msehtt.base['meshes']:
#     msh = msehtt.base['meshes'][msh]
#     msh.visualize()

ts.specify('constant', [0, t_max, total_steps*2], 2)

conditions = ph.samples.ConditionsNormalDipoleCollision2()
Rn2.value = 1 / (2 * Re)

w = obj['w']
u = obj['u']
P = obj['P']

w.cf = conditions.vorticity_initial_condition
u.cf = conditions.velocity_initial_condition
w[0].reduce()
u[0].reduce()

results_dir = './__phcache__/msehtt_normal_dipole_collision/'

# w[None].visualize.quick(
#     color_range=(-300, 300),
# )
msehtt.info()

msehtt_nls = obj['nls'].apply()

msehtt_nls.config(('essential bc', 1), total_boundary, conditions.velocity_boundary_condition, root_form=u)
# essential bc

for step in range(1, total_steps+1):
    system = msehtt_nls(k=step)
    system.customize.linear.set_local_dof(2, 0, 0, 0)
    system.solve(
        [u, w, P],
        # atol=1e-6,
        # inner_solver_scheme='lgmres',
        # inner_solver_kwargs={'inner_m': 500, 'outer_k': 50, 'atol': 1e-6}
    )
    msehtt.info(rf"N={N}", system.solve.message)

    # if step % 100 == 0:
    w[None].visualize.quick(saveto=results_dir + f'w{step}.png', color_range=(-300, 300))
    # else:
    #     pass
