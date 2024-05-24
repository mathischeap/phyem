# -*- coding: utf-8 -*-
r"""
python tests/msepy/MEEVC2/flow_around_cylinder_free_slip_wall.py
"""

import sys

import numpy as np

ph_dir = '../'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(True)

N = 3

Re = 800
t_max = 5
steps_per_second = 800

steps = t_max * steps_per_second

t0 = 0

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

pde.bc.partition(r"\Gamma_{\perp}", r"\Gamma_P")
pde.bc.define_bc(
    {
        r"\Gamma_{\perp}": ph.trace(u),       # essential BC: norm velocity.
        r"\Gamma_P": ph.trace(ph.Hodge(P)),   # natural BC: total pressure.
    }
)

pde.bc.partition(r"\Gamma_{\parallel}", r"\Gamma_w")
pde.bc.define_bc(
    {
        r"\Gamma_w": w,       # manual BC
    }
)

pde.pr(vc=True)

wf = pde.test_with(
    [Out1, Out0, Out2],
    sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}']
)

wf = wf.derive.integration_by_parts('0-3')
wf = wf.derive.integration_by_parts('1-1')
wf = wf.derive.delete('1-2')   # since the natural BC for tangential velocity is 0 all-over the boundary

wf.pr()

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
td.average('0-4', P, ['k-1/2', ])
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
        (w @ ts['k']).cross_product(u @ ts['k']),
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
        0: '0, 3, 4, 5, 6, 8 = 1, 2, 7, 9',
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
mp = wf.mp()
nls = mp.nls()   # nonlinear system
nls.pr()

# ------------- implementation ---------------------------------------------------
msepy, obj = ph.fem.apply('msepy', locals())
manifold = obj['manifold']
msepy.config(manifold)(
    'cylinder_channel', r=0.05, dl=0.2, dr=2, h=0.41, hl=0.2
)
boundary_perp = msepy.base['manifolds'][r"\Gamma_{\perp}"]
boundary_P = msepy.base['manifolds'][r"\Gamma_P"]

msepy.config(boundary_perp)(
    manifold, {
        0: [1, 0, 1, 0],
        1: [0, 0, 1, 1],
        2: [0, 0, 1, 0],
        3: [1, 1, 0, 0],
        4: [1, 0, 0, 0],
        5: [1, 0, 0, 1],
        6: [0, 0, 1, 1],
        7: [0, 0, 0, 1],
    }
)

boundary_w = msepy.base['manifolds'][r"\Gamma_w"]
msepy.config(boundary_w)(
    manifold, {
        0: [0, 0, 1, 0],
        1: [0, 0, 1, 0],
        2: [0, 0, 1, 0],
        3: [0, 0, 0, 0],
        4: [0, 0, 0, 0],
        5: [0, 0, 0, 1],
        6: [0, 0, 0, 1],
        7: [0, 0, 0, 1],
    }
)

_mesh = obj['mesh']
msepy.config(_mesh)(3)

for msh in msepy.base['meshes']:
    msh = msepy.base['meshes'][msh]
    msh.visualize()

Rn2.value = 1 / (2 * Re)


# noinspection PyUnusedLocal
def zero_function(t, x, y):
    """"""
    return np.zeros_like(x)


w = obj['w']
u = obj['u']
P = obj['P']

init_velocity = ph.vc.vector(zero_function, zero_function)
init_vorticity = ph.vc.scalar(zero_function)


w.cf = init_vorticity
u.cf = init_velocity
w[0].reduce()
u[0].reduce()


ts.specify('constant', [0, t_max, steps*2], 2)


# noinspection PyUnusedLocal
def flux_in_outlet(t, y):
    """"""
    return (6 / 0.41**2) * np.sin(np.pi * t / 8) * (y+0.2) * (0.21 - y)


# noinspection PyUnusedLocal
def flux_wall(t, y):
    """"""
    return np.zeros_like(y)


# noinspection PyUnusedLocal
def bc_u(t, x, y):
    """"""
    return ph.tools.genpiecewise(
        [t, y],
        [x < -0.15, np.logical_and(x >= -0.15, x <= 1), x > 1],
        [flux_in_outlet, flux_wall, flux_in_outlet]
    )


bc_velocity = ph.vc.vector(bc_u, 0)
results_dir = './__phcache__/flow_around_cylinder_free_slip_wall/'

import os
if os.path.isdir(results_dir):
    pass
else:
    os.mkdir(results_dir)


nls = obj['nls'].apply()
nls.bc.config(boundary_perp)(bc_velocity)  # essential
# nls.bc.config(boundary_w)(init_vorticity)  # essential

bc_P = u.numeric.tsp.L2_energy()
nls.bc.config(boundary_P)(bc_P)  # natural


for step in range(1, steps+1):

    s_nls = nls(k=step)
    # s_nls.customize.set_no_evaluation(-1)
    s_nls.customize._apply_diagonal_essential_BC_to_linear_part(
        1,   # cause w is the unknown indexed 1 (second unknown)
        boundary_w,   # the boundary section
        init_vorticity,  # the boundary condition: always zero.
    )
    s_nls.solve([u, w, P])

    # # if step % 10 == 0:
    # u[None].visualize(
    #     saveto=[
    #         results_dir + f'ux_{int(step)}.png',
    #         results_dir + f'uy_{int(step)}.png',
    #     ]
    # )
    #
    # P[None].visualize(
    #     plot_type='contourf',
    #     num_levels=50,
    #     saveto=results_dir+f'P_{int(step)}.png'
    # )

    w[None].visualize(
        plot_type='contourf',
        levels=np.linspace(-15, 15, 50),
        saveto=results_dir+f'w_{int(step)}.png'
    )

    u_norm_residual = u.norm_residual()
    msepy.info(rf"N={N}", s_nls.solve.message, f"u residual: {u_norm_residual}")

    # if u_norm_residual < 1e-6:
    #     break
