# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/_2d_boundary_section_config_1.py
"""
import sys

import numpy as np

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph

# ------------ parameters -------------------------------
alpha_1 = 75
alpha_2 = 15

d0l = 0.5
d0r = 0.5

d1l = 0.25
d1r = 0.25

d2l = 0.25
d2r = 0.25

l0 = 1
l1 = 1.5
l2 = 1.5

mesh_refinement_factor = 3

Re = 100

t_max = 10
steps_per_second = 100

polynomial_degree = 2

# -------- parser the geometry ------------------------

real_alpha_1 = 180 + alpha_1
real_alpha_2 = 360 - alpha_2

d0 = d0l + d0r
d1 = d1l + d1r
d2 = d2l + d2r

A = ph.geometries.Point2(0 - d0l, 0)
B = ph.geometries.Point2(0 + d0r, 0)
C = ph.geometries.Point2(0 - d0l, 1)
D = ph.geometries.Point2(0 + d0r, 1)

l0l = ph.geometries.StraightLine2(A, C)
l0r = ph.geometries.StraightLine2(B, D)

gamma1 = real_alpha_1 * 2 * np.pi / 360
beta = (real_alpha_1 - 90) * 2 * np.pi / 360
A = ph.geometries.Point2(d1l * np.cos(beta), d1l * np.sin(beta))
B = ph.geometries.Point2(A.x + np.cos(gamma1), A.y + np.sin(gamma1))
l1l = ph.geometries.StraightLine2(A, B)
beta = (real_alpha_1 + 90) * 2 * np.pi / 360
A = ph.geometries.Point2(d1r * np.cos(beta), d1r * np.sin(beta))
B = ph.geometries.Point2(A.x + np.cos(gamma1), A.y + np.sin(gamma1))
l1r = ph.geometries.StraightLine2(A, B)

gamma2 = real_alpha_2 * 2 * np.pi / 360
beta = (real_alpha_2 - 90) * 2 * np.pi / 360
A = ph.geometries.Point2(d2l * np.cos(beta), d2l * np.sin(beta))
B = ph.geometries.Point2(A.x + np.cos(gamma2), A.y + np.sin(gamma2))
l2l = ph.geometries.StraightLine2(A, B)
beta = (real_alpha_2 + 90) * 2 * np.pi / 360
A = ph.geometries.Point2(d2r * np.cos(beta), d2r * np.sin(beta))
B = ph.geometries.Point2(A.x + np.cos(gamma2), A.y + np.sin(gamma2))
l2r = ph.geometries.StraightLine2(A, B)

A = ph.geometries.line2_line2_intersection(l0l, l1l)
B = ph.geometries.line2_line2_intersection(l1r, l2l)
C = ph.geometries.line2_line2_intersection(l2r, l0r)

D = ph.geometries.Point2(0, l0)
M = ph.geometries.Point2(-d0l, l0)
N = ph.geometries.Point2(d0r, l0)

E = ph.geometries.Point2(l1 * np.cos(gamma1), l1 * np.sin(gamma1))
beta = (real_alpha_1 - 90) * 2 * np.pi / 360
G = ph.geometries.Point2(E.x + d1l * np.cos(beta), E.y + d1l * np.sin(beta))
beta = (real_alpha_1 + 90) * 2 * np.pi / 360
H = ph.geometries.Point2(E.x + d1r * np.cos(beta), E.y + d1r * np.sin(beta))

F = ph.geometries.Point2(l2 * np.cos(gamma2), l2 * np.sin(gamma2))
beta = (real_alpha_2 - 90) * 2 * np.pi / 360
R = ph.geometries.Point2(F.x + d2l * np.cos(beta), F.y + d2l * np.sin(beta))
beta = (real_alpha_2 + 90) * 2 * np.pi / 360
Q = ph.geometries.Point2(F.x + d2r * np.cos(beta), F.y + d2r * np.sin(beta))

outlet_1_vector = (G.x - A.x, G.y - A.y)
outlet_2_vector = (Q.x - C.x, Q.y - C.y)

DIS = np.sqrt(outlet_1_vector[0]**2 + outlet_1_vector[1]**2)
outlet_1_vector = (outlet_1_vector[0] / DIS, outlet_1_vector[1] / DIS)

DIS = np.sqrt(outlet_2_vector[0]**2 + outlet_2_vector[1]**2)
outlet_2_vector = (outlet_2_vector[0] / DIS, outlet_2_vector[1] / DIS)

# --------- abstract level ----------------------------------------------
ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

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

pde.bc.partition(r"\Gamma_{\perp}", r"\Gamma_P")
pde.bc.define_bc(
    {
        r"\Gamma_{\perp}": ph.trace(u),       # essential BC: norm velocity.
        r"\Gamma_P": ph.trace(ph.Hodge(P)),   # natural BC: total pressure.
    }
)

pde.bc.partition(r"\Gamma_I", r"\Gamma_L", r"\Gamma_R", r"\Gamma_W")

# pde.pr(vc=True)

wf = pde.test_with(
    [Out1, Out0, Out2],
    sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}']
)

wf = wf.derive.integration_by_parts('0-3')
wf = wf.derive.integration_by_parts('1-1')
wf = wf.derive.delete('1-2')   # since the natural BC for tangential velocity is 0 all-over the boundary

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

# wf.pr()

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
# wf.pr()

ph.space.finite(polynomial_degree)
mp = wf.mp()
nls = mp.nls()   # nonlinear system
# nls.pr()


# ------- implementation -------------------------------------------------

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
# msehtt.config(tgm)('meshpy', ts=1, points=points, max_volume=0.01)

type_dict = {
    0: 9,
    1: 9,
    2: 9,
    3: 9,
}

coo_dict = {
    'A': (A.x, A.y),
    'B': (B.x, B.y),
    'C': (C.x, C.y),
    'G': (G.x, G.y),
    'H': (H.x, H.y),
    'R': (R.x, R.y),
    'Q': (Q.x, Q.y),
    'M': (M.x, M.y),
    'N': (N.x, N.y),
    'D': (D.x, D.y),
}

map_dict = {
    # 0: ['M', 'A', 'C', 'N'],
    0: ['M', 'A', 'B', "D"],
    3: ['D', 'B', 'C', 'N'],
    1: ['A', 'G', 'H', 'B'],
    2: ['B', 'R', 'Q', 'C'],
    # 3: ['A', 'B', 'C']
}

msehtt.config(tgm)(
    {'indicator': 'tqr', 'args': (type_dict, coo_dict, map_dict)}, element_layout=mesh_refinement_factor)
# tgm.visualize(rank_wise_colored=False, quality=False)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)

# msehtt_mesh.visualize()

boundary_perp = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_{\perp}\right)"]
boundary_P = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_P\right)"]
boundary_I = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_I\right)"]
boundary_L = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_L\right)"]
boundary_R = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_R\right)"]
boundary_W = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_W\right)"]

msehtt.config(boundary_perp)(
    tgm,
    including={
        'type': 'boundary section',
        'partial elements': msehtt_mesh,
        'except on straight lines': ([G, H], [R, Q],)    # outward unit norm vector.
    }
)
# boundary_perp.visualize()

msehtt.config(boundary_P)(
    tgm,
    including={
        'type': 'boundary-section',
        'partial elements': msehtt_mesh,
        'on straight lines': ([G, H], [R, Q],)    # outward unit norm vector.
    }
)
# boundary_P.visualize()


def _x(t):
    return t


# noinspection PyUnusedLocal
def _y(t):
    return l0


curve = ph.geometries.Curve2(_x, _y, [-d0l, d0r])


msehtt.config(boundary_I)(
    tgm,
    including={
        'type': 'boundary-section',
        'partial elements': msehtt_mesh,
        # 'on straight segments': [(M, D), [D, N]]
        'on curves': (curve, )    # outward unit norm vector.
    }
)
# boundary_I.visualize()

msehtt.config(boundary_L)(
    tgm,
    including={
        'type': 'boundary-section',
        'partial elements': msehtt_mesh,
        'on straight segments': ([G, H], )    # outward unit norm vector.
    }
)
# boundary_L.visualize()

msehtt.config(boundary_R)(
    tgm,
    including={
        'type': 'boundary-section',
        'partial elements': msehtt_mesh,
        'on straight lines': ([R, Q], )    # outward unit norm vector.
    }
)
# boundary_R.visualize()

polygon_left = ph.geometries.Polygon2(M, A, G)
polygon_right = ph.geometries.Polygon2(N, C, Q)
# polygon_right.visualize()

msehtt.config(boundary_W)(
    tgm,
    including={
        'type': 'boundary-section',
        'partial elements': msehtt_mesh,
        # 'except on straight lines': ([R, Q], [G, H], [M, N])    # outward unit norm vector.
        "in polygons": [polygon_left, polygon_right]
    }
)
# boundary_W.visualize()

assert boundary_perp.composition._num_global_faces == 64
assert boundary_P.composition._num_global_faces == 16
assert boundary_I.composition._num_global_faces == 16
assert boundary_L.composition._num_global_faces == 8
assert boundary_R.composition._num_global_faces == 8
assert boundary_W.composition._num_global_faces == 32
