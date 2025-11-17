# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/dMHD_LDC.py
"""
import numpy as np

import phyem as ph

# --- config program --------------------------------------------------------------------
ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

# --- setting up the problem ------------------------------------------------------------
N = 4
K = 24

t = 100
steps_per_second = 500

_rf_ = 400
_rm_ = 400
_s_ = 1
_c_ = _s_/_rm_

total_steps = steps_per_second * t
element_layout = [
    [1, 2, 4] + [8] * (K-6) + [4, 2, 1],
    [1, 2, 4] + [8] * (K-6) + [4, 2, 1],
]

manifold = ph.manifold(2, periodic=False)
mesh = ph.mesh(manifold)

out0 = ph.space.new('Lambda', 0, orientation='outer')
out1 = ph.space.new('Lambda', 1, orientation='outer')
out2 = ph.space.new('Lambda', 2, orientation='outer')

inn0 = ph.space.new('Lambda', 0, orientation='inner')
inn1 = ph.space.new('Lambda', 1, orientation='inner')
inn2 = ph.space.new('Lambda', 2, orientation='inner')

c = ph.constant_scalar(r'\mathsf{c}', "factor")
Rf = ph.constant_scalar(r'\frac{1}{\mathrm{R_f}}', "Re")
Rf2 = ph.constant_scalar(r'\frac{1}{2\mathrm{R_f}}', "Re2")
Rm = ph.constant_scalar(r'\frac{1}{\mathrm{R_m}}', "Rm")
Rm2 = ph.constant_scalar(r'\frac{1}{2\mathrm{R_m}}', "Rm2")

ts = ph.time_sequence()  # initialize a time sequence
dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')
Dt = ts.make_time_interval('k-1/2', 'k+1/2', sym_repr=r'\varDelta t')

w = out0.make_form(r'\omega', 'vorticity')
u = out1.make_form(r'u', 'velocity')
P = out2.make_form(r'P', 'pressure')

B = inn1.make_form(r'B', 'magnetic')
j = inn2.make_form(r'j', 'current_density')
E = inn2.make_form(r'E', 'electronic')
# g = inn0.make_form(r'g', 'lm')

t0 = inn0.make_form(r't0', 'test0')
t1 = inn0.make_form(r't1', 'test1')

b0 = inn1.make_form(r'b0', 'test_b0')
b1 = inn1.make_form(r'b1', 'test_b1')

D_U = out2.make_form(r'du', 'd-velocity')

# --------- NS ----------------------------------------------------------------------------------

du_dt = u.time_derivative()
wXu = w.cross_product(u)
dw = Rf * w.exterior_derivative()
cdu = u.codifferential()
jXB = c * ph.Hodge(j.cross_product(B))
cd_P = P.codifferential()

du = u.exterior_derivative()

expression = [
    'du_dt + wXu + dw  - jXB - cd_P = 0',
    'w - cdu = 0',
    'du = 0',
]
pde = ph.pde(expression, locals())
pde.unknowns = [u, w, P]
pde.bc.define_bc(
    {
        r"\partial\mathcal{M}": ph.trace(u),          # essential: u-norm. All-zero

    }
)

pde.bc.partition(r"\Gamma_l", r"\Gamma_w")
pde.bc.define_bc(
    {
        r"\Gamma_l": ph.trace(ph.Hodge(u)),           # natural bc: u-tangential

    }
)

# pde.pr()
wf = pde.test_with(
    [out1, out0, out2],
    sym_repr=[r'v', r'w', r'q']
)
wf = wf.derive.switch_to_duality_pairing('0-3')
wf = wf.derive.integration_by_parts('1-1')
wf = wf.derive.integration_by_parts('0-4')
wf = wf.derive.delete('0-5')
# wf.pr()

td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
td.differentiate('0-0', 'k-1', 'k')
td.average('0-1', u, 'k-1', 'k')
td.average('0-1', w, 'k-1', 'k')
td.average('0-2', w, 'k-1', 'k')
td.average('0-3', j, 'k-1/2')
td.average('0-3', B, 'k-1/2')
td.average('0-4', P, 'k-1/2')
td.average('1-0', w, 'k')
td.average('1-1', u, 'k')
td.average('1-2', u, 'k')
td.average('2-0', u, 'k')
wf = td()
wf.unknowns = [
    u @ 'k',
    w @ 'k',
    P @ 'k-1/2',
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
    factors=[Rf2, Rf2],
)

term = wf.terms['0-3']
term.add_extra_info(
    {'known-forms': w @ ts['k-1']}
)

term = wf.terms['0-4']
term.add_extra_info(
    {'known-forms': u @ ts['k-1']}
)
term = wf.terms['0-2']
term.add_extra_info(
    {'known-forms': [w @ ts['k-1'], u @ ts['k-1']]}
)
term = wf.terms['0-8']
term.add_extra_info(
    {'known-forms': [j @ ts['k-1/2'], B @ ts['k-1/2']]}
)

wf = wf.derive.rearrange(
    {
        0: '0, 3, 4, 5, 6, 9 = ',
        1: '0, 1 = ',
    }
)

# wf.pr()
ph.space.finite(N)
mp = wf.mp()
nls_NS = mp.nls()
# nls_NS.pr()

# ----- MAXWELL ---------------------------------------------------------------------------------------

dB_dt = B.time_derivative()
Rm_dj = Rm * (B.exterior_derivative()).codifferential()
uXB = u.cross_product(B)
cd_uXB = uXB.codifferential()
# dg = g.exterior_derivative()
# cdB = B.codifferential()

expression_Bj = [
    'dB_dt + Rm_dj - cd_uXB = 0',
]
pde_Bj = ph.pde(expression_Bj, locals())
pde_Bj.unknowns = [B, ]
# pde_Bj.bc.partition(r"\Gamma_B", )
# pde_Bj.bc.define_bc(
#     {
#         r"\Gamma_B": ph.trace(ph.Hodge(B)),  # natural bc: u-tangential
#     }
# )

# pde_Bj.pr(vc=False)

wf_Bj = pde_Bj.test_with(
    [inn1, ],
    sym_repr=[r'b', ]
)
# wf_Bj.pr()
wf_Bj = wf_Bj.derive.integration_by_parts('0-1')
wf_Bj = wf_Bj.derive.integration_by_parts('0-3')
# wf_Bj = wf_Bj.derive.integration_by_parts('1-0')
wf_Bj = wf_Bj.derive.delete('0-2')
wf_Bj = wf_Bj.derive.delete('0-3')
td = wf_Bj.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1/2', 'k', 'k+1/2')
td.differentiate('0-0', 'k-1/2', 'k+1/2')
td.average('0-1', B, ['k-1/2', 'k+1/2'])
td.average('0-2', u, 'k')
td.average('0-2', B, 'k+1/2', 'k-1/2')
# td.average('0-3', g, 'k')
# td.average('1-0', B, 'k+1/2')
# td.average('1-1', B, 'k+1/2')
wf_Bj = td()
wf_Bj.unknowns = [
    B @ 'k+1/2',
    # g @ 'k',
]
wf_Bj = wf_Bj.derive.split(
    '0-0', 'f0',
    [B @ ts['k+1/2'], B @ ts['k-1/2']],
    ['+', '-'],
    factors=[1/Dt, 1/Dt],
)
wf_Bj = wf_Bj.derive.split(
    '0-2', 'f0',
    [
        (B @ 'k-1/2').exterior_derivative(),
        (B @ 'k+1/2').exterior_derivative(),
    ],
    ['+', '+'],
    factors=[Rm2, Rm2],
)
wf_Bj = wf_Bj.derive.split(
    '0-4', 'f0',
    [
        (u @ 'k').cross_product(B @ 'k+1/2'),
        (u @ 'k').cross_product(B @ 'k-1/2'),
    ],
    ['+', '+'],
    factors=[0.5, 0.5],
)

term = wf_Bj.terms['0-4']
term.add_extra_info(
    {'known-forms': u @ 'k'}
)

term = wf_Bj.terms['0-5']
term.add_extra_info(
    {'known-forms': [u @ 'k', B @ 'k-1/2']}
)

wf_Bj = wf_Bj.derive.rearrange(
    {
        0: '0, 3, 4 = 1, 2, 5',
        # 1: '0 = '
    }
)

# wf_Bj.pr()
ph.space.finite(N)
mp = wf_Bj.mp()
ls_Bj = mp.ls()
# ls_Bj.pr()


# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=element_layout,
    c=0,
    bounds=([0, 1], [0, 1]),
    periodic=False)

# tgm.visualize()

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize(
#     labelsize=22,
#     ticksize=18,
#     xlim=[0, 1], ylim=[0, 1],
#     color='b',
#     title=False,
#     saveto=data_dir + f"/mesh.pdf",
# )

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)

boundary_lid = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_l\right)"]
# boundary_B = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_B\right)"]

msehtt.config(boundary_lid)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'ounv': ([0, 1],)    # outward unit norm vector.
    }
)
# boundary_lid.visualize()

# msehtt.config(boundary_B)(
#     tgm,
#     including={
#         'type': 'boundary_section',
#         'partial elements': msehtt_mesh,
#         'ounv': ([0, 1], [0, -1])    # outward unit norm vector.
#     }
# )
# # boundary_B.visualize()


ts.specify('constant', [0, t, total_steps*2], 2)

Rf2.value = 1 / (2 * _rf_)
Rm2.value = 1 / (2 * _rm_)
c.value = _c_

w = obj['w']
u = obj['u']
P = obj['P']
B = obj['B']
j = obj['j']
# g = obj['g']

T0 = obj['t0']
T1 = obj['t1']
B0 = obj['b0']
B1 = obj['b1']

d_u = obj['D_U']


conditions = ph.samples.ConditionsLidDrivenCavity_2dMHD_1(lid_speed=1)

u.cf = conditions.velocity_initial_condition
w.cf = conditions.vorticity_initial_condition

B.cf = conditions.B_initial_condition
j.cf = conditions.j_initial_condition

u['0'].reduce()     # t_0
w['0'].reduce()     # t_{0.5}
j['1/2'].reduce()   # t_{0.5}
B['1/2'].reduce()   # t_{0.5}

B_energy_t0 = 0.5 * _c_ * B['1/2'].norm() ** 2
u_energy_t0 = 0.5 * u['0'].norm() ** 2
energy_t0 = B_energy_t0 + u_energy_t0

msehtt_ls_Maxwell = obj['ls_Bj'].apply()
msehtt_nls_NS = obj['nls_NS'].apply()

# # msehtt_nls_NS.pr()

msehtt_nls_NS.config(('essential bc', 1),
                     total_boundary, conditions.velocity_initial_condition, root_form=u)  # essential bc
msehtt_nls_NS.config(['natural bc', 1],
                     boundary_lid, conditions.velocity_boundary_condition_tangential, root_form=u)
# msehtt_ls_Maxwell.config(['natural bc', 1],
#                          boundary_B, conditions.B_initial_condition, root_form=B)


def solver(k):
    """

    Parameters
    ----------
    k

    Returns
    -------
    exit_code :
    message :
    t :
    u_energy :
    B_energy :
    energy :

    """
    system = msehtt_nls_NS(k=k)
    system.customize.linear.set_local_dof(2, 0, 0, 0)
    system.solve(
        [u, w, P],
        atol=1e-9,
    )

    linear_system = msehtt_ls_Maxwell(k=k)
    Axb = linear_system.assemble()
    x, message, info = Axb.solve('spsolve')
    linear_system.x.update(x)

    t_plus = ts['k+1/2'](k=k)()
    j[t_plus].cochain = B[t_plus].cochain.coboundary()

    T = ts['k'](k=k)()
    B[T].cochain = B(T).cochain
    B_energy = 0.5 * _c_ * B[T].norm() ** 2
    u_energy = 0.5 * u[T].norm() ** 2

    energy = u_energy + B_energy

    return 0, [system.solve.message, message], T, u_energy, B_energy, energy


iterator = ph.iterator(
    solver,
    [0, u_energy_t0, B_energy_t0, energy_t0],
    name='quantities'
)

test_results = iterator.test([1, ], show_info=False)

u_energy, B_energy, energy = test_results[0][-3:]

np.testing.assert_almost_equal(u_energy, 0.0003915, decimal=7)
np.testing.assert_almost_equal(B_energy, 0.0012501, decimal=7)
np.testing.assert_almost_equal(energy, 0.0016415, decimal=7)
