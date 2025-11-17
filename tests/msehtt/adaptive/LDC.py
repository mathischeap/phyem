# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/adaptive/LDC.py
"""
import numpy as np

import phyem as ph

# --- config program -------------------------------------------------
ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

# --- setting up the problem -----------------------------------------
N = 2
t = 0.2
steps_per_second = 50

res_tol = 1e-5

real_tol = res_tol / steps_per_second

_rf_ = 10
_rm_ = 10
_s_ = 1
_c_ = _s_/_rm_

total_steps = int(steps_per_second * t)

manifold = ph.manifold(2, periodic=False)
mesh = ph.mesh(manifold)

out0 = ph.space.new('Lambda', 0, orientation='outer')
out1 = ph.space.new('Lambda', 1, orientation='outer')
out2 = ph.space.new('Lambda', 2, orientation='outer')

inn0 = ph.space.new('Lambda', 0, orientation='inner')
inn1 = ph.space.new('Lambda', 1, orientation='inner')
inn2 = ph.space.new('Lambda', 2, orientation='inner')

c = ph.constant_scalar(r'\mathsf{c}', "factor")
c4 = ph.constant_scalar(r'\dfrac{\mathsf{c}}{4}', r"coupling4")
Rf = ph.constant_scalar(r'\frac{1}{\mathrm{R_f}}', "Re")
Rf2 = ph.constant_scalar(r'\frac{1}{2\mathrm{R_f}}', "Re2")
Rm = ph.constant_scalar(r'\frac{1}{\mathrm{R_m}}', "Rm")
Rm2 = ph.constant_scalar(r'\frac{1}{2\mathrm{R_m}}', "Rm2")

ts = ph.time_sequence()  # initialize a time sequence
dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')

w = out0.make_form(r'\omega', 'vorticity')
u = out1.make_form(r'u', 'velocity')
P = out2.make_form(r'P', 'pressure')

B = inn1.make_form(r'B', 'magnetic')
j = inn2.make_form(r'j', 'current_density')

# --------- NS ----------------------------------------------------------------------------------

du_dt = u.time_derivative()
wXu = w.cross_product(u)
dw = Rf * w.exterior_derivative()
cdu = u.codifferential()
jXB = c * ph.Hodge(j.cross_product(B))
cd_P = P.codifferential()

du = u.exterior_derivative()

dB_dt = B.time_derivative()
Rm_dj = Rm * (B.exterior_derivative()).codifferential()
uXB = u.cross_product(B)
cd_uXB = uXB.codifferential()
dB = B.exterior_derivative()

expression = [
    'du_dt + wXu + dw - jXB - cd_P = 0',
    'w - cdu = 0',
    'du = 0',
    'dB_dt + Rm_dj - cd_uXB = 0',
    'j - dB = 0'
]
pde = ph.pde(expression, locals())
pde.unknowns = [u, w, P, B, j]
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

# pde.pr(vc=True)

wf = pde.test_with(
    [out1, out0, out2, inn1, inn2],
    sym_repr=[r'v', r'w', r'q', 'b', 'J']
)
wf = wf.derive.switch_to_duality_pairing('0-3')
wf = wf.derive.integration_by_parts('1-1')
wf = wf.derive.integration_by_parts('0-4')
wf = wf.derive.delete('0-5')
wf = wf.derive.integration_by_parts('3-2')
wf = wf.derive.integration_by_parts('3-1')
wf = wf.derive.delete('3-4')
wf = wf.derive.delete('3-2')
# wf.pr()

td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
td.differentiate('0-0', 'k-1', 'k')
td.average('0-1', u, 'k-1', 'k')
td.average('0-1', w, 'k-1', 'k')
td.average('0-2', w, 'k-1', 'k')
td.average('0-3', j, 'k-1', 'k')
td.average('0-3', B, 'k-1', 'k')
td.average('0-4', P, 'k-1/2')
td.average('1-0', w, 'k')
td.average('1-1', u, 'k')
td.average('1-2', u, 'k')
td.average('2-0', u, 'k')
td.differentiate('3-0', 'k-1', 'k')
td.average('3-1', B, 'k-1', 'k')
td.average('3-2', u, 'k-1', 'k')
td.average('3-2', B, 'k-1', 'k')
td.average('4-0', j, 'k')
td.average('4-1', B, 'k')
wf = td()
wf.unknowns = [
    u @ 'k',
    w @ 'k',
    P @ 'k-1/2',
    B @ 'k',
    j @ 'k',
]

wf = wf.derive.split(
    '0-3', 'f0',
    [
        (j @ ts['k-1']).cross_product(B @ ts['k-1']),
        (j @ ts['k-1']).cross_product(B @ ts['k']),
        (j @ ts['k']).cross_product(B @ ts['k-1']),
        (j @ ts['k']).cross_product(B @ ts['k'])
    ],
    ['+', '+', '+', '+'],
    factors=[c4, c4, c4, c4],
)

wf = wf.derive.split(
    '0-2', 'f0',
    [(w @ ts['k']).exterior_derivative(), (w @ ts['k-1']).exterior_derivative()],
    ['+', '+'],
    factors=[Rf2, Rf2],
)
wf = wf.derive.split(
    '0-1', 'f0',
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
    '0-0', 'f0',
    [u @ ts['k'], u @ ts['k-1']],
    ['+', '-'],
    factors=[1 / dt, 1 / dt],
)
wf = wf.derive.split(
    '3-2', 'f0',
    [
        (u @ ts['k-1']).cross_product(B @ ts['k-1']),
        (u @ ts['k-1']).cross_product(B @ ts['k']),
        (u @ ts['k']).cross_product(B @ ts['k-1']),
        (u @ ts['k']).cross_product(B @ ts['k'])
    ],
    ['+', '+', '+', '+'],
    factors=[1/4, 1/4, 1/4, 1/4],
)

wf = wf.derive.split(
    '3-1', 'f0',
    [(B @ ts['k']).exterior_derivative(), (B @ ts['k-1']).exterior_derivative()],
    ['+', '+'],
    factors=[Rm2, Rm2],
)


wf = wf.derive.split(
    '3-0', 'f0',
    [B @ ts['k'], B @ ts['k-1']],
    ['+', '-'],
    factors=[1 / dt, 1 / dt],
)

term = wf.terms['0-2']
term.add_extra_info(
    {'known-forms': [w @ ts['k-1'], u @ ts['k-1']]}
)
term = wf.terms['0-3']
term.add_extra_info(
    {'known-forms': w @ ts['k-1']}
)
term = wf.terms['0-4']
term.add_extra_info(
    {'known-forms': u @ ts['k-1']}
)

term = wf.terms['0-8']
term.add_extra_info(
    {'known-forms': [j @ ts['k-1'], B @ ts['k-1']]}
)
term = wf.terms['0-9']
term.add_extra_info(
    {'known-forms': j @ ts['k-1']}
)
term = wf.terms['0-10']
term.add_extra_info(
    {'known-forms': B @ ts['k-1']}
)

term = wf.terms['3-4']
term.add_extra_info(
    {'known-forms': [u @ ts['k-1'], B @ ts['k-1']]}
)
term = wf.terms['3-5']
term.add_extra_info(
    {'known-forms': u @ ts['k-1']}
)
term = wf.terms['3-6']
term.add_extra_info(
    {'known-forms': B @ ts['k-1']}
)

wf = wf.derive.rearrange(
    {
        0: '0, 3, 4, 5, 6, 9, 10, 11, 12 = ',
        1: '0, 1 = ',
        3: '0, 2, 5, 6, 7 = ',
    }
)

# wf.pr()
ph.space.finite(N)
mp = wf.mp()
nls_MHD = mp.nls()
# nls_MHD.pr()

# ------------- implementation ---------------------------------------------------

import os
data_dir = os.path.dirname(__file__)

msehtt, obj = ph.fem.apply('msehtt-a', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'meshpy',
    points=(
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],
    ),
    max_volume=0.05,
    ts=1,
    renumbering=True,
)
# tgm.visualize()
# print(tgm.elements.statistics)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)

boundary_lid = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_l\right)"]

msehtt.config(boundary_lid)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'ounv': ([0, 1],)    # outward unit norm vector.
    }
)

msehtt.initialize()

ts.specify('constant', [0, t, total_steps*2], 2)

Rf2.value = 1 / (2 * _rf_)
Rm2.value = 1 / (2 * _rm_)
c4.value = _c_ / 4

w = obj['w']
u = obj['u']
P = obj['P']
B = obj['B']
j = obj['j']

conditions = ph.samples.ConditionsLidDrivenCavity_2dMHD_1(lid_speed=1)

# msehtt_mesh.visualize(
#     labelsize=22,
#     ticksize=18,
#     xlim=[0, 1], ylim=[0, 1],
#     color='b',
#     title=False,
#     saveto=data_dir + f"/mesh_G0.png",
# )

u.cf = conditions.velocity_initial_condition
w.cf = conditions.vorticity_initial_condition

B.cf = conditions.B_initial_condition
j.cf = conditions.j_initial_condition

# total_boundary.visualize()
# boundary_lid.visualize()

u['0'].reduce()     # t_0
w['0'].reduce()     # t_{0.5}
j['0'].reduce()   # t_{0.5}
B['0'].reduce()   # t_{0.5}

B_energy_t0 = 0.5 * _c_ * B['0'].norm() ** 2
u_energy_t0 = 0.5 * u['0'].norm() ** 2
energy_t0 = B_energy_t0 + u_energy_t0

msehtt_nls_MHD = obj['nls_MHD']

# msehtt_nls_MHD.pr()

msehtt_nls_MHD.config(
    ('essential bc', 1),
    total_boundary, conditions.velocity_initial_condition, root_form=u
)  # essential bc
msehtt_nls_MHD.config(
    ['natural bc', 1],
    boundary_lid, conditions.velocity_boundary_condition_tangential, root_form=u
)


def refining_function(x, y):
    r""""""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


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
    u_residual :
    B_residual :

    """
    if k == 1:
        msehtt.renew(
            trf={
                'rff': refining_function,
                'rft': [0.75, ],
                'rcm': 'center'
            }
        )

    SYS = msehtt_nls_MHD.apply()
    system = SYS(k=k)
    system.customize.linear.set_local_dof(2, 0, 0, 0)
    system.solve(
        [u, w, P, B, j],
        atol=1e-7,
        # inner_solver_scheme='lgmres',
        # inner_solver_kwargs={'inner_m': 500, 'outer_k': 50, 'atol': 1e-6}
    )

    u_norm_residual = u.norm_residual()
    B_norm_residual = B.norm_residual()

    residual = max([u_norm_residual, B_norm_residual])

    T = ts['k'](k=k)()
    B_energy = 0.5 * _c_ * B[T].norm() ** 2
    u_energy = 0.5 * u[T].norm() ** 2

    energy = u_energy + B_energy

    if k % (5 * steps_per_second) == 0 or residual < real_tol:

        # ph.vtk(data_dir + rf'/step_{k}.vtu', u[None], w[None], P[None], B[None], j[None])
        # ph.rws(data_dir + rf"/solu_{k}.rws", u[None], w[None], P[None], B[None], j[None])

        if residual < real_tol:
            exit_code = 1   # stop iterations
        else:
            exit_code = 0   # continue iterations

    else:
        exit_code = 0       # continue iterations

    return (exit_code, [system.solve.message], T, u_energy, B_energy, energy,
            u_norm_residual, B_norm_residual)


filename = 'LDC_ph_cache'

iterator = ph.iterator(
    solver,
    [0, u_energy_t0, B_energy_t0, energy_t0, np.nan, np.nan],
    name=data_dir + r'/' + filename
)

iterator.cache(w, u, P, B, j, time=np.inf)

iterator.run(range(1, total_steps + 1), pbar=True)

if ph.config.RANK == 0:
    import pandas as pd
    data = pd.read_csv(data_dir + r'/' + filename + '.csv', index_col=0)
    results = data.to_numpy()[-1, :]
    ph.php(results)

ph.os.remove(data_dir + r'/' + filename + '.csv')
ph.os.remove(data_dir + r'/' + filename + '.png')
