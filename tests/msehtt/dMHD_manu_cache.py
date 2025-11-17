# -*- coding: utf-8 -*-
r"""
mpiexec -n 5 python tests/msehtt/dMHD_manu_cache.py
"""
import numpy as np

import phyem as ph


N = 2
K = 10
C = 0.

ph.config.set_embedding_space_dim(3)
ph.config.set_pr_cache(False)

steps_per_second = 100

_rf_ = 1
_rm_ = 1
_c_ = 1

import os
file_dir = os.path.dirname(__file__)
filename = f"ph_cache_test"

manifold = ph.manifold(3, periodic=False)
mesh = ph.mesh(manifold)

out1 = ph.space.new('Lambda', 1, orientation='outer')
out2 = ph.space.new('Lambda', 2, orientation='outer')
out3 = ph.space.new('Lambda', 3, orientation='outer')

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

w = out1.make_form(r'\omega', 'vorticity')
u = out2.make_form(r'u', 'velocity')
P = out3.make_form(r'P', 'pressure')
f = out2.make_form(r'f', 'body-force')

DU = out3.make_form(r'd_u', 'd-velocity')

B = inn1.make_form(r'B', 'magnetic')
j = inn2.make_form(r'j', 'current_density')
E = inn2.make_form(r'E', 'electronic')
m = inn2.make_form(r'm', 'electronic-source')

t0 = inn0.make_form(r't0', 'test0')
t1 = inn0.make_form(r't1', 'test1')
t2 = inn0.make_form(r't2', 'test2')
b0 = inn1.make_form(r'b0', 'test_b0')
b1 = inn1.make_form(r'b1', 'test_b1')
b2 = inn1.make_form(r'b2', 'test_b2')

# --------- NS ----------------------------------------------------------------------------------

du_dt = u.time_derivative()
wXu = w.cross_product(u)
dw = Rf * w.exterior_derivative()
cdu = u.codifferential()
jXB = c * ph.Hodge(j.cross_product(B))
cd_P = P.codifferential()

du = u.exterior_derivative()

expression = [
    'du_dt + wXu + dw  - jXB - cd_P = f',
    'w - cdu = 0',
    'du = 0',
]
pde = ph.pde(expression, locals())
pde.unknowns = [u, w, P]

pde.bc.partition(r"\Gamma_u", r"\Gamma_P")
pde.bc.define_bc(
    {
        r"\Gamma_u": ph.trace(u),       # essential BC: norm velocity.
        r"\Gamma_P": ph.trace(ph.Hodge(P)),   # natural BC: total pressure.
        # r"\partial\mathcal{M}": ph.trace(ph.Hodge(P)),   # natural BC: total pressure.
    }
)

pde.bc.partition(r"\Gamma_u_t", r"\Gamma_w")
pde.bc.define_bc(
    {
        r"\Gamma_u_t": ph.trace(ph.Hodge(u)),   # natural: u-tangential.
        r"\Gamma_w": ph.trace(w),   # essential: omega
    }
)

# pde.pr(vc=True)
wf = pde.test_with(
    [out2, out1, out3],
    sym_repr=[r'v', r'w', r'q']
)
wf = wf.derive.switch_to_duality_pairing('0-3')
wf = wf.derive.integration_by_parts('1-1')
wf = wf.derive.integration_by_parts('0-4')
# wf = wf.derive.delete('0-5')
# wf = wf.derive.delete('1-2')
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
td.average('0-5', P, 'k-1/2')
td.average('0-6', f, 'k-1/2')
td.average('1-0', w, 'k')
td.average('1-1', u, 'k')
td.average('1-2', u, 'k')
td.average('2-0', u, 'k')
wf = td()
# wf.pr()
#
wf.unknowns = [
    u @ 'k',
    w @ 'k',
    P @ 'k-1/2',
]
#
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

# wf.pr()
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
cd_m = m.codifferential()
uXB = u.cross_product(B)
cd_uXB = uXB.codifferential()

expression_Bj = [
    'dB_dt + Rm_dj - cd_uXB - cd_m = 0',
]
pde_Bj = ph.pde(expression_Bj, locals())
pde_Bj.unknowns = [B, ]

pde.bc.partition(r"\Gamma_B", r"\Gamma_E")
pde.bc.define_bc(
    {
        r"\Gamma_B": ph.trace(B),   # essential: B
    }
)

# pde_Bj.pr(vc=False)

wf_Bj = pde_Bj.test_with(
    [inn1, ],
    sym_repr=[r'b', ]
)
# wf_Bj.pr()
wf_Bj = wf_Bj.derive.integration_by_parts('0-1')
wf_Bj = wf_Bj.derive.integration_by_parts('0-3')
wf_Bj = wf_Bj.derive.integration_by_parts('0-5')
wf_Bj = wf_Bj.derive.delete('0-2')
wf_Bj = wf_Bj.derive.delete('0-3')
wf_Bj = wf_Bj.derive.delete('0-4')
td = wf_Bj.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1/2', 'k', 'k+1/2')
td.differentiate('0-0', 'k-1/2', 'k+1/2')
td.average('0-1', B, ['k-1/2', 'k+1/2'])
td.average('0-2', u, 'k')
td.average('0-2', B, 'k+1/2', 'k-1/2')
td.average('0-3', m, 'k')
wf_Bj = td()
wf_Bj.unknowns = [
    B @ 'k+1/2',
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
        0: '0, 3, 4 = ',
    }
)

# wf_Bj.pr()
ph.space.finite(N)
mp = wf_Bj.mp()
ls_Bj = mp.ls()
# ls_Bj.pr()
#
# ----- implementation ----------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K, c=C,
    bounds=[[0., 2*np.pi], [0, 2*np.pi], [0, 2*np.pi]],
    periodic=False
)
# tgm.visualize()

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize()

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)
# total_boundary.visualize()

boundary_u = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_u\right)"]
boundary_P = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_P\right)"]

msehtt.config(boundary_P)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'ounv': ([-1, 0, 0], [0, 1, 0], [0, 0, 1],)    # outward unit norm vector.
    }
)
# boundary_P.visualize()

msehtt.config(boundary_u)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'except ounv': ([-1, 0, 0], [0, 1, 0], [0, 0, 1],)    # outward unit norm vector.
    }
)
# boundary_u.visualize()

boundary_ut = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_u_t\right)"]
boundary_w = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_w\right)"]

msehtt.config(boundary_ut)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'ounv': ([1, 0, 0], [0, -1, 0], [0, 0, 1],)    # outward unit norm vector.
    }
)
# boundary_ut.visualize()

msehtt.config(boundary_w)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'except ounv': ([1, 0, 0], [0, -1, 0], [0, 0, 1],)    # outward unit norm vector.
    }
)
# boundary_w.visualize()
boundary_B = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_B\right)"]

msehtt.config(boundary_B)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'except ounv': ([1, 0, 0], [0, 1, 0], [0, 0, -1],)    # outward unit norm vector.
    }
)
# boundary_B.visualize()

ts.specify('constant', [0, 1, steps_per_second*2], 2)

Rf2.value = 1 / (2 * _rf_)
Rm2.value = 1 / (2 * _rm_)
c.value = _c_

w = obj['w']
u = obj['u']
P = obj['P']
f = obj['f']

B = obj['B']
j = obj['j']
m = obj['m']

d_u = obj['DU']

conditions = ph.samples.ManufacturedSolutionMHD3_0(c=_c_, Rm=_rm_, Rf=_rf_)

u.cf = conditions.u
w.cf = conditions.omega
P.cf = conditions.P
f.cf = conditions.f

B.cf = conditions.B
j.cf = conditions.j
m.cf = conditions.m

u['0'].reduce()     # t_0
w['0'].reduce()     # t_{0.5}
P['0'].reduce()   # t_{0.5}

j['1/2'].reduce()   # t_{0.5}
B['1/2'].reduce()   # t_{0.5}

T0 = obj['t0']
T1 = obj['t1']
T2 = obj['t2']
B0 = obj['b0']
B1 = obj['b1']
B2 = obj['b2']

# M3 = msehtt.array('mass matrix', P)[0]  # manually make an array.
# invM3 = M3.inv()

u_L2_error_t0 = u['0'].error()
j_L2_error_t0 = j['1/2'].error()
B_L2_error_t0 = B['1/2'].error()
w_L2_error_t0 = w['0'].error()
P_L2_error_t0 = P['0'].error()

msehtt_ls_Maxwell = obj['ls_Bj'].apply()
msehtt_nls_NS = obj['nls_NS'].apply()

# msehtt_ls_Maxwell.pr()
# msehtt_nls_NS.pr()

msehtt_ls_Maxwell.config(['essential bc', 1], boundary_B, conditions.B, root_form=B)

msehtt_nls_NS.config(['essential bc', 1], boundary_w, conditions.omega, root_form=w)
msehtt_nls_NS.config(('natural bc', 1), boundary_ut, conditions.u, root_form=u)

msehtt_nls_NS.config(['essential bc', 1], boundary_u, conditions.u, root_form=u)
msehtt_nls_NS.config(['natural bc', 1], boundary_P, conditions.P, root_form=P)


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
    u_L2_error :
    j_L2_error :
    B_L2_error :
    w_L2_error :
    P_L2_error :
    u_Hdiv_error :
    w_Hcurl_error :
    B_Hcurl_error :
    du_L2_norm :
    divB_L_infty_norm :
    divB_L2_norm :

    """
    t = ts['k'](k=k)()
    t_plus = ts['k+1/2'](k=k)()
    t_minus = ts['k-1/2'](k=k)()

    f[t_minus].reduce()
    m[t].reduce()

    atol = 1e-5

    system = msehtt_nls_NS(k=k)
    # system.customize.linear.left_matmul_A_block(2, 0, invM3)
    # system.customize.linear.set_local_dof(2, 0, 0, 0)
    system.solve(
        [u, w, P],
        # threshold=1e-6,
        atol=atol,
        inner_solver_scheme='lgmres',
        inner_solver_kwargs={'inner_m': 300, 'outer_k': 25, 'atol': atol}
    )

    linear_system = msehtt_ls_Maxwell(k=k)
    Axb = linear_system.assemble()
    x, message, info = Axb.solve('lgmres', x0=[B, ], inner_m=300, outer_k=25, atol=atol)
    linear_system.x.update(x)

    j[t_plus].cochain = B[t_plus].cochain.coboundary()

    u_L2_error = u[t].error()
    j_L2_error = j[t_plus].error()
    B_L2_error = B[t_plus].error()
    w_L2_error = w[t].error()
    P_L2_error = P[t_minus].error()

    u_Hdiv_error = u[t].error(error_type='H1')
    w_Hcurl_error = w[t].error(error_type='H1')
    B_Hcurl_error = B[t_plus].error(error_type='H1')

    d_u[t].cochain = u[t].cochain.coboundary()
    du_L2_norm = d_u[t].norm()

    c0, c1, c2 = B[t_plus].project.to('m3n3k0')
    T0[t_plus].cochain = c0
    T1[t_plus].cochain = c1
    T2[t_plus].cochain = c2

    B0[t_plus].cochain = T0[t_plus].cochain.coboundary()
    B1[t_plus].cochain = T1[t_plus].cochain.coboundary()
    B2[t_plus].cochain = T2[t_plus].cochain.coboundary()

    dB00 = B0.numeric.rws(t_plus, ddf=3, component_wise=True)
    dB11 = B1.numeric.rws(t_plus, ddf=3, component_wise=True)
    dB22 = B2.numeric.rws(t_plus, ddf=3, component_wise=True)

    if dB00 is not None:
        dB00 = dB00[0]
        dB11 = dB11[1]
        dB22 = dB22[2]
        dB = dB00 + dB11 + dB22
        divB_L_infty_norm = dB.maximum()[0]
        divB_L2_norm = dB.L2_norm()[0]
    else:
        divB_L_infty_norm = 0
        divB_L2_norm = 0

    return (0, [system.solve.message, message], t,
            u_L2_error, j_L2_error, B_L2_error, w_L2_error, P_L2_error,
            u_Hdiv_error, w_Hcurl_error, B_Hcurl_error, du_L2_norm, divB_L_infty_norm, divB_L2_norm
            )


iterator = ph.iterator(
    solver,
    [
        0, u_L2_error_t0, j_L2_error_t0, B_L2_error_t0, w_L2_error_t0, P_L2_error_t0,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ],
    name=file_dir + r'/' + filename
)

iterator.cache(u, w, P, B, j, time=np.inf)  # never make new cache file by setting time=inf.

iterator.run(range(1, 12), pbar=True)

if ph.config.RANK == 0:
    import pandas as pd
    data = pd.read_csv(file_dir + r'/' + filename + '.csv', index_col=0)
    u_L2_error_s11, j_L2_error_s11, B_L2_error_s11, w_L2_error_s11, P_L2_error_s11 = data.to_numpy()[-1, 1:6]

    np.testing.assert_almost_equal(u_L2_error_s11, 0.31800798641808725, decimal=5)
    np.testing.assert_almost_equal(j_L2_error_s11, 0.050362841967988625, decimal=7)
    np.testing.assert_almost_equal(B_L2_error_s11, 0.01653311748657386, decimal=7)
    np.testing.assert_almost_equal(w_L2_error_s11, 0.39391272881882844, decimal=5)
    np.testing.assert_almost_equal(P_L2_error_s11, 0.23328666453865143, decimal=5)

ph.os.remove(file_dir + r'/' + filename + '.csv')
ph.os.remove(file_dir + r'/' + filename + '.png')
