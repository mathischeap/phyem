# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/OTV.py

"""
import numpy as np

import phyem as ph

# --- config program -------------------------------------------------
ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

# --- setting up the problem -----------------------------------------
N = 2
K = 16

steps_per_second = 200

PER = int(steps_per_second / 5)

_rf_ = np.inf
_rm_ = np.inf
_c_ = 1

t = 1
total_steps = steps_per_second * t

manifold = ph.manifold(2, periodic=True)
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
Du = out2.make_form(r'\mathrm{d}u', 'd-u')
P = out2.make_form(r'P', 'pressure')

B = inn1.make_form(r'B', 'magnetic')
j = inn2.make_form(r'j', 'current_density')

# step 1: --------- NS: uP ----------------------------------------------------------------------------------
du_dt = u.time_derivative()
wXu = w.cross_product(u)
dw = Rf * w.exterior_derivative()
jXB = c * ph.Hodge(j.cross_product(B))
cd_P = P.codifferential()
du = u.exterior_derivative()
expression = [
    'du_dt + wXu + dw - jXB - cd_P = 0',
    'du = 0',
]
pde = ph.pde(expression, locals())
pde.unknowns = [u, P]

# pde.pr(vc=True)
wf = pde.test_with(
    [out1, out2],
    sym_repr=[r'v', r'q']
)
wf = wf.derive.integration_by_parts('0-4')
wf = wf.derive.switch_to_duality_pairing('0-3')

td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
td.differentiate('0-0', 'k-1', 'k')
td.average('0-1', u, 'k-1', 'k')
td.average('0-1', w, 'k-1/2')
td.average('0-2', w, 'k-1/2')
td.average('0-3', j, 'k-1/2')
td.average('0-3', B, 'k-1/2')
td.average('0-4', P, 'k-1/2')
td.average('1-0', u, 'k')

wf = td()
wf.unknowns = [
    u @ 'k',
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
        (w @ ts['k-1/2']).cross_product(u @ ts['k-1']),
        (w @ ts['k-1/2']).cross_product(u @ ts['k']),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)
term = wf.terms['0-2']
term.add_extra_info(
    {'known-forms': [w @ ts['k-1/2'], u @ ts['k-1']]}
)

term = wf.terms['0-3']
term.add_extra_info(
    {'known-forms': w @ ts['k-1/2']}
)

term = wf.terms['0-4']
term.add_extra_info(
    {'known-forms': w @ ts['k-1/2']}
)

term = wf.terms['0-5']
term.add_extra_info(
    {'known-forms': [j @ ts['k-1/2'], B @ ts['k-1/2']]}
)

wf = wf.derive.rearrange(
    {
        0: '0, 3, 6 = ',
        1: '0 = ',
    }
)

# wf.pr()
ph.space.finite(N)
mp = wf.mp()
ls_NS_uP = mp.ls()
# ls_NS_uP.pr()

# ----- MAXWELL ---------------------------------------------------------------------------------------

dB_dt = B.time_derivative()
Rm_dj = Rm * (B.exterior_derivative()).codifferential()
uXB = u.cross_product(B)
cd_uXB = uXB.codifferential()

expression_Bj = [
    'dB_dt + Rm_dj - cd_uXB = 0',
]
pde_Bj = ph.pde(expression_Bj, locals())
pde_Bj.unknowns = [B, ]

# pde_Bj.pr(vc=False)

wf_Bj = pde_Bj.test_with(
    [inn1, ],
    sym_repr=[r'b', ]
)
# # wf_Bj.pr()
wf_Bj = wf_Bj.derive.integration_by_parts('0-1')
wf_Bj = wf_Bj.derive.integration_by_parts('0-2')
# wf_Bj = wf_Bj.derive.integration_by_parts('1-0')
td = wf_Bj.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1/2', 'k', 'k+1/2')
td.differentiate('0-0', 'k-1/2', 'k+1/2')
td.average('0-1', B, ['k-1/2', 'k+1/2'])
td.average('0-2', u, 'k')
td.average('0-2', B, 'k+1/2', 'k-1/2')
# td.average('0-3', g, 'k')
# td.average('1-0', B, 'k+1/2')
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
    factors=[1/2, 1/2],
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
        # 1: '0 = '
    }
)
#
# wf_Bj.pr()
ph.space.finite(N)
mp = wf_Bj.mp()
ls_Bj = mp.ls()
# ls_Bj.pr()

# step 3: --------- NS: w ----------------------------------------------------------------------------------
dw_dt = w.time_derivative()
d_wXu = (w.cross_product(u)).codifferential()
ddw = Rf * (w.exterior_derivative()).codifferential()
d_jXB = c * ph.Hodge(j.cross_product(B)).codifferential()
expression = [
    'dw_dt + d_wXu + ddw - d_jXB = 0',
]
pde = ph.pde(expression, locals())
pde.unknowns = [w, ]

# pde.pr()

wf = pde.test_with(
    [out0, ],
    sym_repr=[r'w', ]
)
wf = wf.derive.integration_by_parts('0-1')
wf = wf.derive.integration_by_parts('0-2')
wf = wf.derive.integration_by_parts('0-3')
wf = wf.derive.switch_to_duality_pairing('0-3')

td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1/2', 'k', 'k+1/2')
td.differentiate('0-0', 'k-1/2', 'k+1/2')
td.average('0-1', u, 'k')
td.average('0-1', w, 'k-1/2', 'k+1/2')
td.average('0-2', w, 'k-1/2', 'k+1/2')
td.average('0-3', j, 'k')
td.average('0-3', B, 'k')

wf = td()
wf.unknowns = [
    w @ 'k+1/2',
]

wf = wf.derive.split(
    '0-0', 'f0',
    [w @ ts['k+1/2'], w @ ts['k-1/2']],
    ['+', '-'],
    factors=[1/Dt, 1/Dt],
)

wf = wf.derive.split(
    '0-2', 'f0',
    [
        (w @ ts['k-1/2']).cross_product(u @ ts['k']),
        (w @ ts['k+1/2']).cross_product(u @ ts['k']),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '0-4', 'f0',
    [
        (w @ ts['k-1/2']).d(),
        (w @ ts['k+1/2']).d(),
    ],
    ['+', '+'],
    factors=[Rf2, Rf2],
)

term = wf.terms['0-2']
term.add_extra_info(
    {'known-forms': [w @ ts['k-1/2'], u @ ts['k']]}
)
term = wf.terms['0-3']
term.add_extra_info(
    {'known-forms': u @ ts['k']}
)
term = wf.terms['0-6']
term.add_extra_info(
    {'known-forms': [j @ ts['k'], B @ ts['k']]}
)

wf = wf.derive.rearrange(
    {
        0: '0, 3, 5 = ',
    }
)

# wf.pr()
ph.space.finite(N)
mp = wf.mp()
ls_NS_w = mp.ls()
# ls_NS_w.pr()

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K,
    c=0,
    bounds=([0, 2 * np.pi], [0, 2 * np.pi]),
    periodic=True
)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')

ts.specify('constant', [0, t, total_steps*2], 2)

Rf.value = 1 / _rf_
Rf2.value = 1 / (2 * _rf_)
Rm.value = 1 / _rm_
Rm2.value = 1 / (2 * _rm_)
c.value = _c_

w = obj['w']
u = obj['u']
P = obj['P']
B = obj['B']
j = obj['j']

DU = obj['Du']

conditions = ph.samples.InitialConditionOrszagTangVortex()

u.cf = conditions.u
w.cf = conditions.omega

B.cf = conditions.B
j.cf = conditions.j

u['0'].reduce()       # t_0
B['0'].reduce()
B['1/2'].reduce()     # t_{0.5}
j['1/2'].reduce()     # t_{0.5}
w['1/2'].reduce()     # t_{0.5}

LS_NS_uP = obj['ls_NS_uP'].apply()
LS_Maxwell = obj['ls_Bj'].apply()
LS_NS_w = obj['ls_NS_w'].apply()


step = 1
linear_system = LS_NS_uP(k=step)
linear_system.customize.set_dof(-1, 0)
Axb = linear_system.assemble()
x, message0, info0 = Axb.solve('spsolve')
linear_system.x.update(x)

linear_system = LS_Maxwell(k=step)
Axb = linear_system.assemble()
x, message1, info1 = Axb.solve('spsolve')
linear_system.x.update(x)

tk = ts['k'](k=step)()
B[tk].cochain = B(tk).cochain
j[tk].cochain = B[tk].cochain.coboundary()

DU[tk].cochain = u[tk].cochain.coboundary()

du_norm = DU[tk].norm()

KE = u[tk].norm()
ME = B[tk].norm()
energy = 0.5 * KE + 0.5 * _c_ * ME

t_plus = ts['k+1/2'](k=step)()
j[t_plus].cochain = B[t_plus].cochain.coboundary()

linear_system = LS_NS_w(k=step)
Axb = linear_system.assemble()
x, message2, info2 = Axb.solve('spsolve')
linear_system.x.update(x)

np.testing.assert_almost_equal(du_norm, 0)
np.testing.assert_almost_equal(energy, 12.56625993741975)
