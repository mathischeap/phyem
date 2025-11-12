r"""
Normal dipole collision

mpiexec -n 4 python tests/msehtt/PNPNS/linearSchemeBcManu.py
"""

import sys

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph

import numpy as np

N = 2
K = 10
steps_per_second = 500

epsilon = 1

t_max = 1
total_steps = steps_per_second * t_max
_dt_ = t_max / total_steps

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

manifold = ph.manifold(2)
mesh = ph.mesh(manifold)
boundary_mesh = mesh.boundary()

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')

Inn0 = ph.space.new('Lambda', 0, orientation='inner')
Inn1 = ph.space.new('Lambda', 1, orientation='inner')
Inn2 = ph.space.new('Lambda', 2, orientation='inner')

e = ph.constant_scalar(r'\epsilon', "e")

w = Out0.make_form(r'\tilde{\omega}', 'vorticity')
u = Out1.make_form(r'\tilde{u}', 'velocity')
phi = Out2.make_form(r'\tilde{\phi}', 'modified-pressure')

Sp = Inn0.make_form(r'S', 'source-positive-ions')
Sn = Inn0.make_form(r'N', 'source-negative-ions')
Sf = Out1.make_form(r'F', 'source-velocity')

p = Inn0.make_form(r'p', 'positive-ions')
n = Inn0.make_form(r'n', 'negative-ions')
psi = Inn0.make_form(r'\psi', 'electronic')
mu = Inn0.make_form(r'\mu', 'mu')
nu = Inn0.make_form(r'\nu', 'nu')
delta = Inn0.make_form(r'\delta', 'delta')
theta = Inn0.make_form(r'\vartheta', 'theta')

tau = Inn1.make_form(r'\boldsymbol{\tau}', 'tau')
chi = Inn1.make_form(r'\boldsymbol{\chi}', 'chi')

pTau = Inn1.make_form(r'\widehat{p\boldsymbol{\tau}}', 'p-tau')
nChi = Inn1.make_form(r'\widehat{n\boldsymbol{\chi}}', 'n-chi')
dPsi = Inn1.make_form(r'\widehat{\mathrm{d}\psi}', 'd-psi')

dp_dt = p.time_derivative()
convective_p = u.convect(p)
p_tau = p.multi(tau, output_space=tau)
d_p_tau = p_tau.codifferential()

dn_dt = n.time_derivative()
convective_n = u.convect(n)
n_chi = n.multi(chi, output_space=chi)
d_n_chi = n_chi.codifferential()

d_mu = mu.d()
d_nu = nu.d()

delta_dt = delta.time_derivative()
p_delta_dt = p.multi(delta_dt, output_space=p)

theta_dt = theta.time_derivative()
n_theta_dt = p.multi(theta_dt, output_space=n)

laplace_psi = e * psi.d().codifferential()

du_dt = u.time_derivative()

wxn = w.x(u, form_space=u)

dw = w.d()

d_phi = phi.codifferential()

p_tau_Hodge = ph.Hodge(p_tau)
n_chi_Hodge = ph.Hodge(n_chi)

cd_u = u.codifferential()
du = u.d()
# ph.list_forms()

expression_outer = [
    'dp_dt + convective_p + d_p_tau = Sp',
    'dn_dt + convective_n + d_n_chi = Sn',
    'tau - d_mu = 0',
    'chi - d_nu = 0',
    'p_delta_dt - dp_dt = 0',
    'n_theta_dt - dn_dt = 0',
    'mu - delta - psi = 0',
    'nu - theta + psi = 0',
    'laplace_psi - p + n = 0',
    'du_dt + wxn + dw - d_phi + p_tau_Hodge + n_chi_Hodge = Sf',
    'w - cd_u = 0',
    'du = 0'
]

pde = ph.pde(expression_outer, locals())
pde.unknowns = [p, n, tau, chi, delta, theta, mu, nu, psi, u, w, phi]
pde.bc.define_bc(
    {
        r"\partial\mathcal{M}": ph.trace(ph.Hodge(pTau)),  # natural
    }
)
pde.bc.define_bc(
    {
        r"\partial\mathcal{M}": ph.trace(ph.Hodge(nChi)),  # natural
    }
)
pde.bc.define_bc(
    {
        r"\partial\mathcal{M}": ph.trace(ph.Hodge(dPsi)),  # natural
    }
)
pde.bc.define_bc(
    {
        r'\partial\mathcal{M}': ph.trace(u),                  # essential
    }
)
pde.bc.define_bc(
    {
        r'\partial\mathcal{M}': ph.trace(ph.Hodge(u)),                  # natural
    }
)

# pde.pr(vc=True)

wf = pde.test_with(
    [Inn0, Inn0, Inn1, Inn1, Inn0, Inn0, Inn0, Inn0, Inn0, Out1, Out0, Out2],
    sym_repr=["q", "m", "t", "x", "A", "B", "C", "D", r"\varphi", r"\tilde{v}", r"\tilde{w}", r"\tilde{Q}"],
)

wf = wf.derive.integration_by_parts('0-2')
tfP = wf.test_forms[0]
bt = ph.dp(ph.trace(ph.Hodge(pTau)), ph.trace(tfP))
wf = wf.derive.replace('0-3', bt, '+')

wf = wf.derive.integration_by_parts('1-2')
tfN = wf.test_forms[1]
bt = ph.dp(ph.trace(ph.Hodge(nChi)), ph.trace(tfN))
wf = wf.derive.replace('1-3', bt, '+')


wf = wf.derive.integration_by_parts('8-0')
tfS = wf.test_forms[8]
bt = ph.dp(ph.trace(ph.Hodge(dPsi)), ph.trace(tfS))
wf = wf.derive.replace('8-1', bt, '+')

wf = wf.derive.integration_by_parts('9-3')
wf = wf.derive.integration_by_parts('10-1')

wf = wf.derive.switch_to_duality_pairing('9-5')
wf = wf.derive.switch_to_duality_pairing('9-6')

wf = wf.derive.integration_by_parts('0-1', drop_bi_term=True)
wf = wf.derive.integration_by_parts('1-1', drop_bi_term=True)

tfA = wf.test_forms[4]
bt = ph.inner(delta_dt, p.multi(tfA, output_space=p))
wf = wf.derive.replace('4-0', bt, '+')

tfB = wf.test_forms[5]
bt = ph.inner(theta_dt, n.multi(tfB, output_space=p))
wf = wf.derive.replace('5-0', bt, '+')

wf = wf.derive.rearrange(
    {
        0: '0, 1, 2 = 4, 3',
        1: '0, 1, 2 = 4, 3',
        8: '0, 2, 3 = ',
        9: '0, 1, 2, 3, 5, 6 = 7, 4',
        10: '0, 1 =',
    }
)

wf = wf.derive.delete('9-7')

# wf.pr(patterns=False)

ts = ph.time_sequence()  # initialize a time sequence
dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')
td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
td.differentiate('0-0', 'k-1', 'k')
td.differentiate('1-0', 'k-1', 'k')
td.differentiate('4-0', 'k-1', 'k')
td.differentiate('5-0', 'k-1', 'k')
td.differentiate('4-1', 'k-1', 'k')
td.differentiate('5-1', 'k-1', 'k')
td.differentiate('9-0', 'k-1', 'k')
td.average('0-1', p, ['k-1', ])
td.average('0-1', u, ['k-1', 'k'])
td.average('0-2', p, ['k-1', ])
td.average('0-2', tau, ['k-1', 'k'])
td.average('0-3', Sp, ['k-1/2', ])
td.average('0-4', pTau, ['k-1/2', ])
td.average('1-1', n, ['k-1', ])
td.average('1-1', u, ['k-1', 'k'])
td.average('1-2', n, ['k-1', ])
td.average('1-2', chi, ['k-1', 'k'])
td.average('1-3', Sn, ['k-1/2', ])
td.average('1-4', nChi, ['k-1/2', ])
td.average('2-0', tau, ['k', ])
td.average('2-1', mu, ['k', ])
td.average('3-0', chi, ['k', ])
td.average('3-1', nu, ['k', ])
td.average('4-0', p, ['k-1', ])
td.average('5-0', n, ['k-1', ])
td.average('6-0', mu, ['k', ])
td.average('6-1', delta, ['k', ])
td.average('6-2', psi, ['k', ])
td.average('7-0', nu, ['k', ])
td.average('7-1', theta, ['k', ])
td.average('7-2', psi, ['k', ])
td.average('8-0', psi, ['k', ])
td.average('8-1', p, ['k', ])
td.average('8-2', n, ['k', ])
td.average('8-3', dPsi, ['k', ])
td.average('9-1', w, ['k-1', ])
td.average('9-1', u, ['k-1', 'k'])
td.average('9-2', w, ['k-1', 'k'])
td.average('9-3', phi, ['k-1/2', ])
td.average('9-4', p, ['k-1', ])
td.average('9-4', tau, ['k-1', 'k'])
td.average('9-5', n, ['k-1', ])
td.average('9-5', chi, ['k-1', 'k'])
td.average('9-6', Sf, ['k-1/2', ])
# td.average('9-7', phi, ['k-1/2', ])
td.average('10-0', w, ['k', ])
td.average('10-1', u, ['k', ])
td.average('10-2', u, ['k', ])
td.average('11-0', u, ['k', ])

wf = td()

wf.unknowns = [
    p @ ts['k'],
    n @ ts['k'],
    tau @ ts['k'],
    chi @ 'k',
    delta @ 'k',
    theta @ 'k',
    mu @ 'k',
    nu @ 'k',
    psi @ 'k',
    u @ 'k',
    w @ 'k',
    phi @ 'k-1/2'
]

wf = wf.derive.split(
    '9-5', 'f0',
    [
        (n @ ts['k-1']).multi(chi @ ts['k-1'], output_space=u),
        (n @ ts['k-1']).multi(chi @ ts['k'], output_space=u)
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '9-4', 'f0',
    [
        (p @ ts['k-1']).multi(tau @ ts['k-1'], output_space=u),
        (p @ ts['k-1']).multi(tau @ ts['k'], output_space=u)
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '9-2', 'f0',
    [
        (w @ ts['k']).exterior_derivative(),
        (w @ ts['k-1']).exterior_derivative(),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '9-1', 'f0',
    [
        (w @ ts['k-1']).cross_product(u @ ts['k-1']),
        (w @ ts['k-1']).cross_product(u @ ts['k']),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '9-0', 'f0',
    [u @ ts['k'], u @ ts['k-1']],
    ['+', '-'],
    factors=[1/dt, 1/dt],
)

wf = wf.derive.split(
    '5-1', 'f0',
    [n @ ts['k'], n @ ts['k-1']],
    ['+', '-'],
    factors=[1, 1],
)

wf = wf.derive.split(
    '5-0', 'f0',
    [theta @ ts['k'], theta @ ts['k-1']],
    ['+', '-'],
    factors=[1, 1],
)

# wf = wf.derive.split(
#     '5-1', 'f1',
#     [
#         (n @ ts['k-1']).multi(tfB, output_space=theta),
#         (n @ ts['k']).multi(tfB, output_space=theta),
#     ],
#     ['+', '+'],
#     factors=[0.5, 0.5],
# )
#
# wf = wf.derive.split(
#     '5-0', 'f1',
#     [
#         (n @ ts['k-1']).multi(tfB, output_space=theta),
#         (n @ ts['k']).multi(tfB, output_space=theta),
#     ],
#     ['+', '+'],
#     factors=[0.5, 0.5],
# )

wf = wf.derive.split(
    '4-1', 'f0',
    [p @ ts['k'], p @ ts['k-1']],
    ['+', '-'],
    factors=[1, 1],
)

wf = wf.derive.split(
    '4-0', 'f0',
    [delta @ ts['k'], delta @ ts['k-1']],
    ['+', '-'],
    factors=[1, 1],
)

# wf = wf.derive.split(
#     '4-1', 'f1',
#     [
#         (p @ ts['k-1']).multi(tfA, output_space=theta),
#         (p @ ts['k']).multi(tfA, output_space=theta),
#     ],
#     ['+', '+'],
#     factors=[0.5, 0.5],
# )
#
# wf = wf.derive.split(
#     '4-0', 'f1',
#     [
#         (p @ ts['k-1']).multi(tfA, output_space=theta),
#         (p @ ts['k']).multi(tfA, output_space=theta),
#     ],
#     ['+', '+'],
#     factors=[0.5, 0.5],
# )

wf = wf.derive.split(
    '1-2', 'f0',
    [
        (n @ ts['k-1']).multi(chi @ ts['k-1'], output_space=chi),
        (n @ ts['k-1']).multi(chi @ ts['k'], output_space=chi),
        # (n @ ts['k']).multi(chi @ ts['k-1'], output_space=chi),
        # (n @ ts['k']).multi(chi @ ts['k'], output_space=chi)
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '0-2', 'f0',
    [
        (p @ ts['k-1']).multi(tau @ ts['k-1'], output_space=tau),
        (p @ ts['k-1']).multi(tau @ ts['k'], output_space=tau),
        # (p @ ts['k']).multi(tau @ ts['k-1'], output_space=tau),
        # (p @ ts['k']).multi(tau @ ts['k'], output_space=tau)
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '1-1', 'f0',
    [
        (n @ ts['k-1']).multi(u @ ts['k-1'], output_space=u),
        (n @ ts['k-1']).multi(u @ ts['k'], output_space=u),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '0-1', 'f0',
    [
        (p @ ts['k-1']).multi(u @ ts['k-1'], output_space=u),
        (p @ ts['k-1']).multi(u @ ts['k'], output_space=u),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '1-0', 'f0',
    [n @ ts['k'], n @ ts['k-1']],
    ['+', '-'],
    factors=[1/dt, 1/dt],
)

wf = wf.derive.split(
    '0-0', 'f0',
    [p @ ts['k'], p @ ts['k-1']],
    ['+', '-'],
    factors=[1/dt, 1/dt],
)


wf = wf.derive.rearrange(
    {
        0: '0, 3, 5 =',
        1: '0, 3, 5 =',
        4: '0, 2 =',
        5: '0, 2 =',
        9: '0, 3, 4, 6, 8, 10 ='
    }
)

term = wf.terms['0-1']
term.add_extra_info(
    {'known-forms': p @ ts['k-1']}
)

term = wf.terms['0-2']
term.add_extra_info(
    {'known-forms': p @ ts['k-1']}
)
#
# term = wf.terms['0-3']
# term.add_extra_info(
#     {'known-forms': tau @ ts['k-1']}
# )
#
term = wf.terms['0-4']
term.add_extra_info(
    {'known-forms': [p @ ts['k-1'], u @ 'k-1']}
)

term = wf.terms['0-5']
term.add_extra_info(
    {'known-forms': [p @ ts['k-1'], tau @ 'k-1']}
)

term = wf.terms['1-1']
term.add_extra_info(
    {'known-forms': n @ ts['k-1']}
)

term = wf.terms['1-2']
term.add_extra_info(
    {'known-forms': n @ ts['k-1']}
)

# term = wf.terms['1-3']
# term.add_extra_info(
#     {'known-forms': chi @ ts['k-1']}
# )
#
term = wf.terms['1-4']
term.add_extra_info(
    {'known-forms': [n @ ts['k-1'], u @ 'k-1']}
)

term = wf.terms['1-5']
term.add_extra_info(
    {'known-forms': [n @ ts['k-1'], chi @ 'k-1']}
)

term = wf.terms['4-0']
term.add_extra_info(
    {'known-forms': p @ ts['k-1']}
)

# term = wf.terms['4-2']
# term.add_extra_info(
#     {'known-forms': delta @ ts['k-1']}
# )

term = wf.terms['4-2']
term.add_extra_info(
    {'known-forms': [delta @ ts['k-1'], p @ 'k-1']}
)

term = wf.terms['5-0']
term.add_extra_info(
    {'known-forms': n @ ts['k-1']}
)
#
# term = wf.terms['5-2']
# term.add_extra_info(
#     {'known-forms': theta @ ts['k-1']}
# )

term = wf.terms['5-2']
term.add_extra_info(
    {'known-forms': [theta @ ts['k-1'], n @ 'k-1']}
)

term = wf.terms['9-1']
term.add_extra_info(
    {'known-forms': w @ ts['k-1']}
)

term = wf.terms['9-4']
term.add_extra_info(
    {'known-forms': p @ ts['k-1']}
)

term = wf.terms['9-5']
term.add_extra_info(
    {'known-forms': n @ ts['k-1']}
)

term = wf.terms['9-7']
term.add_extra_info(
    {'known-forms': [w @ ts['k-1'], u @ ts['k-1']]}
)

term = wf.terms['9-9']
term.add_extra_info(
    {'known-forms': [p @ ts['k-1'], tau @ ts['k-1']]}
)

term = wf.terms['9-10']
term.add_extra_info(
    {'known-forms': [n @ ts['k-1'], chi @ ts['k-1']]}
)

ph.space.finite(N)
mp = wf.mp()
ls = mp.ls()

# ------- implementation --------------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K,
    c=0,
    bounds=([0, 2*np.pi], [0, 2*np.pi]),
    periodic=False
)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize()

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)
# total_boundary.visualize()

ts.specify('constant', [0, t_max, total_steps*2], 2)
e.value = epsilon

p = obj['p']
n = obj['n']
tau = obj['tau']
chi = obj['chi']
mu = obj['mu']
nu = obj['nu']
psi = obj['psi']
delta = obj['delta']
theta = obj['theta']
u = obj['u']
w = obj['w']
phi = obj['phi']

Sp = obj['Sp']
Sn = obj['Sn']
Sf = obj['Sf']

pTau = obj['pTau']
nChi = obj['nChi']
dPsi = obj['dPsi']

conditions = ph.samples.Manufactured_Solution_PNPNS_2D_PeriodicDomain1(msehtt_mesh)

w.cf = conditions.omega
u.cf = conditions.u
p.cf = conditions.p
n.cf = conditions.n
delta.cf = conditions.delta
theta.cf = conditions.theta
mu.cf = conditions.mu
nu.cf = conditions.nu
phi.cf = conditions.phi
tau.cf = conditions.tau
chi.cf = conditions.chi
psi.cf = conditions.psi

Sp.cf = conditions.source_p
Sn.cf = conditions.source_n
Sf.cf = conditions.source_f

w[0].reduce()
u[0].reduce()
p[0].reduce()
n[0].reduce()
tau[0].reduce()
chi[0].reduce()
delta[0].reduce()
theta[0].reduce()

LS = obj['ls'].apply()

LS.config(['natural bc', 1], total_boundary, conditions.pTau, root_form=pTau)    # natural bc
LS.config(['natural bc', 1], total_boundary, conditions.nChi, root_form=nChi)    # natural bc
LS.config(['natural bc', 1], total_boundary, conditions.psi.gradient, root_form=dPsi)    # natural bc
LS.config(['natural bc', 1], total_boundary, conditions.u, root_form=u)    # natural bc
LS.config(('essential bc', 1), total_boundary, conditions.u, root_form=u)    # essential bc

# LS.pr()


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
    L2_error_p :
    L2_error_n :
    L2_error_psi :
    L2_error_u :
    L2_error_omega :
    L2_error_phi :
    """
    t = ts['k'](k=k)()
    t_minus = ts['k-1/2'](k=k)()

    psi[t].reduce()
    Sp[t_minus].reduce()
    Sn[t_minus].reduce()
    Sf[t_minus].reduce()
    phi[t_minus].reduce()
    # phi[t_minus].visualize()

    linear_system = LS(k=k)
    linear_system.customize.set_local_dof(8, 0, 0, psi[t].cochain.of_local_dof(0, 0))
    linear_system.customize.set_local_dof(11, 0, 0, phi[t_minus].cochain.of_local_dof(0, 0))
    Axb = linear_system.assemble()
    x, message, info = Axb.solve('direct')
    # x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)
    linear_system.x.update(x)

    L2_error_p = p[t].error()
    L2_error_n = n[t].error()
    L2_error_psi = psi[t].error()
    L2_error_u = u[t].error()
    L2_error_omega = w[t].error()
    L2_error_phi = phi[t_minus].error()

    if k == 3:
        benchmark_Results = np.array([
            0.006, 0.020816704355997886, 0.025914864517428034, 0.011813869770750366, 0.06576854638135196,
            0.09344914719026799, 0.27316769885091513
        ])
        Results = np.array([
            t, L2_error_p, L2_error_n, L2_error_psi, L2_error_u, L2_error_omega, L2_error_phi
        ])
        np.testing.assert_array_almost_equal(benchmark_Results, Results)

    else:
        pass

    return 0, message, t, L2_error_p, L2_error_n, L2_error_psi, L2_error_u, L2_error_omega, L2_error_phi


iterator = ph.iterator(
    solver,
    [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    name=rf'/linearSchemeBcManuTest'
)

iterator.test([1, 2, 3], show_info=False)
