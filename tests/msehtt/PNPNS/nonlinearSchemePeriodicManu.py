r"""
Normal dipole collision

mpiexec -n 4 python tests/msehtt/PNPNS/nonlinearSchemePeriodicManu.py
"""

import phyem as ph

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

manifold = ph.manifold(2, periodic=True)
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

delta = Inn0.make_form(r'\delta', 'delta')
theta = Inn0.make_form(r'\vartheta', 'theta')

d_delta = delta.d()
d_theta = theta.d()

tau = Inn1.make_form(r'\boldsymbol{\tau}', 'tau')
chi = Inn1.make_form(r'\boldsymbol{\chi}', 'chi')

dp_dt = p.time_derivative()
convective_p = u.convect(p)
p_tau = p.multi(tau, output_space=tau)
d_p_tau = p_tau.codifferential()

dn_dt = n.time_derivative()
convective_n = u.convect(n)
n_chi = n.multi(chi, output_space=chi)
d_n_chi = n_chi.codifferential()

delta_dt = delta.time_derivative()
p_delta_dt = p.multi(delta_dt, output_space=p)

theta_dt = theta.time_derivative()
n_theta_dt = p.multi(theta_dt, output_space=n)

psi = Inn0.make_form(r'\psi', 'electronic')
d_psi = psi.d()
laplace_psi = e * d_psi.codifferential()

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
    'p_delta_dt - dp_dt = 0',
    'n_theta_dt - dn_dt = 0',
    'tau - d_delta - d_psi = 0',
    'chi - d_theta + d_psi = 0',
    'laplace_psi - p + n = 0',
    'du_dt + wxn + dw - d_phi + p_tau_Hodge + n_chi_Hodge = Sf',
    'w - cd_u = 0',
    'du = 0'
]

pde = ph.pde(expression_outer, locals())
pde.unknowns = [p, n, delta, theta, tau, chi, psi, u, w, phi]
# pde.pr(vc=True)

wf = pde.test_with(
    [Inn0, Inn0, Inn0, Inn0, Inn1, Inn1, Inn0, Out1, Out0, Out2],
    sym_repr=["q", "m", "A", "B", "C", "D", r"\varphi", r"\tilde{v}", r"\tilde{w}", r"\tilde{Q}"],
)

wf = wf.derive.integration_by_parts('0-2')
wf = wf.derive.integration_by_parts('1-2')
wf = wf.derive.integration_by_parts('6-0')
wf = wf.derive.integration_by_parts('7-3')
wf = wf.derive.integration_by_parts('8-1')

wf = wf.derive.switch_to_duality_pairing('7-4')
wf = wf.derive.switch_to_duality_pairing('7-5')

wf = wf.derive.integration_by_parts('0-1', drop_bi_term=True)
wf = wf.derive.integration_by_parts('1-1', drop_bi_term=True)

tfA = wf.test_forms[2]
bt = ph.inner(delta_dt, p.multi(tfA, output_space=p))
wf = wf.derive.replace('2-0', bt, '+')

tfB = wf.test_forms[3]
bt = ph.inner(theta_dt, n.multi(tfB, output_space=p))
wf = wf.derive.replace('3-0', bt, '+')

# wf.pr(patterns=False)

ts = ph.time_sequence()  # initialize a time sequence
dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')
td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
td.differentiate('0-0', 'k-1', 'k')
td.differentiate('1-0', 'k-1', 'k')
td.differentiate('2-0', 'k-1', 'k')
td.differentiate('3-0', 'k-1', 'k')
td.differentiate('2-1', 'k-1', 'k')
td.differentiate('3-1', 'k-1', 'k')
td.differentiate('7-0', 'k-1', 'k')
td.average('0-1', p, ['k-1', ])
td.average('0-1', u, ['k-1', 'k'])
td.average('0-2', p, ['k-1', 'k'])
td.average('0-2', tau, ['k-1', 'k'])
td.average('0-3', Sp, ['k-1/2', ])
td.average('1-1', n, ['k-1', ])
td.average('1-1', u, ['k-1', 'k'])
td.average('1-2', n, ['k-1', 'k'])
td.average('1-2', chi, ['k-1', 'k'])
td.average('1-3', Sn, ['k-1/2', ])
td.average('2-0', p, ['k-1', 'k'])
td.average('3-0', n, ['k-1', 'k'])
td.average('4-0', tau, ['k', ])
td.average('4-1', delta, ['k', ])
td.average('4-2', psi, ['k', ])
td.average('5-0', chi, ['k', ])
td.average('5-1', theta, ['k', ])
td.average('5-2', psi, ['k', ])
td.average('6-0', psi, ['k', ])
td.average('6-1', p, ['k', ])
td.average('6-2', n, ['k', ])
td.average('7-1', w, ['k-1', ])
td.average('7-1', u, ['k-1', 'k'])
td.average('7-2', w, ['k-1', 'k'])
td.average('7-3', phi, ['k-1/2', ])
td.average('7-4', p, ['k-1', ])
td.average('7-4', tau, ['k-1', 'k'])
td.average('7-5', n, ['k-1', ])
td.average('7-5', chi, ['k-1', 'k'])
td.average('7-6', Sf, ['k-1/2', ])
td.average('8-0', w, ['k', ])
td.average('8-1', u, ['k', ])
td.average('9-0', u, ['k', ])

wf = td()

wf.unknowns = [
    p @ 'k',
    n @ 'k',
    delta @ 'k',
    theta @ 'k',
    tau @ 'k',
    chi @ 'k',
    psi @ 'k',
    u @ 'k',
    w @ 'k',
    phi @ 'k-1/2'
]

# wf.pr(patterns=False)

wf = wf.derive.split(
    '7-5', 'f0',
    [
        (n @ ts['k-1']).multi(chi @ ts['k-1'], output_space=u),
        (n @ ts['k-1']).multi(chi @ ts['k'], output_space=u)
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '7-4', 'f0',
    [
        (p @ ts['k-1']).multi(tau @ ts['k-1'], output_space=u),
        (p @ ts['k-1']).multi(tau @ ts['k'], output_space=u)
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '7-2', 'f0',
    [
        (w @ ts['k']).exterior_derivative(),
        (w @ ts['k-1']).exterior_derivative(),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '7-1', 'f0',
    [
        (w @ ts['k-1']).cross_product(u @ ts['k-1']),
        (w @ ts['k-1']).cross_product(u @ ts['k']),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '7-0', 'f0',
    [u @ ts['k'], u @ ts['k-1']],
    ['+', '-'],
    factors=[1/dt, 1/dt],
)

wf = wf.derive.split(
    '3-1', 'f0',
    [n @ ts['k'], n @ ts['k-1']],
    ['+', '-'],
    factors=[1, 1],
)

wf = wf.derive.split(
    '3-0', 'f0',
    [theta @ ts['k'], theta @ ts['k-1']],
    ['+', '-'],
    factors=[1, 1],
)

wf = wf.derive.split(
    '3-1', 'f1',
    [
        (n @ ts['k-1']).multi(tfB, output_space=theta),
        (n @ ts['k']).multi(tfB, output_space=theta),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '3-0', 'f1',
    [
        (n @ ts['k-1']).multi(tfB, output_space=theta),
        (n @ ts['k']).multi(tfB, output_space=theta),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '2-1', 'f0',
    [p @ ts['k'], p @ ts['k-1']],
    ['+', '-'],
    factors=[1, 1],
)

wf = wf.derive.split(
    '2-0', 'f0',
    [delta @ ts['k'], delta @ ts['k-1']],
    ['+', '-'],
    factors=[1, 1],
)

wf = wf.derive.split(
    '2-1', 'f1',
    [
        (p @ ts['k-1']).multi(tfA, output_space=theta),
        (p @ ts['k']).multi(tfA, output_space=theta),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '2-0', 'f1',
    [
        (p @ ts['k-1']).multi(tfA, output_space=theta),
        (p @ ts['k']).multi(tfA, output_space=theta),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '1-2', 'f0',
    [
        (n @ ts['k-1']).multi(chi @ ts['k-1'], output_space=chi),
        (n @ ts['k-1']).multi(chi @ ts['k'], output_space=chi),
        (n @ ts['k']).multi(chi @ ts['k-1'], output_space=chi),
        (n @ ts['k']).multi(chi @ ts['k'], output_space=chi)
    ],
    ['+', '+', '+', '+'],
    factors=[1/4, 1/4, 1/4, 1/4],
)

wf = wf.derive.split(
    '0-2', 'f0',
    [
        (p @ ts['k-1']).multi(tau @ ts['k-1'], output_space=tau),
        (p @ ts['k-1']).multi(tau @ ts['k'], output_space=tau),
        (p @ ts['k']).multi(tau @ ts['k-1'], output_space=tau),
        (p @ ts['k']).multi(tau @ ts['k'], output_space=tau)
    ],
    ['+', '+', '+', '+'],
    factors=[1/4, 1/4, 1/4, 1/4],
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

# wf.pr(patterns=False)

wf = wf.derive.rearrange(
    {
        0: '0, 3, 5, 6, 7 =',
        1: '0, 3, 5, 6, 7 =',
        2: '0, 1, 3, 4 =',
        3: '0, 1, 3, 4 =',
        7: '0, 3, 4, 6, 8, 10 ='
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

term = wf.terms['0-3']
term.add_extra_info(
    {'known-forms': tau @ ts['k-1']}
)

term = wf.terms['0-6']
term.add_extra_info(
    {'known-forms': [p @ ts['k-1'], u @ 'k-1']}
)

term = wf.terms['0-7']
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

term = wf.terms['1-3']
term.add_extra_info(
    {'known-forms': chi @ ts['k-1']}
)

term = wf.terms['1-6']
term.add_extra_info(
    {'known-forms': [n @ ts['k-1'], u @ 'k-1']}
)

term = wf.terms['1-7']
term.add_extra_info(
    {'known-forms': [n @ ts['k-1'], chi @ 'k-1']}
)

term = wf.terms['2-0']
term.add_extra_info(
    {'known-forms': p @ ts['k-1']}
)

term = wf.terms['2-2']
term.add_extra_info(
    {'known-forms': delta @ ts['k-1']}
)

term = wf.terms['2-4']
term.add_extra_info(
    {'known-forms': [delta @ ts['k-1'], p @ 'k-1']}
)

term = wf.terms['3-0']
term.add_extra_info(
    {'known-forms': n @ ts['k-1']}
)

term = wf.terms['3-2']
term.add_extra_info(
    {'known-forms': theta @ ts['k-1']}
)

term = wf.terms['3-4']
term.add_extra_info(
    {'known-forms': [theta @ ts['k-1'], n @ 'k-1']}
)

term = wf.terms['7-1']
term.add_extra_info(
    {'known-forms': w @ ts['k-1']}
)

term = wf.terms['7-4']
term.add_extra_info(
    {'known-forms': p @ ts['k-1']}
)

term = wf.terms['7-5']
term.add_extra_info(
    {'known-forms': n @ ts['k-1']}
)

term = wf.terms['7-7']
term.add_extra_info(
    {'known-forms': [w @ ts['k-1'], u @ ts['k-1']]}
)

term = wf.terms['7-9']
term.add_extra_info(
    {'known-forms': [p @ ts['k-1'], tau @ ts['k-1']]}
)

term = wf.terms['7-10']
term.add_extra_info(
    {'known-forms': [n @ ts['k-1'], chi @ ts['k-1']]}
)

# wf.pr(patterns=False)

ph.space.finite(N)
# ap = wf.ap()
# ap.pr()

nls = wf.mp().nls()
# nls.pr()

# ------- implementation --------------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K,
    c=0,
    bounds=([0, 2*np.pi], [0, 2*np.pi]),
    periodic=True
)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize()

# total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
# msehtt.config(total_boundary)(tgm, including=msehtt_mesh)
# # total_boundary.visualize()

ts.specify('constant', [0, t_max, total_steps*2], 2)
e.value = epsilon

p = obj['p']
n = obj['n']
delta = obj['delta']
theta = obj['theta']
tau = obj['tau']
chi = obj['chi']
psi = obj['psi']
u = obj['u']
w = obj['w']
phi = obj['phi']

Sp = obj['Sp']
Sn = obj['Sn']
Sf = obj['Sf']

conditions = ph.samples.Manufactured_Solution_PNPNS_2D_PeriodicDomain1(
    mesh=msehtt_mesh, epsilon=epsilon, shift=3)

w.cf = conditions.omega
u.cf = conditions.u
p.cf = conditions.p
n.cf = conditions.n
delta.cf = conditions.delta
theta.cf = conditions.theta
phi.cf = conditions.phi
tau.cf = conditions.tau
chi.cf = conditions.chi
psi.cf = conditions.psi

Sp.cf = conditions.source_p
Sn.cf = conditions.source_n
Sf.cf = conditions.source_f

# p.cf.visualize()
# conditions.u.visualize()

w[0].reduce()
u[0].reduce()
p[0].reduce()
n[0].reduce()
tau[0].reduce()
chi[0].reduce()
delta[0].reduce()
theta[0].reduce()

# mu[0].reduce()
# nu[0].reduce()
# psi[0].reduce()
# psi[0].visualize()

# w[None].visualize(saveto=file_dir + rf"/w_0.png")
# p[None].visualize(saveto=file_dir + rf"/p_0.png")
# n[None].visualize(saveto=file_dir + rf"/n_0.png")

nLS = obj['nls'].apply()
# nLS.pr()

# NLS.config(
#     ('essential bc', 1),
#     total_boundary,
#     conditions.velocity_boundary_condition,
#     root_form=u
# )  # essential bc


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

    Sp[t_minus].reduce()
    Sn[t_minus].reduce()
    Sf[t_minus].reduce()
    phi[t_minus].reduce()
    psi[t].reduce()
    # phi[t_minus].visualize()

    system = nLS(k=k)
    # system.customize.linear.set_local_dof(6, 0, 0, psi[t].cochain.of_local_dof(0, 0))
    # system.customize.fixed_global_dofs_for_unknown(6, [0, ])
    system.customize.add_additional_constrain__fix_a_global_dof(6, 0)
    system.customize.linear.set_local_dof(9, 0, 0, phi[t_minus].cochain.of_local_dof(0, 0))
    system.solve(
        [p, n, delta, theta, tau, chi, psi, u, w, phi],
        atol=1e-6,
        # inner_solver_scheme='lgmres',
        # inner_solver_kwargs={'inner_m': 500, 'outer_k': 50, 'atol': 1e-6}
    )
    message = system.solve.message

#     linear_system = LS(k=k)
#     linear_system.customize.set_local_dof(8, 0, 0, 0)
#     linear_system.customize.set_local_dof(11, 0, 0, phi[t_minus].cochain.of_local_dof(0, 0))
#     Axb = linear_system.assemble()
#     x, message, info = Axb.solve('direct')
#     # x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)
#     linear_system.x.update(x)

    L2_error_p = p[t].error()
    L2_error_n = n[t].error()
    L2_error_psi = psi[t].error()
    L2_error_u = u[t].error()
    L2_error_omega = w[t].error()
    L2_error_phi = phi[t_minus].error()

    benchmark = np.array([
        0.006406499625160592, 0.014382406840383086, 0.006509556393853668,
        0.06545490941736584, 0.030442489048373007, 0.2508104544608167
    ])
    results = np.array([
        L2_error_p, L2_error_n, L2_error_psi, L2_error_u, L2_error_omega, L2_error_phi
    ])
    np.testing.assert_array_almost_equal(benchmark, results)
    return 0, message, t, L2_error_p, L2_error_n, L2_error_psi, L2_error_u, L2_error_omega, L2_error_phi


iterator = ph.iterator(
    solver,
    [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    name='iterator'
)

iterator.test([1, ], show_info=False)
