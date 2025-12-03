r"""
mpiexec -n 4 python tests/msehtt/multigrid/PNPNS_decoupled_linearized.py
"""
import phyem as ph
import numpy as np

N = 2
K = 4
steps_per_second = 100

epsilon = 1

t_max = 1
total_steps = steps_per_second * t_max
_dt_ = t_max / total_steps

iterator_name = f"N{N}K{K}s{steps_per_second}"

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
n = Inn0.make_form(r' n ', 'negative-ions')

mu = Inn0.make_form(r'\mu', 'mu')
nu = Inn0.make_form(r'\nu', 'nu')

# tau = Inn1.make_form(r'\tau', 'tau')
# chi = Inn1.make_form(r'\chi', 'chi')

dp_dt = p.time_derivative()
convective_p = u.convect(p)

# d_mu = mu.d()
# d_nv = nu.d()

p_d_mu = p.multi(mu.d(), output_space=Inn1)
cd_p_d_mu = p_d_mu.codifferential()

dn_dt = n.time_derivative()
convective_n = u.convect(n)
n_d_nu = n.multi(nu.d(), output_space=Inn1)
cd_n_d_nu = n_d_nu.codifferential()

log_p = p.log()
log_n = n.log()

psi = Inn0.make_form(r'\psi', 'electronic')
d_psi = psi.d()
laplace_psi = e * d_psi.codifferential()

du_dt = u.time_derivative()

wxn = w.x(u, form_space=u)

dw = w.d()

d_phi = phi.codifferential()

p_tau_Hodge = ph.Hodge(p_d_mu)
n_chi_Hodge = ph.Hodge(n_d_nu)

cd_u = u.codifferential()
du = u.d()
# ph.list_forms()

# ------------ PNP ------------------------------------------------------------------
expression = [
    'dp_dt + convective_p + cd_p_d_mu = Sp',
    'dn_dt + convective_n + cd_n_d_nu = Sn',
    'laplace_psi - p + n = 0',
]

pde = ph.pde(expression, locals())
pde.unknowns = [p, n, psi]
# pde.pr(vc=True)

wf = pde.test_with(
    [Inn0, Inn0, Inn0],
    sym_repr=["q", "m", r"\varphi"],
)
wf = wf.derive.integration_by_parts('0-2')
wf = wf.derive.integration_by_parts('1-2')
wf = wf.derive.integration_by_parts('2-0')
wf = wf.derive.integration_by_parts('0-1', drop_bi_term=True)
wf = wf.derive.integration_by_parts('1-1', drop_bi_term=True)

ts = ph.time_sequence()  # initialize a time sequence

dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')
td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
td.differentiate('0-0', 'k-1', 'k')
td.differentiate('1-0', 'k-1', 'k')
td.average('0-1', p, ['k-1', 'k'])
td.average('0-1', u, ['k-1/2', ])
td.average('0-2', p, ['k-1', 'k'])
td.average('0-2', mu, ['k-1', ])
td.average('0-3', Sp, ['k-1/2', ])
td.average('1-1', n, ['k-1', 'k'])
td.average('1-1', u, ['k-1/2', ])
td.average('1-2', n, ['k-1', 'k'])
td.average('1-2', nu, ['k-1', ])
td.average('1-3', Sn, ['k-1/2', ])
td.average('2-0', psi, ['k', ])
td.average('2-1', p, ['k', ])
td.average('2-2', n, ['k', ])

wf = td()

wf.unknowns = [
    p @ 'k',
    n @ 'k',
    psi @ 'k',
]

wf = wf.derive.split(
    '1-2', 'f0',
    [
        (n @ ts['k-1']).multi((nu @ ts['k-1']).d(), output_space=Inn1),
        (n @ ts['k']).multi((nu @ ts['k-1']).d(), output_space=Inn1),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '0-2', 'f0',
    [
        (p @ ts['k-1']).multi((mu @ ts['k-1']).d(), output_space=Inn1),
        (p @ ts['k']).multi((mu @ ts['k-1']).d(), output_space=Inn1),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '1-1', 'f0',
    [
        (n @ ts['k-1']).multi(u @ ts['k-1/2'], output_space=u),
        (n @ ts['k']).multi(u @ ts['k-1/2'], output_space=u),
    ],
    ['+', '+'],
    factors=[1/2, 1/2],
)

wf = wf.derive.split(
    '0-1', 'f0',
    [
        (p @ ts['k-1']).multi(u @ ts['k-1/2'], output_space=u),
        (p @ ts['k']).multi(u @ ts['k-1/2'], output_space=u),
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
    }
)

term = wf.terms['0-1']
term.add_extra_info(
    {'known-forms': u @ ts['k-1/2']}
)

term = wf.terms['0-2']
term.add_extra_info(
    {'known-forms': mu @ ts['k-1']}
)

term = wf.terms['0-4']
term.add_extra_info(
    {'known-forms': [p @ ts['k-1'], u @ 'k-1/2']}
)

term = wf.terms['0-5']
term.add_extra_info(
    {'known-forms': [p @ ts['k-1'], mu @ 'k-1']}
)

term = wf.terms['1-1']
term.add_extra_info(
    {'known-forms': u @ ts['k-1/2']}
)

term = wf.terms['1-2']
term.add_extra_info(
    {'known-forms': nu @ ts['k-1']}
)

term = wf.terms['1-4']
term.add_extra_info(
    {'known-forms': [n @ ts['k-1'], u @ 'k-1/2']}
)

term = wf.terms['1-5']
term.add_extra_info(
    {'known-forms': [n @ ts['k-1'], nu @ 'k-1']}
)

# wf.pr(patterns=True)

ph.space.finite(N)
# ap = wf.ap()
# ap.pr()

PNP_ls = wf.mp().ls()
# PNP_ls.pr()

# --------------- mu, nu ------------------------------------------------------
expression = [
    'mu = log_p + psi',
    'nu = log_n - psi',
]

pde = ph.pde(expression, locals())
pde.unknowns = [mu, nu]
# pde.pr(vc=False)

wf = pde.test_with(
    [Inn0, Inn0],
    sym_repr=["A", "B"],
)

td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k')
td.average('0-0', mu, ['k', ])
td.average('0-1', p, ['k', ])
td.average('0-2', psi, ['k', ])
td.average('1-0', nu, ['k', ])
td.average('1-1', n, ['k', ])
td.average('1-2', psi, ['k', ])

wf = td()

wf.unknowns = [
    mu @ 'k',
    nu @ 'k',
]

# wf.pr(patterns=False)

ph.space.finite(N)
# ap = wf.ap()
# ap.pr()

# mp = wf.mp()
# mp.pr()

MuNu_ls = wf.mp().ls()
# MuNu_ls.pr()

# ----------- NS ---------------------------------------------------------------
expression = [
    'du_dt + wxn + dw - d_phi = - p_tau_Hodge - n_chi_Hodge + Sf',
    'w - cd_u = 0',
    'du = 0'
]

pde = ph.pde(expression, locals())
pde.unknowns = [u, w, phi]
# pde.pr(vc=True)

wf = pde.test_with(
    [Out1, Out0, Out2],
    sym_repr=[r"\tilde{v}", r"\tilde{w}", r"\tilde{Q}"],
)

wf = wf.derive.integration_by_parts('0-3')
wf = wf.derive.integration_by_parts('1-1')

wf = wf.derive.switch_to_duality_pairing('0-4')
wf = wf.derive.switch_to_duality_pairing('0-5')

dth = ts.make_time_interval('k-1/2', 'k+1/2', sym_repr=r'\delta t')
td = wf.td
td.set_time_sequence(ts)
td.define_abstract_time_instants('k-1/2', 'k', 'k+1/2')
td.differentiate('0-0', 'k-1/2', 'k+1/2')
td.average('0-1', w, ['k-1/2', ])
td.average('0-1', u, ['k+1/2', ])
td.average('0-2', w, ['k+1/2', ])
td.average('0-3', phi, ['k', ])
td.average('0-4', p, ['k', ])
td.average('0-4', mu, ['k', ])
td.average('0-5', n, ['k', ])
td.average('0-5', nu, ['k', ])
td.average('0-6', Sf, ['k', ])
td.average('1-0', w, ['k+1/2', ])
td.average('1-1', u, ['k+1/2', ])
td.average('2-0', u, ['k+1/2', ])
wf = td()

wf.unknowns = [
    u @ 'k+1/2',
    w @ 'k+1/2',
    phi @ 'k'
]

wf = wf.derive.split(
    '0-0', 'f0',
    [u @ ts['k+1/2'], u @ ts['k-1/2']],
    ['+', '-'],
    factors=[1/dth, 1/dth],
)

wf = wf.derive.rearrange(
    {
        0: '0, 2, 3, 4 =',
    }
)

term = wf.terms['0-1']
term.add_extra_info(
    {'known-forms': w @ ts['k-1/2']}
)

term = wf.terms['0-5']
term.add_extra_info(
    {'known-forms': [p @ ts['k'], mu @ 'k']}
)

term = wf.terms['0-6']
term.add_extra_info(
    {'known-forms': [n @ ts['k'], nu @ 'k']}
)

# wf.pr(patterns=True)

# _ = (mu @ 'k').d()
# _ = (nu @ 'k').d()

ph.space.finite(N)
# ap = wf.ap()
# ap.pr()

# mp = wf.mp()
# mp.pr()

NS_ls = wf.mp().ls()
# NS_ls.pr()

# ------- implementation --------------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-smg', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K,
    c=0,
    bounds=([0, 2*np.pi], [0, 2*np.pi]),
    periodic=True,
    mgc=3,
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
psi = obj['psi']
mu = obj['mu']
nu = obj['nu']
u = obj['u']
w = obj['w']
phi = obj['phi']

Sp = obj['Sp']
Sn = obj['Sn']
Sf = obj['Sf']


conditions = ph.samples.Manufactured_Solution_PNPNS_2D_PeriodicDomain1(
    mesh=msehtt_mesh, epsilon=epsilon, shift=3)

p.cf = conditions.p
n.cf = conditions.n
psi.cf = conditions.psi

mu.cf = conditions.mu
nu.cf = conditions.nu

u.cf = conditions.u
w.cf = conditions.omega
phi.cf = conditions.phi

Sp.cf = conditions.source_p
Sn.cf = conditions.source_n
Sf.cf = conditions.source_f

# p.cf.visualize()
# conditions.u.visualize()

p[0].reduce()
n[0].reduce()
psi[0].reduce()

mu[0].reduce()
nu[0].reduce()

u['1/2'].reduce()
w['1/2'].reduce()

# msehtt.info()

PNP_LS = obj['PNP_ls'].apply()
MuNu_LS = obj['MuNu_ls'].apply()
NS_LS = obj['NS_ls'].apply()

# linear_system = MuNu_LS(k=0)

# PNP_LS.pr()
# MuNu_LS.pr()
# NS_LS.pr()


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
    L2_error_mu :
    L2_error_nu :
    L2_error_u :
    L2_error_omega :
    L2_error_phi :
    """
    t = ts['k'](k=k)()
    t_minus = ts['k-1/2'](k=k)()
    t_plus = ts['k+1/2'](k=k)()

    Sp[t_minus].reduce()
    Sn[t_minus].reduce()

    Sf[t].reduce()
    phi[t].reduce()

    linear_system = PNP_LS(k=k)
    linear_system.customize.set_local_dof(2, 0, 0, 0)
    Axb = linear_system.assemble()
    x, message0, info = Axb.solve('direct', inner_m=300, outer_k=30)
    # x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)
    linear_system.x.update(x)

    linear_system = MuNu_LS(k=k)
    Axb = linear_system.assemble()
    x, message1, info = Axb.solve('direct', inner_m=300, outer_k=30)
    # x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)
    linear_system.x.update(x)

    linear_system = NS_LS(k=k)
    linear_system.customize.set_local_dof(2, 0, 0, phi[t].cochain.of_local_dof(0, 0))
    Axb = linear_system.assemble()
    x, message2, info = Axb.solve('direct', inner_m=300, outer_k=30)
    # x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)
    linear_system.x.update(x)

    L2_error_p = p[t].error()
    L2_error_n = n[t].error()
    L2_error_psi = psi[t].error()

    L2_error_mu = mu[t].error()
    L2_error_nu = nu[t].error()

    L2_error_u = u[t_plus].error()
    L2_error_omega = w[t_plus].error()
    L2_error_phi = phi[t].error()

    # msehtt.info(
    #     f"L2_error_p={L2_error_p}",
    #     f"L2_error_n={L2_error_n}",
    #     f"L2_error_psi={L2_error_psi}",
    #     f"L2_error_mu={L2_error_mu}",
    #     f"L2_error_nu={L2_error_nu}",
    #     f"L2_error_u={L2_error_u}",
    #     f"L2_error_omega={L2_error_omega}",
    #     f"L2_error_phi={L2_error_phi}",
    # )

    if k == 3:
        res = np.array(
            [
                L2_error_p, L2_error_n, L2_error_psi,
                L2_error_mu, L2_error_nu,
                L2_error_u, L2_error_omega, L2_error_phi
            ]
        )
        np.testing.assert_array_almost_equal(
            res,
            np.array([0.00376778, 0.00822698, 0.00288885, 0.00372026,
                      0.00736167, 0.02647005, 0.00376469, 0.09859863])
        )
    else:
        pass

    return (0, [message0, message1, message2], t,
            L2_error_p, L2_error_n, L2_error_psi,
            L2_error_mu, L2_error_nu,
            L2_error_u, L2_error_omega, L2_error_phi
            )


iterator = ph.iterator(
    solver,
    [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    name=iterator_name
)

iterator.test([1, 2, 3], show_info=False)
