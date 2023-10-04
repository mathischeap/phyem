# -*- coding: utf-8 -*-
r"""
python tests/msepy/dualNS3/conservation_test.py
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph
ph.config.set_embedding_space_dim(3)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(True)

N = 2
K = 5
t_max = 10
steps = 20 * t_max

N = int(N)
K = int(K)
t_max = int(t_max)
steps = int(steps)

manifold = ph.manifold(3, is_periodic=True)
mesh = ph.mesh(manifold)

Inn0 = ph.space.new('Lambda', 0, orientation='inner')
Inn1 = ph.space.new('Lambda', 1, orientation='inner')
Inn2 = ph.space.new('Lambda', 2, orientation='inner')

Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')
Out3 = ph.space.new('Lambda', 3, orientation='outer')

Pi = Inn0.make_form(r'P', 'inner-pressure')
ui = Inn1.make_form(r'u', 'inner-velocity')
wi = Inn2.make_form(r'\omega', 'inner-vorticity')

wo = Out1.make_form(r'\tilde{\omega}', 'outer-vorticity')
uo = Out2.make_form(r'\tilde{u}', 'outer-velocity')
Po = Out3.make_form(r'\tilde{P}', 'outer-pressure')

wo_x_ui = wo.cross_product(ui)
wi_x_uo = wi.cross_product(uo)
# ph.list_forms()

dPi = Pi.exterior_derivative()
cd_ui = ui.codifferential()
d_ui = ui.exterior_derivative()

cd_Po = Po.codifferential()
cd_uo = uo.codifferential()
d_uo = uo.exterior_derivative()

dui_dt = ui.time_derivative()
duo_dt = uo.time_derivative()

expression_inner = [
    'dui_dt + wo_x_ui + dPi = 0',
    'wi - d_ui = 0',
    '- cd_ui = 0',
]

inner_pde = ph.pde(expression_inner, locals())
inner_pde.unknowns = [ui, wi, Pi]
inner_pde.pr(vc=True)

expression_outer = [
    'duo_dt + wi_x_uo - cd_Po = 0',
    'wo - cd_uo = 0',
    'd_uo = 0',
]

outer_pde = ph.pde(expression_outer, locals())
outer_pde.unknowns = [uo, wo, Po]
outer_pde.pr(vc=True)

inner_wf = inner_pde.test_with([Inn1, Inn2, Inn0], sym_repr=['v', 'w', 'q'])
inner_wf = inner_wf.derive.integration_by_parts('2-0')

outer_wf = outer_pde.test_with([Out2, Out1, Out3], sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}'])
outer_wf = outer_wf.derive.integration_by_parts('0-2')
outer_wf = outer_wf.derive.integration_by_parts('1-1')
#
inner_wf.pr()
outer_wf.pr()

ts = ph.time_sequence()  # initialize a time sequence
dto = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t_o')
dti = ts.make_time_interval('k-1/2', 'k+1/2', sym_repr=r'\Delta t_i')
hdt = ts.make_time_interval('0', '1/2', sym_repr=r'\Delta t_{0}')

# -------------- the first half temporal step ------------------------------------------------------------
itd0 = inner_wf.td
itd0.set_time_sequence(ts)
itd0.define_abstract_time_instants('0', '1/2')
itd0.differentiate('0-0', '0', '1/2')
itd0.average('0-1', wo, ['0', ])
itd0.average('0-1', ui, ['0', '1/2'])
itd0.average('0-2', Pi, ['0', ])
itd0.average('1-0', wi, ['1/2', ])
itd0.average('1-1', ui, ['1/2', ])
itd0.average('2-0', ui, ['1/2', ])
iwf0 = itd0()

iwf0.unknowns = [
    ui @ ts['1/2'],
    wi @ ts['1/2'],
    Pi @ ts['0'],
    ]

iwf0 = iwf0.derive.split(
    '0-0', 'f0',
    [ui @ ts['1/2'], ui @ ts['0']],
    ['+', '-'],
    factors=[1/hdt, 1/hdt],
)

iwf0 = iwf0.derive.split(
    '0-2', 'f0',
    [(wo @ ts['0']).cross_product(ui @ ts['0']), (wo @ ts['0']).cross_product(ui @ ts['1/2'])],
    ['+', '+'],
    factors=[1/2, 1/2],
)

iwf0 = iwf0.derive.rearrange(
    {
        0: '0, 3, 4 = 1, 2',
    }
)

iwf0.pr()

term = iwf0.terms['0-4']
term.add_extra_info(
    {'known-cross-product-form': wo @ ts['0']}
)
term = iwf0.terms['0-1']
term.add_extra_info(
    {'known-cross-product-form': wo @ ts['0']}
)
ph.space.finite(N)
mp0 = iwf0.mp()
ls0 = mp0.ls()
ls0.pr()

# -------------- half step discretization --------------------------------------------------------
itd = inner_wf.td
itd.set_time_sequence(ts)
itd.define_abstract_time_instants('k-1/2', 'k', 'k+1/2')
itd.differentiate('0-0', 'k-1/2', 'k+1/2')
itd.average('0-1', wo, ['k', ])
itd.average('0-1', ui, ['k-1/2', 'k+1/2'])
itd.average('0-2', Pi, ['k', ])
itd.average('1-0', wi, ['k+1/2', ])
itd.average('1-1', ui, ['k+1/2', ])
itd.average('2-0', ui, ['k+1/2', ])
iwf = itd()
iwf.pr()
iwf.unknowns = [
    ui @ ts['k+1/2'],
    wi @ ts['k+1/2'],
    Pi @ ts['k'],
    ]
iwf = iwf.derive.split(
    '0-0', 'f0',
    [ui @ ts['k+1/2'], ui @ ts['k-1/2']],
    ['+', '-'],
    factors=[1/dti, 1/dti],
)

iwf = iwf.derive.split(
    '0-2', 'f0',
    [(wo @ ts['k']).cross_product(ui @ ts['k-1/2']), (wo @ ts['k']).cross_product(ui @ ts['k+1/2'])],
    ['+', '+'],
    factors=[1/2, 1/2],
)

iwf = iwf.derive.rearrange(
    {
        0: '0, 3, 4 = 1, 2',
    }
)

iwf.pr()

term = iwf.terms['0-4']
term.add_extra_info(
    {'known-cross-product-form': wo @ ts['k']}
)
term = iwf.terms['0-1']
term.add_extra_info(
    {'known-cross-product-form': wo @ ts['k']}
)
ph.space.finite(N)
mpi = iwf.mp()
lsi = mpi.ls()
lsi.pr()

# ----------- outer ----------------------------------------------------------------
otd = outer_wf.td
otd.set_time_sequence(ts)
otd.define_abstract_time_instants('k-1', 'k-1/2', 'k')
otd.differentiate('0-0', 'k-1', 'k')
otd.average('0-1', wi, ['k-1/2', ])
otd.average('0-1', uo, ['k-1', 'k'])
otd.average('0-2', Po, ['k-1/2', ])
otd.average('1-0', wo, ['k', ])
otd.average('1-1', uo, ['k', ])
otd.average('2-0', uo, ['k', ])
owf = otd()

owf.unknowns = [
    uo @ ts['k'],
    wo @ ts['k'],
    Po @ ts['k-1/2'],
    ]
owf = owf.derive.split(
    '0-0', 'f0',
    [uo @ ts['k'], uo @ ts['k-1']],
    ['+', '-'],
    factors=[1/dto, 1/dto],
)

owf = owf.derive.split(
    '0-2', 'f0',
    [(wi @ ts['k-1/2']).cross_product(uo @ ts['k-1']), (wi @ ts['k-1/2']).cross_product(uo @ ts['k'])],
    ['+', '+'],
    factors=[1/2, 1/2],
)

owf = owf.derive.rearrange(
    {
        0: '0, 3, 4 = 1, 2',
    }
)
term = owf.terms['0-4']
term.add_extra_info(
    {'known-cross-product-form': wi @ ts['k-1/2']}
)
term = owf.terms['0-1']
term.add_extra_info(
    {'known-cross-product-form': wi @ ts['k-1/2']}
)
owf.pr()

ph.space.finite(N)
mpo = owf.mp()
lso = mpo.ls()
lso.pr()

# ------------- implementation ---------------------------------------------------
msepy, obj = ph.fem.apply('msepy', locals())
manifold = obj['manifold']
msepy.config(manifold)(
    'crazy', c=0., bounds=[[0, 1], [0, 1], [0, 1]], periodic=True,
)
mesh = obj['mesh']
msepy.config(mesh)([K, K, K])
# ts.specify('constant', [0, t_max, 2*steps], 2)

initial_condition = ph.samples.ManufacturedSolutionNS3Conservation1()

wo = obj['wo']
wi = obj['wi']
uo = obj['uo']
ui = obj['ui']
wo.cf = initial_condition.omega
wi.cf = initial_condition.omega
uo.cf = initial_condition.u
ui.cf = initial_condition.u
wo[0].reduce()
wi[0].reduce()
uo[0].reduce()
ui[0].reduce()

ls0 = obj['ls0'].apply()
lsi = obj['lsi'].apply()
lso = obj['lso'].apply()

s_ls0 = ls0()
s_ls0.customize.set_dof(-1, 0)
As_ls0 = s_ls0.assemble()
As_ls0.solve()

msepy.info()
K2_t0 = 0.5 * uo[None].norm() ** 2
K1_t_half = 0.5 * ui[None].norm() ** 2
#
E1_t0 = 0.5 * wo[None].norm() ** 2
E2_t_half = 0.5 * wi[None].norm() ** 2


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
    K2 :
    K1 :
    E1 :
    E2 :

    """
    s_lso = lso(k=k)
    s_lso.customize.set_dof(-1, 0)
    As_lso = s_lso.assemble()
    results = As_lso.solve()
    s_lso.x.update(results[0])

    s_lsi = lsi(k=k)
    s_lsi.customize.set_dof(-1, 0)
    As_lsi = s_lsi.assemble()
    results = As_lsi.solve()
    s_lsi.x.update(results[0])

    t = uo.cochain.newest

    K2 = 0.5 * uo[None].norm() ** 2
    K1 = 0.5 * ui[None].norm() ** 2

    E1 = 0.5 * wo[None].norm() ** 2
    E2 = 0.5 * wi[None].norm() ** 2

    m0 = As_lso.solve.message
    m1 = As_lsi.solve.message

    return 0, m0+m1, t, K2, K1, E1, E2


path = rf'__phcache__/NS3_conservation_test/N{N}K{K}_t{t_max}steps{steps}'
iterator = ph.iterator(
    solver,
    [0, K2_t0, K1_t_half, E1_t0, E2_t_half],
    name=path,
)
iterator.run(range(1, steps+1))
