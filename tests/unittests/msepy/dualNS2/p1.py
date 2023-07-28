# -*- coding: utf-8 -*-
r"""
program # 1 of the dual NS scheme in 2D
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 3:28 PM on 7/21/2023

python tests/unittests/msepy/dualNS2/p1.py
"""
from numpy import pi
import sys

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph
ph.config.set_embedding_space_dim(2)

N = 3
K = 8

manifold = ph.manifold(2, is_periodic=True)
mesh = ph.mesh(manifold)

Inn0 = ph.space.new('Lambda', 0, orientation='inner')
Inn1 = ph.space.new('Lambda', 1, orientation='inner')
Inn2 = ph.space.new('Lambda', 2, orientation='inner')

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')

Pi = Inn0.make_form(r'P', 'inner-pressure')
ui = Inn1.make_form(r'u', 'inner-velocity')
wi = Inn2.make_form(r'\omega', 'inner-vorticity')

wo = Out0.make_form(r'\tilde{\omega}', 'outer-vorticity')
uo = Out1.make_form(r'\tilde{u}', 'outer-velocity')
Po = Out2.make_form(r'\tilde{P}', 'outer-pressure')

# ph.list_forms()

Rn = ph.constant_scalar(r'\frac{1}{\mathrm{Re}}', "Re")
Rn2 = ph.constant_scalar(r'\frac{1}{2\mathrm{Re}}', "Re2")

wo_x_ui = wo.cross_product(ui)
wi_x_uo = wi.cross_product(uo)

dPi = Pi.exterior_derivative()
cd_wi = Rn * wi.codifferential()
cd_ui = ui.codifferential()
d_ui = ui.exterior_derivative()

cd_Po = Po.codifferential()
d_wo = Rn * wo.exterior_derivative()
cd_uo = uo.codifferential()
d_uo = uo.exterior_derivative()

dui_dt = ui.time_derivative()
duo_dt = uo.time_derivative()

expression_inner = [
    'dui_dt + wo_x_ui + cd_wi + dPi = 0',
    'wi - d_ui = 0',
    '- cd_ui = 0',
]

inner_pde = ph.pde(expression_inner, locals())
inner_pde.unknowns = [ui, wi, Pi]
# inner_pde.pr()

expression_outer = [
    'duo_dt + wi_x_uo + d_wo - cd_Po = 0',
    'wo - cd_uo = 0',
    'd_uo = 0',
]

outer_pde = ph.pde(expression_outer, locals())
outer_pde.unknowns = [uo, wo, Po]
# outer_pde.pr()

inner_wf = inner_pde.test_with([Inn1, Inn2, Inn0], sym_repr=['v', 'w', 'q'])
inner_wf = inner_wf.derive.integration_by_parts('0-2')
inner_wf = inner_wf.derive.integration_by_parts('2-0')
# inner_wf.pr()

outer_wf = outer_pde.test_with([Out1, Out0, Out2], sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}'])
outer_wf = outer_wf.derive.integration_by_parts('0-3')
outer_wf = outer_wf.derive.integration_by_parts('1-1')
# outer_wf.pr()

ts = ph.time_sequence()  # initialize a time sequence
hdt = ts.make_time_interval('0', '1/2', sym_repr=r'\Delta t_{0}')

itd0 = inner_wf.td
itd0.set_time_sequence(ts)
itd0.define_abstract_time_instants('0', '1/2')
itd0.differentiate('0-0', '0', '1/2')
itd0.average('0-1', wo, ['0', ])
itd0.average('0-1', ui, ['0', '1/2'])
itd0.average('0-2', wi, ['0', '1/2'])
itd0.average('0-3', Pi, ['0', ])
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

iwf0 = iwf0.derive.split(
    '0-4', 'f0',
    [wi @ ts['0'], wi @ ts['1/2']],
    ['+', '+'],
    factors=[Rn2, Rn2],
)

iwf0 = iwf0.derive.rearrange(
    {
        0: '0, 3, 5, 6 = 1, 2, 4',
    }
)

# iwf0.pr()

term = iwf0.terms['0-5']
term.add_extra_info(
    {'known-cross-product-form': wo @ ts['0']}
)

term = iwf0.terms['0-1']
term.add_extra_info(
    {'known-cross-product-form': wo @ ts['0']}
)

ph.space.finite(N)

mp0 = iwf0.mp()
# mp0.pr()

ls0 = mp0.ls()
ls0.pr()

# ---- msepy implementation --------------------------------------------
msepy, obj = ph.fem.apply('msepy', locals())
manifold = obj['manifold']
msepy.config(manifold)(
    'crazy', c=0., bounds=[[0., 2*pi], [0, 2*pi]], periodic=True,
)
mesh = obj['mesh']
msepy.config(mesh)([15, 15])
# mesh.visualize()

ts.specify('constant', [0, 1, 100], 2)
from tests.samples.initial_condition_shear_layer_rollup import InitialConditionShearLayerRollUp
initial_condition = InitialConditionShearLayerRollUp()
Rn2.value = 1/100

wo = obj['wo']
ui = obj['ui']
wi = obj['wi']
wo.cf = initial_condition.vorticity
wi.cf = initial_condition.vorticity
ui.cf = initial_condition.velocity
wo[0].reduce()
wi[0].reduce()
ui[0].reduce()

ls0 = obj['ls0'].apply()
ls0.pr()

sls0 = ls0()
sls0.pr()
