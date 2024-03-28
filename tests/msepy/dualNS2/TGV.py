# -*- coding: utf-8 -*-
r"""
2D TGV test.

python tests/msepy/dualNS2/TGV.py
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

import __init__ as ph


def TGV2(N, K, steps):
    """Compute from t=0 to t=1.

    dt = 1/steps.

    Parameters
    ----------
    N :
        mimetic polynomial degree
    K :
        K by K elements
    steps :
        The number of time steps.

    Returns
    -------

    """
    ph.config.set_embedding_space_dim(2)
    ph.config.set_high_accuracy(True)

    N = int(N)
    K = int(K)
    steps = int(steps)

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
    # inner_pde.pr(vc=True)

    expression_outer = [
        'duo_dt + wi_x_uo + d_wo - cd_Po = 0',
        'wo - cd_uo = 0',
        'd_uo = 0',
    ]

    outer_pde = ph.pde(expression_outer, locals())
    outer_pde.unknowns = [uo, wo, Po]
    # outer_pde.pr(vc=True)
    #
    inner_wf = inner_pde.test_with([Inn1, Inn2, Inn0], sym_repr=['v', 'w', 'q'])
    inner_wf = inner_wf.derive.integration_by_parts('0-2')
    inner_wf = inner_wf.derive.integration_by_parts('2-0')

    outer_wf = outer_pde.test_with([Out1, Out0, Out2], sym_repr=[r'\tilde{v}', r'\tilde{w}', r'\tilde{q}'])
    outer_wf = outer_wf.derive.integration_by_parts('0-3')
    outer_wf = outer_wf.derive.integration_by_parts('1-1')

    ts = ph.time_sequence()  # initialize a time sequence
    dto = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t_o')
    dti = ts.make_time_interval('k-1/2', 'k+1/2', sym_repr=r'\Delta t_i')
    hdt = ts.make_time_interval('0', '1/2', sym_repr=r'\Delta t_{0}')

    # -------------- half step discretization --------------------------------------------------------
    itd = inner_wf.td
    itd.set_time_sequence(ts)
    itd.define_abstract_time_instants('k-1', 'k-1/2', 'k', 'k+1/2')
    itd.differentiate('0-0', 'k-1/2', 'k+1/2')
    itd.average('0-1', wo, ['k', ])
    itd.average('0-1', ui, ['k-1/2', 'k+1/2'])
    itd.average('0-2', wi, ['k-1/2', 'k+1/2'])
    itd.average('0-3', Pi, ['k', ])
    itd.average('1-0', wi, ['k+1/2', ])
    itd.average('1-1', ui, ['k+1/2', ])
    itd.average('2-0', ui, ['k+1/2', ])
    iwf = itd()
    # iwf.pr()
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

    iwf = iwf.derive.split(
        '0-4', 'f0',
        [wi @ ts['k-1/2'], wi @ ts['k+1/2']],
        ['+', '+'],
        factors=[Rn2, Rn2],
    )

    iwf = iwf.derive.rearrange(
        {
            0: '0, 3, 5, 6 = 1, 2, 4',
        }
    )

    # iwf.pr()

    term = iwf.terms['0-5']
    term.add_extra_info(
        {'known-forms': wo @ ts['k']}
    )
    term = iwf.terms['0-1']
    term.add_extra_info(
        {'known-forms': wo @ ts['k']}
    )
    ph.space.finite(N)
    mpi = iwf.mp()
    lsi = mpi.ls()
    # lsi.pr()

    # ----------- outer ----------------------------------------------------------------
    otd = outer_wf.td
    otd.set_time_sequence(ts)
    otd.define_abstract_time_instants('k-1', 'k-1/2', 'k')
    otd.differentiate('0-0', 'k-1', 'k')
    otd.average('0-1', wi, ['k-1/2', ])
    otd.average('0-1', uo, ['k-1', 'k'])
    otd.average('0-2', wo, ['k-1', 'k'])
    otd.average('0-3', Po, ['k-1/2', ])
    otd.average('1-0', wo, ['k', ])
    otd.average('1-1', uo, ['k', ])
    otd.average('2-0', uo, ['k', ])
    owf = otd()
    # owf.pr()
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

    owf = owf.derive.split(
        '0-4', 'f0',
        [(wo @ ts['k-1']).exterior_derivative(), (wo @ ts['k']).exterior_derivative()],
        ['+', '+'],
        factors=[Rn2, Rn2],
    )

    owf = owf.derive.rearrange(
        {
            0: '0, 3, 5, 6 = 1, 2, 4',
        }
    )

    term = owf.terms['0-5']
    term.add_extra_info(
        {'known-forms': wi @ ts['k-1/2']}
    )
    term = owf.terms['0-1']
    term.add_extra_info(
        {'known-forms': wi @ ts['k-1/2']}
    )

    ph.space.finite(N)
    mpo = owf.mp()
    lso = mpo.ls()
    # lso.pr()

    # ------------- implementation ---------------------------------------------------
    msepy, obj = ph.fem.apply('msepy', locals())
    manifold = obj['manifold']
    msepy.config(manifold)(
        'crazy', c=0., bounds=[[0., 2], [0, 2]], periodic=True,
    )
    mesh = obj['mesh']
    msepy.config(mesh)([K, K])

    dt = 1 / steps

    ts.specify('constant', [0, 1, 2*steps], 2)

    Re = 100
    initial_condition = ph.samples.ManufacturedSolutionNS2TGV(Re=Re)
    Rn2.value = 1 / (2*Re)

    wo = obj['wo']
    wi = obj['wi']
    uo = obj['uo']
    ui = obj['ui']
    Po = obj['Po']
    Pi = obj['Pi']

    wo.cf = initial_condition.omega
    wi.cf = initial_condition.omega
    uo.cf = initial_condition.u
    ui.cf = initial_condition.u
    Po.cf = initial_condition.P
    Pi.cf = initial_condition.P

    wo[0].reduce()
    uo[0].reduce()

    ui[dt/2].reduce()
    wi[dt/2].reduce()

    Pi[0].reduce()
    Po[0].reduce()

    msepy.info()

    ui_L2_error0 = ui[None].error()
    uo_L2_error0 = uo[None].error()
    wi_L2_error0 = wi[None].error()
    wo_L2_error0 = wo[None].error()
    Pi_L2_error0 = Pi[None].error()
    Po_L2_error0 = Po[None].error()

    lsi = obj['lsi'].apply()
    lso = obj['lso'].apply()

    ati_o = ts['k-1/2']
    ati_i = ts['k']

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
        ui_L2_error :
        uo_L2_error :
        wi_L2_error :
        wo_L2_error :
        Pi_L2_error :
        Po_L2_error :

        """
        Po(ati=ati_o, k=k).reduce()
        Pi(ati=ati_i, k=k).reduce()

        s_lso = lso(k=k)
        s_lso.customize.set_local_dof_ij_of_unknown_k_to_value(
            i=0, j=0, k=2, value=Po[None].cochain.local[0, 0]
        )
        As_lso = s_lso.assemble()
        results = As_lso.solve()
        s_lso.x.update(results[0])

        s_lsi = lsi(k=k)
        s_lsi.customize.set_local_dof_ij_of_unknown_k_to_value(
            i=0, j=0, k=2, value=Pi[None].cochain.local[0, 0]
        )
        As_lsi = s_lsi.assemble()
        results = As_lsi.solve()
        s_lsi.x.update(results[0])

        ui_L2_error = ui[None].error()
        uo_L2_error = uo[None].error()
        wi_L2_error = wi[None].error()
        wo_L2_error = wo[None].error()
        Pi_L2_error = Pi[None].error()
        Po_L2_error = Po[None].error()

        return 0, As_lso.solve.message + As_lsi.solve.message, uo.cochain.newest, \
            ui_L2_error, uo_L2_error, wi_L2_error, wo_L2_error, Pi_L2_error, Po_L2_error

    import os
    current_dir = os.path.dirname(__file__)
    path = current_dir + rf'\results\TGV_N{N}_K{K}_steps{steps}'
    iterator = ph.iterator(
        solver,
        [0, ui_L2_error0, uo_L2_error0, wi_L2_error0, wo_L2_error0, Pi_L2_error0, Po_L2_error0],
        name=path,
    )
    iterator.run(range(1, steps+1))
    # os.remove(f'{path}.png')


if __name__ == '__main__':
    # python tests/msepy/dualNS2/TGV.py
    TGV2(4, 32, 2)
