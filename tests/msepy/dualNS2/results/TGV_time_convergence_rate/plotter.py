# -*- coding: utf-8 -*-
"""
python tests/msepy/dualNS2/results/TGV_time_convergence_rate/plotter.py
"""

import os
import sys

if './' not in sys.path:
    sys.path.append('./')
import __init__ as ph
import pandas as pd
from tools.miscellaneous.numpy_styple import NumpyStyleDocstringReader
current_dir = os.path.dirname(__file__)


def data_reader(N, steps, c):
    """

    Parameters
    ----------
    N :
        Polynomial degree
    steps :
        the amount of time steps
    c :
        Mesh deformation factor.

    Returns
    -------
    ui_L2_error :
    uo_L2_error :
    wi_L2_error :
    wo_L2_error :
    Pi_L2_error :
    Po_L2_error :

    """
    N = int(N)
    steps = int(steps)
    _ = c

    filename = current_dir + rf"\TGV_N{N}_K{32}_steps{steps}.csv"

    data = pd.read_csv(filename, index_col=0)
    array = data.to_numpy()
    keys = list(data.keys())
    returns = NumpyStyleDocstringReader(data_reader).Returns

    RT = list()
    for rt in returns:
        RT.append(array[-1, keys.index(rt)])

    return RT


step_range = [1, 2, 4, 8]

N = [[4 for _ in step_range], ]
Ts = [[i for i in step_range], ]
cs = [0, ]

pr = ph.run(data_reader)

if os.path.isfile(current_dir + '/run.txt'):
    os.remove(current_dir + '/run.txt')

pr.iterate(
    N, Ts, cs,
    writeto=current_dir + '/run.txt',
    show_progress=True,  # turn off iterator progress.
)

pr.visualize(
    'loglog', 'N', 'uo_L2_error', prime='input2', hcp=1, usetex=True,
    labels=['$N=4, K=32$', ],
    # styles=["-s", "-v"],
    # colors=[(0, 0, 0, 1), (0.75, 0.75, 0.75, 1)],
    title=False,
    # yticks=[1e1, 1e0,  1e-1, 1e-2, 1e-3],
    xlabel=r'$\Delta t$',
    ylabel=r"$\left\| u^{n-1}_h\right\|_{L^2-\mathrm{error}}$",
    order_text_size=15,
    show_order=True,
    plot_order_triangle={
        0: {'tp': (0.02, 0.25), 'order': 2},
    },
    saveto=current_dir + '/dual_velocity.png',
)
