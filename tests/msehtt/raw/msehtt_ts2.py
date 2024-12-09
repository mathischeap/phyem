# -*- coding: utf-8 -*-
"""
Test the reduction and reconstruction for msehtt mesh built upon msepy 2d meshes.

mpiexec -n 4 python tests/msehtt/raw/msehtt_ts2.py
"""

import sys

import numpy as np

ph_dir = './'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import __init__ as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 3
K = 5
c = 0

manifold = ph.manifold(2)
mesh = ph.mesh(manifold)

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')

Inn0 = ph.space.new('Lambda', 0, orientation='inner')
Inn1 = ph.space.new('Lambda', 1, orientation='inner')
Inn2 = ph.space.new('Lambda', 2, orientation='inner')

o0 = Out0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
o1 = Out1.make_form(r'\tilde{\omega}^1', 'outer-form-1')
o2 = Out2.make_form(r'\tilde{\omega}^2', 'outer-form-2')

i0 = Inn0.make_form(r'{\omega}^0', 'inner-form-0')
i1 = Inn1.make_form(r'{\omega}^1', 'inner-form-1')
i2 = Inn2.make_form(r'{\omega}^2', 'inner-form-2')

ph.space.finite(N)

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)('crazy', element_layout=K, c=c, ts=True)
tgm.visualize()
