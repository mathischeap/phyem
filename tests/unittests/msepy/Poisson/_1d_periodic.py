# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 3:58 PM on 5/9/2023

$ python tests/unittests/msepy/Poisson/_1d_periodic.py
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
import numpy as np
import __init__ as ph

ls = ph.samples.wf_Poisson(n=1, degree=2, orientation='outer', periodic=True)
ls.pr()

msepy, obj = ph.fem.apply('msepy', locals())

manifold = msepy.base['manifolds'][r"\mathcal{M}"]
mesh = msepy.base['meshes'][r'\mathfrak{M}']

msepy.config(manifold)(
    'crazy_multi', c=0, bounds=[[0, 2], ], periodic=True,
)
msepy.config(mesh)(3)

phi = msepy.base['forms'][r'potential']
u = msepy.base['forms'][r'velocity']
f = msepy.base['forms'][r'source']

ls = obj['ls'].apply()

ls.pr()

def f_func(t, x):
    """"""
    return np.sin(2* np.pi * x) + t

scalar = ph.vc.scalar(f_func)

f.cf = scalar

f[0].reduce()

ls0 = ls(0)
