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

import __init__ as ph

ls = ph.samples.wf_Poisson(n=1, degree=2, orientation='outer', periodic=True)

msepy, obj = ph.fem.apply('msepy', locals())

manifold = msepy.base['manifolds'][r"\mathcal{M}"]
mesh = msepy.base['meshes'][r'\mathfrak{M}']

phi = msepy.base['forms'][r'potential']
u = msepy.base['forms'][r'velocity']
f = msepy.base['forms'][r'source']

ls = obj['ls']
