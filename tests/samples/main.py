# -*- coding: utf-8 -*-
"""Here we store all helpful samples. They are mainly for testing purpose. You can also call them
for your own programs once you find them useful.

You can call them from ``ph.samples``.
"""

__all__ = [
    'pde_canonical_pH',
    'wf_div_grad',
    "InitialConditionShearLayerRollUp",
    "InitialConditionOrszagTangVortex",
]

from tests.samples.canonical_pH_pde import pde_canonical_pH
from tests.samples.div_grad_wf import wf_div_grad
from tests.samples.iniCond_shear_layer_rollup import InitialConditionShearLayerRollUp
from tests.samples.iniCond_Orszag_Tang_vortex import InitialConditionOrszagTangVortex
