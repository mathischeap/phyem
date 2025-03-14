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
    "InitialCondition_LoopAdvection",

    "Eigen2",
    "ManufacturedSolutionMHD2Ideal1",
    "ManufacturedSolutionNS2TGV",
    "ConditionsFlowAroundCylinder2",
    "ConditionsNormalDipoleCollision2",
    "ConditionsLidDrivenCavity2",
    "ConditionsLidDrivenCavity3",

    "ConditionsLidDrivenCavity_2dMHD_1",
    "ManufacturedSolutionMHD3_0",
    "ManufacturedSolutionMHD3_1",

    "ManufacturedSolutionNS3Conservation1",

    "ManufacturedSolutionLSingularity",

    "MHD3_Helicity_Conservation_test1",
]

from tests.samples.canonical_pH_pde import pde_canonical_pH
from tests.samples.div_grad_wf import wf_div_grad
from tests.samples.iniCond_shear_layer_rollup import InitialConditionShearLayerRollUp

from tests.samples.iniCond_Orszag_Tang_vortex import InitialConditionOrszagTangVortex
from tests.samples.iniCond_loop_advection import InitialCondition_LoopAdvection

from tests.samples.conditions_flow_around_cylinder_2d import ConditionsFlowAroundCylinder2
from tests.samples.conditions_normal_dipole_collision import ConditionsNormalDipoleCollision2
from tests.samples.conditions_lid_driven_cavity import ConditionsLidDrivenCavity2
from tests.samples.conditions_lid_driven_cavity import ConditionsLidDrivenCavity3

from tests.samples.conditions_lid_driven_cavity_MHD import ConditionsLidDrivenCavity_2dMHD_1

from tests.samples.manuSolution_Eigen2 import Eigen2
from tests.samples.manuSolution_MHD2 import ManufacturedSolutionMHD2Ideal1
from tests.samples.manuSolution_MHD3 import ManufacturedSolutionMHD3_0
from tests.samples.manuSolution_MHD3 import ManufacturedSolutionMHD3_1

from tests.samples.manuSolution_NS2 import ManufacturedSolutionNS2TGV

from tests.samples.manuSolution_NS3 import ManufacturedSolutionNS3Conservation1

from tests.samples.manuSolution_L_singularity import ManufacturedSolutionLSingularity

from tests.samples.condition_MHD3_Hu_helicity_test import MHD3_Helicity_Conservation_test1
