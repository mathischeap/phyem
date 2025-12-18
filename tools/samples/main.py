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
    "InitialConditionOrszagTangVortex_3D",
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

    "ManufacturedSolution_Hall_MHD3_0",  # periodic domain [0, 2pi]^3
    "ManufacturedSolution_Hall_MHD3_1",  # [0, 2pi]^3, uxn = Bxn = 0, P =0
    "ManufacturedSolution_Hall_MHD3_2",  # [0, 2pi]^3, uxn = Exn = 0, P =0
    "ManufacturedSolution_Hall_MHD3_3",  # [0, 2pi]^3  omega x n = 0 = Bxn = 0, P = 0
    "ManufacturedSolution_Hall_MHD3_4_TemporalAccuracy",  # [0, 1]^3  uxn = Bxn = 0, P = 0; For temporal accuracy.
    "ManufacturedSolution_Hall_MHD3_Conservation0",  # [0, 1]^3 uxn = Bxn = 0, P = 0; initial condition;
                                                     # For conservation tests only.
    "ManufacturedSolution_Hall_MHD3_NullPoints",     # Null Points in [-1, 1]^3

    "Manufactured_Solution_PNPNS_2D_PeriodicDomain1",
    "Manufactured_Solution_PNPNS_2D_InitialDiscontinuousConcentrations",
]


from phyem.tools.samples.canonical_pH_pde import pde_canonical_pH
from phyem.tools.samples.div_grad_wf import wf_div_grad
from phyem.tools.samples.iniCond_shear_layer_rollup import InitialConditionShearLayerRollUp

from phyem.tools.samples.iniCond_Orszag_Tang_vortex import InitialConditionOrszagTangVortex
from phyem.tools.samples.iniCond_Orszag_Tang_vortex import InitialConditionOrszagTangVortex_3D

from phyem.tools.samples.iniCond_loop_advection import InitialCondition_LoopAdvection

from phyem.tools.samples.conditions_flow_around_cylinder_2d import ConditionsFlowAroundCylinder2
from phyem.tools.samples.conditions_normal_dipole_collision import ConditionsNormalDipoleCollision2
from phyem.tools.samples.conditions_lid_driven_cavity import ConditionsLidDrivenCavity2
from phyem.tools.samples.conditions_lid_driven_cavity import ConditionsLidDrivenCavity3

from phyem.tools.samples.conditions_lid_driven_cavity_MHD import ConditionsLidDrivenCavity_2dMHD_1

from phyem.tools.samples.manuSolution_Eigen2 import Eigen2
from phyem.tools.samples.manuSolution_MHD2 import ManufacturedSolutionMHD2Ideal1
from phyem.tools.samples.manuSolution_MHD3 import ManufacturedSolutionMHD3_0
from phyem.tools.samples.manuSolution_MHD3 import ManufacturedSolutionMHD3_1

from phyem.tools.samples.manuSolution_NS2 import ManufacturedSolutionNS2TGV

from phyem.tools.samples.manuSolution_NS3 import ManufacturedSolutionNS3Conservation1

from phyem.tools.samples.manuSolution_L_singularity import ManufacturedSolutionLSingularity

from phyem.tools.samples.condition_MHD3_Hu_helicity_test import MHD3_Helicity_Conservation_test1

from phyem.tools.samples.manuSolution_Hall_MHD import ManufacturedSolution_Hall_MHD3_0
from phyem.tools.samples.manuSolution_Hall_MHD import ManufacturedSolution_Hall_MHD3_1
from phyem.tools.samples.manuSolution_Hall_MHD import ManufacturedSolution_Hall_MHD3_2
from phyem.tools.samples.manuSolution_Hall_MHD import ManufacturedSolution_Hall_MHD3_3
from phyem.tools.samples.manuSolution_Hall_MHD import ManufacturedSolution_Hall_MHD3_4_TemporalAccuracy
from phyem.tools.samples.manuSolution_Hall_MHD import ManufacturedSolution_Hall_MHD3_Conservation0
from phyem.tools.samples.manuSolution_Hall_MHD import ManufacturedSolution_Hall_MHD3_NullPoints

from phyem.tools.samples.manuSolution_PNPNS2 import Manufactured_Solution_PNPNS_2D_PeriodicDomain1
from phyem.tools.samples.iniCond_PNPNS_InitialDiscontinuousConcentrations import (
    Manufactured_Solution_PNPNS_2D_InitialDiscontinuousConcentrations)
