# -*- coding: utf-8 -*-
r"""Save some objects to a dds-rws-grouped object based on this partial elements (basically a mesh).
"""
from src.config import RANK, MASTER_RANK
from tools.frozen import Frozen
from msehtt.static.form.addons.static import MseHttFormStaticCopy
from tools.dds.region_wise_structured_group import DDS_RegionWiseStructured_Group


class MseHtt_PartialMesh_Elements_ExportTo_DDS_RWS_Grouped(Frozen):
    """With this property for partial mesh of elements, we can export objects (for example, forms) to
    grouped_rws data structure.

    """

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._freeze()

    def __call__(self, *objs, saveto=None, ddf=1):
        """"""
        XYZ = None
        val_list = list()
        for obj in objs:

            if obj.__class__ is MseHttFormStaticCopy:
                assert obj._f.tpm.composition is self._elements, f"{obj} is living in a wrong partial mesh."
                rws = obj.numeric.rws(ddf=ddf, data_only=True)
                xyz, val = rws

                if XYZ is None:
                    XYZ = xyz
                else:
                    pass

                val_list.append(val)
            else:
                raise NotImplementedError(f"msehtt partial elements rws not implemented for {obj}.")

        if RANK == MASTER_RANK:
            dds_rws_grouped = DDS_RegionWiseStructured_Group(XYZ, val_list)
            if saveto is None:
                return dds_rws_grouped
            else:
                dds_rws_grouped.saveto(saveto)
        else:
            pass
