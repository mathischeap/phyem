# -*- coding: utf-8 -*-
r"""Save objects to grouped region-wise structured data file.
"""
from numpy import testing
from src.config import RANK, MASTER_RANK
from msehtt.static.form.addons.static import MseHttFormStaticCopy
from tools.dds.region_wise_structured_group import DDS_RegionWiseStructured_Group


def _rws_grouped_saving(filename, *objs, ddf=1):
    """"""
    XYZ = None
    val_list = list()
    regions = None
    for obj in objs:

        if obj.__class__ is MseHttFormStaticCopy:
            rws = obj.numeric.rws(ddf=ddf, data_only=True)
            if RANK == MASTER_RANK:
                xyz, val = rws

                if XYZ is None:
                    XYZ = xyz
                    regions = xyz[0].keys()
                else:
                    for X, x in zip(XYZ, xyz):
                        assert x.keys() == regions, f"regions do not match."
                        for region in regions:
                            Coo = X[region]
                            coo = x[region]
                            testing.assert_array_almost_equal(Coo, coo, decimal=8)
                val_list.append(val)
            else:
                pass
        else:
            raise NotImplementedError(f"dds-rws-grouped saving not implemented for {obj}.")

    if RANK == MASTER_RANK:
        dds_rws_grouped = DDS_RegionWiseStructured_Group(XYZ, val_list)
        dds_rws_grouped.saveto(filename)
    else:
        pass
