# -*- coding: utf-8 -*-
"""
"""
import pickle
from src.config import MASTER_RANK, RANK
from tools.dds.region_wise_structured import DDSRegionWiseStructured
from tools.dds.region_wise_structured_group import DDS_RegionWiseStructured_Group


def read(filename, root=MASTER_RANK):
    """Read from objects."""
    if RANK == root:
        with open(filename, 'rb') as inputs:
            obj = pickle.load(inputs)
        inputs.close()
        if 'key' in obj:   # object that has particular reader.
            key = obj['key']
            if key == 'dds-rws':  # dds-rws data, in the master core only.
                return DDSRegionWiseStructured.read(filename)
            elif key == 'dds-rws-grouped':  # dds-rws-grouped data, in the master core only.
                return DDS_RegionWiseStructured_Group.read(filename)
            else:
                raise NotImplementedError(key)
        else:
            return obj
    else:
        pass
