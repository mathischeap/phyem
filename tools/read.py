# -*- coding: utf-8 -*-
r"""
"""
import pickle
from src.config import MASTER_RANK, RANK
from tools.dds.region_wise_structured import DDSRegionWiseStructured
from tools.dds.region_wise_structured_group import DDS_RegionWiseStructured_Group


def read(filename, root=MASTER_RANK):
    """Read from objects. These objects can only be read into a single rank!"""
    if RANK == root:
        with open(filename, 'rb') as inputs:
            obj = pickle.load(inputs)
        inputs.close()
        assert isinstance(obj, dict), f"phyem saving info must be a dict."
        if 'key' in obj:   # object that has particular reader.
            # in this case, we save a particular object in a dict whose key indicates its type
            key = obj['key']
            if key == 'dds-rws':  # dds-rws data, in the master core only.
                return DDSRegionWiseStructured.read(filename)
            elif key == 'dds-rws-grouped':  # dds-rws-grouped data, in the master core only.
                return DDS_RegionWiseStructured_Group.read(filename)
            else:
                raise NotImplementedError(key)
        else:  # we are reading a file coming from `ph.save`
            return obj  # a dict of multiple objects.
    else:
        pass
