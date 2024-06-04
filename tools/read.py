# -*- coding: utf-8 -*-
"""
"""
import pickle
from src.config import SIZE, MASTER_RANK, RANK
from tools.dds.region_wise_structured import DDSRegionWiseStructured


def read(filename, root=MASTER_RANK):
    """Read from objects."""
    if SIZE == 1:
        with open(filename, 'rb') as inputs:
            objs = pickle.load(inputs)
        inputs.close()
        return objs
    else:
        if RANK == root:
            with open(filename, 'rb') as inputs:
                obj = pickle.load(inputs)
            inputs.close()
            if 'key' in obj:
                key = obj['key']
                if key == 'dds-rws':
                    return DDSRegionWiseStructured.read(filename)
                else:
                    raise NotImplementedError(key)
            else:
                return obj
        else:
            pass
