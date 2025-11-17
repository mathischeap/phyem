# -*- coding: utf-8 -*-
r"""
"""
import pickle

from phyem.src.config import MASTER_RANK, RANK, SIZE, COMM
from phyem.tools.dds.region_wise_structured import DDSRegionWiseStructured
from phyem.tools.dds.region_wise_structured_group import DDS_RegionWiseStructured_Group


def read(filename, root=MASTER_RANK):
    """Read objects from a file. These objects can only be read into a single rank!

    The file can be resulted from "ph.save" (the standard saving interface) or not (a personal saving interface).

    It first must return a dict. And then we check if 'key' is a key of the dict.
    If it is, it means we are reading an object of a particular type.
    So we will call that particular class according to this 'key' to read the file again, such that
    we can return an instance of a correct type.

    Some classes may have its own read method. In that case, mostly, we need first set up an instance then
    read to that instance. But here, we call it by `ph.read` such that we can reconstruct an instance (or
    some instances) from nothing. The output will be in a dict even if there is only one instance being read.

    """
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
        return None


def read_tsf(filename):
    r"""This is a special reader. It read tsf objects to all ranks."""
    tsf = None
    for rank in range(SIZE):
        if rank == RANK:
            with open(filename, 'rb') as inputs:
                tsf = pickle.load(inputs)
            inputs.close()
        else:
            pass
        COMM.barrier()
    assert tsf._is_time_space_func(), f"I can only read time-space functions. Now I read {tsf}."
    return tsf
