# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:59 PM on 7/30/2023
"""
import pickle
from src.config import SIZE


def save(*objs, filename='phyem_objects'):
    """Save objs (be put in a dict) to filename."""
    obj_dict = dict()
    for obj in objs:
        saving_key, saving_obj = obj._saving_check()
        obj_dict[saving_key] = saving_obj

    if SIZE == 1:
        # we are only calling one thread, so just go ahead with it.
        with open(filename, 'wb') as output:
            pickle.dump(obj_dict, output, pickle.HIGHEST_PROTOCOL)
        output.close()

    else:
        raise NotImplementedError()
