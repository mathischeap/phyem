# -*- coding: utf-8 -*-
r"""Save objects to file. The objects must have method `_saving_check`. In other words, if we provide
a `_saving_check` method, we can save it using this function.
"""
import pickle
from src.config import RANK, MASTER_RANK


def save(*objs, filename='phyem_objects'):
    """Save objs (be put in a dict) to filename using the standard interface.

    A class has a standard interface means it has a `_saving_check` method.

    Remember that, a class may have its own save and read methods. That it fine. Because not every class
    could or is suitable for a standard save interface.
    """
    if RANK == MASTER_RANK:
        obj_dict = {}
    else:
        pass

    for obj in objs:
        assert hasattr(obj, "_saving_check"), \
            (f"obj: {obj} cannot be saved using ph.save, "
             f"implement _saving_check method for its class first.")
        saving_info = obj._saving_check()
        if RANK == MASTER_RANK:
            saving_key, saving_obj = saving_info
            assert saving_key != 'key', f"it cannot be `key`."
            # noinspection PyUnboundLocalVariable
            obj_dict[saving_key] = saving_obj

    if RANK == MASTER_RANK:
        # we are only calling one thread, so just go ahead with it.
        with open(filename, 'wb') as output:
            # noinspection PyTypeChecker
            pickle.dump(obj_dict, output, pickle.HIGHEST_PROTOCOL)
        output.close()
    else:
        pass
