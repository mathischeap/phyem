# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


___cache_info___ = {}


def ndarray_key_comparer(cache, arrays, check_str=''):
    """"""
    assert isinstance(cache, dict), f"cache must be a dict."
    assert isinstance(arrays, (list, tuple)), f"pls put arrays in list or tuple"

    cid = id(cache)
    if cid not in ___cache_info___:
        ___cache_info___[cid] = {
            'attempts': 0,
            'hit': 0,
            'keep trying': True
        }
    else:
        pass

    info = ___cache_info___[cid]
    if info['keep trying']:
        pass
    else:
        return 0, None

    if len(cache) != 0:
        assert len(cache) == 3 and 'keys' in cache and 'values' in cache and 'check strings' in cache, \
            f"cache format wrong!"
    else:
        cache['keys'] = dict()
        cache['values'] = dict()
        cache['check strings'] = dict()

    keys = cache['keys']
    # `keys` itself is a dict:
    #  keys = {
    #       0: arrays0,
    #       1: arrays1,
    #       2: arrays2,
    #       3: arrays3,
    #       ...
    #  }

    values = cache['values']
    # `values` itself is a dict:
    #  values = {
    #       0: results0,
    #       1: results1,
    #       2: results2,
    #       3: results3,
    #       ...
    #  }

    check_strings = cache['check strings']
    # `check_strings` itself is a dict:
    #  check_strings = {
    #       0: str0,
    #       1: str1,
    #       2: str2,
    #       3: str3,
    #       ...
    #  }

    key = -1
    t_o_f = False
    for key in keys:
        cache_str = check_strings[key]

        if cache_str != check_str:
            pass
        else:

            c_arrays = keys[key]

            if len(c_arrays) == len(arrays):

                t_o_f = True
                for ca, ia in zip(c_arrays, arrays):  # ca: cached array, ia: input arrays
                    if not isinstance(ia, np.ndarray):
                        t_o_f = False
                        break

                    if ca.shape != ia.shape:
                        t_o_f = False
                        break

                    if not np.allclose(ca, ia):
                        t_o_f = False
                        break

            else:
                pass

        if t_o_f:  # we found it
            break
        else:
            pass

    if t_o_f:  # we found it

        cache_arrays = keys[key]
        assert (len(cache_arrays) == len(arrays) and
                all([np.allclose(cache_arrays[_], arrays[_]) for _ in range(len(cache_arrays))]) and
                check_strings[key] == check_str), \
            f"safety check!"
        _, __ = 1, values[key]
        hit_it = 1

    else:
        _, __ = 0, None
        hit_it = 0

    info['attempts'] += 1
    info['hit'] += hit_it

    hit_rate = info['hit'] / info['attempts']

    if info['attempts'] > 100 and hit_rate < 0.05:  # hit rate too low, stop caching
        info['keep trying'] = False
        # then we clean all cached data.
        keys = cache['keys']
        keys_to_clean = list(keys.keys())
        for key in keys_to_clean:  # clean cache since it will never be used.
            del keys[key]
            del values[key]
            del check_strings[key]
    else:
        pass

    return _, __


def add_to_ndarray_cache(cache, arrays, results, check_str='', maximum=25):
    """"""
    cid = id(cache)
    info = ___cache_info___[cid]
    if info['keep trying']:  # If we have stopped trying, do not cache anymore.
        pass
    else:
        return

    assert isinstance(cache, dict), f"cache must be a dict."
    assert isinstance(arrays, (list, tuple)), f"pls put arrays in list or tuple"
    assert maximum > 0 and maximum % 1 == 0, f"maximum={maximum} wrong, it must be a positive integer."

    keys = cache['keys']
    values = cache['values']
    check_strings = cache['check strings']

    for i, ar in enumerate(arrays):
        assert isinstance(ar, np.ndarray), f"arrays[{i}]={ar} is not a ndarray."

    # when there are more keys than 2 * maximum, we clean half (oldest) of them.
    if len(keys) > 2 * maximum:
        list_keys = list(keys.keys())
        list_keys.sort()
        keys_to_clean = list_keys[:maximum]

        for key in keys_to_clean:
            del keys[key]
            del values[key]
            del check_strings[key]
    else:
        pass

    if len(keys) > 0:
        current_key = max(keys.keys()) + 1
    else:
        current_key = 0

    keys[current_key] = arrays
    values[current_key] = results
    check_strings[current_key] = check_str
