r"""Direct linear system solver for the multigrid implementation of msehtt.

So, we directly solve the system on the lvl#0 mesh, its solution is projected to lvl#1 as the initial
guess. Then on lvl#1 it is solved with lgmres. And it's solution is passed to the next level until we
we have found the solution on the max-level mesh.

"""
from time import time


def _msehtt_multigrid_direct_linear_system_solver_(mg_global_linear_system, **kwargs):
    r""""""
    # -- we first select a linear system solver from kwargs ------------------------------
    if len(kwargs) == 0 or 'inner_m' in kwargs or 'outer_k' in kwargs:
        linearSolverName = 'lgmres'
    elif 'restart' in kwargs:
        linearSolverName = 'gmres'
    else:
        raise NotImplementedError(f"cannot find a linear solver from kwargs={kwargs}")

    intermediate_kwargs = {}
    intermediate_maxiter = 3
    for key in kwargs:
        if key == 'maxiter':
            maxiter = kwargs['maxiter']
            intermediate_maxiter = int(0.1 * maxiter) + 1
            if intermediate_maxiter > 5:
                intermediate_maxiter = 5
            else:
                assert intermediate_maxiter >= 1, f"must be!"
        else:
            intermediate_kwargs[key] = kwargs[key]

    # ====================================================================================

    mg_local_linear_system = mg_global_linear_system._s_l_ls_
    tgm = mg_local_linear_system._mg_dls._base['the_great_mesh']
    max_lvl = tgm.max_level
    level_range = tgm.level_range
    assert level_range[0] == 0, f"`level_range[0]` must be 0."
    assert level_range[-1] == max_lvl, f"`level_range[-1]` must be `max_level`."

    UNKNOWNS = list()
    TIMES = list()
    for lvl in level_range:
        static_local_system = mg_local_linear_system.get_level(lvl)
        lvl_unknowns = []
        lvl_times = []
        for x in static_local_system.x._x:
            lvl_unknowns.append(x._f)
            lvl_times.append(x._time)
        UNKNOWNS.append(tuple(lvl_unknowns))
        TIMES.append(tuple(lvl_times))

    x = 0
    MESSAGE = []
    info = -1

    for lvl in level_range:
        lvl_global_linear_system = mg_global_linear_system.get_level(lvl)
        lvl_local_linear_system = mg_local_linear_system.get_level(lvl)
        if lvl == 0:
            ts_ = time()
            x, message, _ = lvl_global_linear_system.solve('direct')
            info = 0
            _et = time()
            MESSAGE.append(f'lvl#0 [{lvl_global_linear_system.shape}] > direct ~ %.2f(s)' % (_et - ts_))
        else:
            ts_ = time()
            # noinspection PyUnboundLocalVariable
            pre_static_cochains = pre_lvl_local_linear_system.x._x
            cur_static_cochains = lvl_local_linear_system.x._x
            cfs = []  # current forms
            for i, pre_cochain in enumerate(pre_static_cochains):
                pf = pre_cochain._f
                p_time = pre_cochain._time
                f_pure_lin_repr = pf.abstract._pure_lin_repr
                assert p_time in pf.cochain, f"must be! as we have compute it on the previous level."
                cur_cochain = cur_static_cochains[i]
                cf = cur_cochain._f
                assert cur_cochain._time == p_time, f"must be! Since the system is at same time instances."
                assert f_pure_lin_repr == cf.abstract._pure_lin_repr, f"must be! abstract form must be the same one."
                tgm.pass_cochain(pf, p_time, cf, p_time)
                cfs.append(cf)
            if lvl == max_lvl:
                x, message, info = lvl_global_linear_system.solve(linearSolverName, x0=cfs, **kwargs)
            else:
                x, message, info = lvl_global_linear_system.solve(
                    linearSolverName, x0=cfs, **intermediate_kwargs, maxiter=intermediate_maxiter,
                )

            _et = time()
            MESSAGE.append(f'lvl#{lvl} [{lvl_global_linear_system.shape}] > lgmres ~ %.2f(s)' % (_et - ts_))
        lvl_local_linear_system.x.update(x)
        if lvl == max_lvl:
            MESSAGE.append('<TOP LVL MSG>:' + message)
            break
        else:
            pass
        pre_lvl_local_linear_system = lvl_local_linear_system

    reversed_level_range = level_range[::-1]
    for L, from_lvl in enumerate(reversed_level_range[:-1]):
        from_unknowns = UNKNOWNS[from_lvl]
        from_times = TIMES[from_lvl]

        to_lvl = reversed_level_range[L+1]
        to_unknowns = UNKNOWNS[to_lvl]
        to_times = TIMES[to_lvl]

        for ft, tt in zip(from_times, to_times):
            assert abs(ft - tt) < 1e-12, \
                f"form times on {from_lvl} and ({to_lvl}) layers (now {ft} and {tt}) must be same."

        for ff, ft, tf, tt in zip(from_unknowns, from_times, to_unknowns, to_times):
            tgm.pass_cochain(ff, ft, tf, tt)

    MESSAGE = '\n'.join(MESSAGE)
    return x, MESSAGE, info
