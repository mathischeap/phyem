# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.src.config import COMM, MPI, RANK, MASTER_RANK
from phyem.tools.frozen import Frozen
from phyem.tools.miscellaneous.multigrid_scheme import MultiGridSchemeConfig, InterpolationStep, RestrictionStep

from phyem.msehtt.tools.vector.static.global_gathered import MseHttGlobalVectorGathered
from phyem.msehtt.tools.vector.static.global_distributed import MseHttGlobalVectorDistributed

from phyem.msehtt.tools.linear_system.static.global_.solvers.scipy_ import spsolve
from phyem.msehtt.tools.linear_system.static.global_.solvers.mpi_py import lgmres, gmres
from phyem.msehtt.tools.linear_system.static.global_.solvers.mpi_py import ___clean_cache___


def ___pair_levels___(scheme_max_num_layers, tgm_level_range):
    r""""""
    scheme_layer_to_mesh_level = dict()
    assert len(tgm_level_range) == scheme_max_num_layers, \
        f"mesh num levels ({len(tgm_level_range)}) does not match that of the scheme ({scheme_max_num_layers})."
    for i in range(scheme_max_num_layers):
        scheme_layer_to_mesh_level[i] = tgm_level_range[i]
    return scheme_layer_to_mesh_level


def _msehtt_multigrid_solver_(
        mg_global_linear_system,
        scheme, clean=True,
        x0=None, **kwargs,
):
    r""""""
    assert isinstance(scheme, MultiGridSchemeConfig), f"scheme={scheme} is not a <{MultiGridSchemeConfig}>."
    mg_local_linear_system = mg_global_linear_system._s_l_ls_
    tgm = mg_local_linear_system._mg_dls._base['the_great_mesh']
    layer2level = ___pair_levels___(scheme._num_max_layers, tgm.level_range)
    last_step = scheme._scheme_[-1]
    assert isinstance(last_step, InterpolationStep) and last_step._layer + 1 == tgm.level_range[-1], \
        f"The last step must be an interpolation and its output must be on the top-level mesh."

    # ------------ define a cache-key to cache the error of iterative solvers ---------------------------------
    ___clean_cache___()
    # ----------- take care of kwargs -------------------------------------------------------------------------
    # : we first select a linear system solver from kwargs ---
    if len(kwargs) == 0 or 'inner_m' in kwargs or 'outer_k' in kwargs:
        linearSolver = lgmres
        linearSolverName = 'lgmres'
    elif 'restart' in kwargs:
        linearSolver = gmres
        linearSolverName = 'gmres'
    else:
        raise NotImplementedError(f"cannot find a linear solver from kwargs={kwargs}")

    # : Then we select kwargs for the intermediate steps --
    intermediate_kwargs = {}
    intermediate_maxiter = 2
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

    # ----------- first prepare all A matrices, b vectors and unknowns -------------------------------------
    AAA = list()  # AAA[l] returns the global matrix A (MseHttGlobalMatrix) on the layer #l.
    bbb = list()  # bbb[l] returns the global vector b (MseHttGlobalVectorDistributed) on the layer #l.
    UNKNOWNS = list()
    TIMES = list()
    top_global_linear_system = None
    for layer in range(scheme._num_max_layers):
        lvl = layer2level[layer]
        lvl_global_linear_system = mg_global_linear_system.get_level(lvl)
        AAA.append(lvl_global_linear_system.A)
        bbb.append(lvl_global_linear_system.b)
        static_local_system = mg_local_linear_system.get_level(lvl)
        layer_unknowns = []
        layer_times = []
        for x in static_local_system.x._x:
            layer_unknowns.append(x._f)
            layer_times.append(x._time)
        UNKNOWNS.append(tuple(layer_unknowns))
        TIMES.append(tuple(layer_times))

        if layer == scheme._num_max_layers - 1:
            top_global_linear_system = lvl_global_linear_system
        else:
            pass

    stepIOs = dict()   # step input-output interfaces.

    # ------- prepare input data for the step#0 -------------------------------------------------
    # remember, a step is "——o". So a step takes the input (data) of the current layer,
    # then restrict or interpolate it to the next level
    # and finally does a computation on the next level.
    # the computation can be solving a system or add error to the existing solution or else.

    step = scheme[0]
    sIO = ___StepIO___(0, step._layer)
    if isinstance(step, InterpolationStep):
        assert step._layer == 0, \
            f"If we start with an {InterpolationStep}, initial step must be on base layer, i.e. bottom level, layer#0"

        # we will use direct solver to solve Ax=b such that we obtain the initial data for step0.

        A = AAA[0]
        b = bbb[0]

        if A._gm_col is None:
            raise NotImplementedError(f"Assembled A has no gathering matrix. Probably because that we customize it"
                                      f"during the assembling. So we need to clean it before use it.")
        else:
            x, message, info = spsolve(A, b)
            x = MseHttGlobalVectorGathered(x, gm=A._gm_col)
            error = A.M @ x.V - b.V

        sIO.Input = [x, error, None]  # we will use direct solve for step 0 on the base level
        sIO.itype = 'SOLUTION-ERROR-RHS'

    elif isinstance(step, RestrictionStep):
        assert step._layer == tgm.level_range[-1], \
            f"If we start with an {RestrictionStep}, initial step must be on max-layer, i.e. top-layer."
        A = AAA[step._layer]
        b = bbb[step._layer]
        x, message, info = top_global_linear_system.solve(
            linearSolverName, x0=x0,
            **intermediate_kwargs,
            maxiter=intermediate_maxiter,
            beta_cache_key='S0',
        )
        x = MseHttGlobalVectorGathered(x, gm=A._gm_col)
        error = A.M @ x.V - b.V
        sIO.Input = [x, error, b.V]  # we will use direct solve for step 0 on the base level
        sIO.itype = 'SOLUTION-ERROR-RHS'

    else:
        raise Exception()

    del top_global_linear_system

    stepIOs[0] = sIO

    # ----- go through all multigrid steps -------------------------------------------------------
    pairs = scheme.pairs

    new_unpaired_interpolation = False

    for s in scheme:
        step = scheme[s]
        layer = step._layer
        # A = AAA[layer]
        # b = bbb[layer]

        if s == 0:  # the first step, we should have produced the intput data.
            pass
        else:  # take the output of previous step as my input.
            assert s not in stepIOs
            sIO = ___StepIO___(s, layer)
            sIO.Input = stepIOs[s-1].output
            sIO.itype = stepIOs[s-1].o_type
            stepIOs[s] = sIO

        Input = stepIOs[s].Input
        itype = stepIOs[s].itype

        if isinstance(step, InterpolationStep):  # this is an interpolation step.
            restriction_index = pairs[s]
            if restriction_index is None:
                # this interpolation step is not corresponded to a restriction step.
                new_unpaired_interpolation = True
                if itype == 'SOLUTION-ERROR-RHS':
                    layer_unknowns = UNKNOWNS[layer]
                    upper_unknowns = UNKNOWNS[layer + 1]
                    layer_vector = Input[0]
                    layer_vectors = layer_vector.split()
                    upper_vectors = list()
                    for i, vec in enumerate(layer_vectors):
                        upper_vec = tgm.pass_vector(vec, layer_unknowns[i], upper_unknowns[i], complete_only=False)
                        upper_vectors.append(upper_vec)
                    A = AAA[layer + 1]
                    b = bbb[layer + 1]
                    upper_vector = A._gm_col.merge(*upper_vectors)  # this is used as x0
                    if layer + 1 == scheme._num_max_layers - 1:
                        beta_cache_key = f'S{s}'
                    else:
                        beta_cache_key = ''
                    x, message, info = linearSolver(A, b, upper_vector,
                                                    **intermediate_kwargs,
                                                    maxiter=intermediate_maxiter,
                                                    beta_cache_key=beta_cache_key)
                    x = MseHttGlobalVectorGathered(x, gm=A._gm_col)
                    error = A.M @ x.V - b.V
                    stepIOs[s].output = [x, error, b.V]
                    stepIOs[s].o_type = 'SOLUTION-ERROR-RHS'
                else:
                    raise NotImplementedError(
                        f"Not implemented for step[{s}]={step}, input={Input} of type {itype}.")

            else:
                # this interpolation step is corresponded to a restriction step.
                restriction = scheme[restriction_index]
                assert isinstance(restriction, RestrictionStep), f"must find a restriction step."
                layer_unknowns = UNKNOWNS[layer]
                upper_unknowns = UNKNOWNS[layer + 1]
                dx, _, _ = Input
                assert isinstance(dx, MseHttGlobalVectorGathered), \
                    f"layer_vector={dx} ({dx.__class__.__name__}) is wrong!"
                layer_vectors = dx.split()
                upper_dxs = list()
                for i, vec in enumerate(layer_vectors):
                    upper_vec = tgm.pass_vector(vec, layer_unknowns[i], upper_unknowns[i], complete_only=False)
                    upper_dxs.append(upper_vec)
                A = AAA[layer + 1]
                upper_dx = A._gm_col.merge(*upper_dxs)
                x, _, BV = stepIOs[restriction_index].Input
                assert isinstance(x, MseHttGlobalVectorGathered) and isinstance(upper_dx, MseHttGlobalVectorGathered)
                x._V = x._V + upper_dx._V
                error = A.M @ x.V - BV
                stepIOs[s].output = [x, error, BV]
                stepIOs[s].o_type = 'SOLUTION-ERROR-RHS'

                if clean:
                    stepIOs[restriction_index]._data = None
                    stepIOs[s-1]._data = None
                else:
                    pass

        elif isinstance(step, RestrictionStep):  # this is a restriction step.
            A = AAA[layer]  # must be MseHttGlobalMatrix
            error = Input[1]
            layer_error = np.zeros_like(error, dtype=float)
            COMM.Allreduce(error, layer_error, op=MPI.SUM)
            layer_error = MseHttGlobalVectorGathered(layer_error, gm=A._gm_col)
            layer_errors = layer_error.split()
            layer_unknowns = UNKNOWNS[layer]
            lower_unknowns = UNKNOWNS[layer-1]
            lower_errors = list()
            for i, err in enumerate(layer_errors):
                lower_err = tgm.pass_vector(err, layer_unknowns[i], lower_unknowns[i], complete_only=True)
                lower_errors.append(lower_err)
            A = AAA[layer - 1]
            lower_error = A._gm_col.merge(*lower_errors)
            if RANK == MASTER_RANK:
                lower_error_V = lower_error.V
            else:
                lower_error_V = np.zeros_like(lower_error.V)
            lower_error = MseHttGlobalVectorDistributed(lower_error_V)
            if layer - 1 == 0:
                x, message, info = spsolve(A, lower_error)
            else:
                x, message, info = linearSolver(A, lower_error, 0,
                                                **intermediate_kwargs,
                                                maxiter=intermediate_maxiter)
            x = MseHttGlobalVectorGathered(x, gm=A._gm_col)
            error = A.M @ x.V - lower_error.V
            stepIOs[s].output = [x, error, lower_error.V]
            stepIOs[s].o_type = 'SOLUTION-ERROR-RHS'

            if clean and new_unpaired_interpolation:  # clean useless stepIOs data
                for _s_ in stepIOs:  # to clean IO data of all unpaired InterpolationStep so far.
                    _step_ = scheme[_s_]
                    if isinstance(_step_, InterpolationStep):  # this is an interpolation.
                        restriction_index = pairs[_s_]
                        if restriction_index is None:  # this interpolation is unpaired.
                            stepIOs[_s_]._data = None
                        else:
                            pass
                    else:
                        pass
                new_unpaired_interpolation = False
            else:
                pass

        else:
            raise Exception(f"step={step} is wrong!")

    x = stepIOs[len(scheme)-1].output[0]
    A = AAA[scheme._num_max_layers-1]
    b = bbb[scheme._num_max_layers-1]
    x, message, info = linearSolver(A, b, x, **kwargs, beta_cache_key='SEND')

    top_lvl = layer2level[scheme._num_max_layers-1]
    static_local_system = mg_local_linear_system.get_level(top_lvl)
    static_local_system.x.update(x)

    for layer in range(scheme._num_max_layers)[::-1][1:]:
        layer_unknowns = UNKNOWNS[layer]
        layer_times = TIMES[layer]

        upper_unknowns = UNKNOWNS[layer + 1]
        upper_times = TIMES[layer + 1]
        for lt, ut in zip(layer_times, upper_times):
            assert abs(lt - ut) < 1e-12, \
                f"form times on {layer} and ({layer+1}) layers (now {lt} and {ut}) must be same."

        for upper_form, ut, layer_form, lt in zip(upper_unknowns, upper_times, layer_unknowns, layer_times):
            tgm.pass_cochain(upper_form, ut, layer_form, lt, complete_only=True)
    # if RANK == MASTER_RANK:
    #     from phyem.msehtt.tools.linear_system.static.global_.solvers.mpi_py import ___beta_cache___
    #     print(___beta_cache___)
    return x, message, info


class ___StepIO___(Frozen):
    r""""""
    def __init__(self, sth_step, layer):
        r""""""
        self._s = sth_step
        self._layer = layer
        self._data = {
            'Input': '',
            'itype': '',   # -SOLUTION : the Input is the solution of the system at the current layer.
            'output': '',
            'o_type': '',
            'rhs': '',
        }
        self._freeze()

    @property
    def Input(self):
        return self._data['Input']

    @property
    def itype(self):
        return self._data['itype']

    @property
    def output(self):
        return self._data['output']

    @property
    def o_type(self):
        return self._data['o_type']

    @Input.setter
    def Input(self, _Input):
        self._data['Input'] = _Input

    @itype.setter
    def itype(self, _itype):
        self._data['itype'] = _itype

    @output.setter
    def output(self, _output):
        self._data['output'] = _output

    @o_type.setter
    def o_type(self, _o_type):
        self._data['o_type'] = _o_type
