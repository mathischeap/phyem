# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from tools.frozen import Frozen
from msepy.form.cochain.time_instant import _CochainAtOneTime
from msepy.form.main import MsePyRootForm
from msepy.form.static import MsePyRootFormStaticCopy
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix


class MsePyStaticLocalNonlinearOperator(Frozen):
    """"""

    def __init__(self, local_static_NOD, particular_correspondence, direct_derivative_contribution=False):
        """

        Parameters
        ----------
        local_static_NOD
        particular_correspondence:
            the corresponding msepy forms at particular time instances.
        """
        if isinstance(local_static_NOD, dict) and \
           all([isinstance(local_static_NOD[_], np.ndarray) for _ in local_static_NOD]):
            # local_static_MDA is provided as a dict of ndarray(s).
            self._dtype = 'ndarray'
            self._data = local_static_NOD
            self._derivative_method = self._default_ndarray_derivative
        elif callable(local_static_NOD):
            self._dtype = 'abstract'  # then probably, the data cannot be formatted into a ndarray. It is abstract.
            self._data = local_static_NOD
            assert hasattr(local_static_NOD, "derivative"), \
                f"callable nonlinear operator data must can compute derivative through method `_derivative`."
            # noinspection PyUnresolvedReferences
            self._derivative_method = local_static_NOD.derivative
        else:
            raise NotImplementedError()

        for i, cf in enumerate(particular_correspondence):
            assert cf.__class__ in (MsePyRootForm, MsePyRootFormStaticCopy), \
                f"corresponding form class wrong."
            for j, ocf in enumerate(particular_correspondence):

                if i == j:
                    pass
                else:
                    assert cf is not ocf, f"correspondence cannot be repeated."

        self._correspondence = particular_correspondence
        self._direct_derivative_contribution = direct_derivative_contribution
        self._mesh = None
        self._operands_cache = dict()
        self._freeze()

    @property
    def correspondence(self):
        """"""
        return self._correspondence

    @property
    def mesh(self):
        if self._mesh is None:
            for form in self.correspondence:
                if self._mesh is None:
                    self._mesh = form.mesh
                else:
                    assert self._mesh is form.mesh, f"correspondence forms have different meshes."
        return self._mesh

    def __call__(self, e, cochain_providers=None):
        """provide another layer such that in the future we can implement something new.

        Note that we DO NOT adjust of customize a nonlinear local data!
        """
        return self._get_raw_data(e, cochain_providers=cochain_providers)

    def _get_raw_data(self, e, cochain_providers=None):
        """# evaluate the mda in mesh element #e by involving the cochains from ``cochain_providers``.

        For example, ``cochain_providers`` == [f0, None, f2], we will use f0 as cochain for the correspondence[0],
        and use f2 for correspondence[2]. Then we will have a 1d vector as output for element #e.

        This must return a k-d ndarray, k is the amount of None in cochain_providers.

        """
        # parse default cochain_providers:
        if cochain_providers is None:
            cochain_providers = [None for _ in self.correspondence]
        else:
            pass

        # check cochain_providers .
        assert isinstance(cochain_providers, (list, tuple)), f"pls put cochain_providers in a list or tuple."
        if len(cochain_providers) == len(self.correspondence):
            pass
        else:
            raise Exception(f"wrong amount of cochain provided.")

        # parse cochain_providers to get local-cochain ndarray objects (1d arrays)
        provided_cochains = list()
        valid_cochains = list()
        for i, cp in enumerate(cochain_providers):

            if cp is None:
                provided_cochains.append(None)

            else:
                if isinstance(cp, np.ndarray):

                    if cp.ndim == 2:
                        _1d_array = cp[e]
                    elif cp.ndim == 1:
                        _1d_array = cp
                    else:
                        raise Exception()

                else:
                    cp_class = cp.__class__
                    if cp_class is _CochainAtOneTime:
                        _1d_array = cp.local[e]
                    elif cp_class is MsePyRootFormStaticCopy:
                        _1d_array = cp.cochain.local[e]
                    elif cp_class is MsePyRootFormStaticCochainVector:
                        _1d_array = cp[e]  # this will raise Error if cp has no (None) data.
                    else:
                        raise NotImplementedError(
                            f"cannot get local cochain for element #{e} from {cp}"
                        )

                assert isinstance(_1d_array, np.ndarray) and _1d_array.ndim == 1, \
                    f"I must be a 1d ndarray. Now I am of shape {_1d_array.shape}."

                provided_cochains.append(_1d_array)
                valid_cochains.append(_1d_array)

        # let compute the ndarray
        if self._dtype == 'ndarray':  # we already have a ndarray, let eliminate some axes.
            if len(valid_cochains) == 0:
                # noinspection PyUnresolvedReferences
                return self._data[e]
            else:
                operands = self._make_operands(provided_cochains)
                # noinspection PyUnresolvedReferences
                return np.einsum(operands, self._data[e], *valid_cochains, optimize='optimal')

        elif self._dtype == 'abstract':  # we have an abstract data, we compute the output from the data.
            return self._data(e, provided_cochains)

        else:
            raise Exception()

    def _make_operands(self, provided_cochains):
        """make operands for the einsum in __call__ method."""
        regularity = tuple([0 if _ is None else 1 for _ in provided_cochains])
        if regularity in self._operands_cache:
            return self._operands_cache[regularity]

        else:
            num_cr = len(self.correspondence)
            indices = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
            if num_cr >= len(indices):
                raise NotImplementedError(f"expand the indices")
            else:
                pass

            mda_indices = ''.join(indices[:num_cr])
            given_indices = list()
            result_indices = list()
            i = 0
            for pc in provided_cochains:
                if pc is None:
                    result_indices.append(indices[i])
                else:
                    given_indices.append(indices[i])
                i += 1
            given_indices = ','.join(given_indices)
            result_indices = ''.join(result_indices)

            operands = mda_indices + ',' + given_indices + '->' + result_indices
            self._operands_cache[regularity] = operands
            return operands

    def __iter__(self):
        """go through all mesh elements."""
        for e in self.mesh.elements:
            yield e

    def __len__(self):
        """How many mesh elements?"""
        return len(self.mesh.elements)

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            # c * self; c is a number
            helper = _RmulFactorHelper(self, other)
            # noinspection PyTypeChecker
            return MsePyStaticLocalNonlinearOperator(
                helper, self._correspondence, direct_derivative_contribution=self._direct_derivative_contribution
            )

        else:
            raise Exception()

    def eliminate(self, cochain_providers):
        """Do the __call__ for provided_cochains over all mesh elements, collect the results and make it a new
        self.__class__ object.
        """
        # check cochain_providers .
        assert isinstance(cochain_providers, (list, tuple)), f"pls put cochain_providers in a list or tuple."

        if all([_ is None for _ in cochain_providers]):
            raise Exception('must eliminate something; must provide some cochain!')
        else:
            pass

        if len(cochain_providers) == len(self.correspondence):
            pass
        else:
            raise Exception(f"wrong amount of cochain provided.")

        # parse cochain_providers to get local-cochain ndarray objects (1d arrays)
        provided_cochains = list()
        new_correspondence = list()
        for i, cp in enumerate(cochain_providers):

            if cp is None:  # not provided, remaining part
                provided_cochains.append(None)
                new_correspondence.append(self.correspondence[i])

            else:

                if isinstance(cp, np.ndarray):

                    if cp.ndim == 2:
                        _2d_array = cp
                    else:
                        raise Exception()

                else:
                    cp_class = cp.__class__
                    if cp_class is _CochainAtOneTime:
                        _2d_array = cp.local
                    elif cp_class is MsePyRootFormStaticCopy:
                        _2d_array = cp.cochain.local
                    elif cp_class is MsePyRootFormStaticCochainVector:
                        _2d_array = cp.data  # raise Error if we cannot do this
                    else:
                        raise NotImplementedError(
                            f"cannot get local cochain from {cp}"
                        )

                assert isinstance(_2d_array, np.ndarray) and _2d_array.ndim == 2, \
                    f"I must be a 2d ndarray. Now I am of shape {_2d_array.shape}."

                provided_cochains.append(_2d_array)

        the_data = dict()

        for e in self:
            # ``provided_cochains`` are all 2-d array, the full local cochain of each form.
            the_data[e] = self(e, provided_cochains)  # values are add ndarray.

        return self.__class__(the_data, new_correspondence)

    def matrix(self, transpose=False):
        """if self(e) gives only 2-d ndarray, we can express this mda as a 2d local matrix."""
        M = _Matrix(self, transpose)
        assert len(self.correspondence) == 2, \
            f"I can only have two corresponding forms, now I have {len(self.correspondence)}"
        gms = list()
        for cf in self.correspondence:
            if cf.__class__ is MsePyRootForm:
                gm = cf.cochain.gathering_matrix
            elif cf.__class__ is MsePyRootFormStaticCopy:
                gm = cf._f.cochain.gathering_matrix
            else:
                raise Exception()
            gms.append(gm)
        return MsePyStaticLocalMatrix(M, *gms, cache_key='unique')  # 2d csr_matrix is made in realtime

    def derivative(self, i, e):
        """derivative of correspondence[i] for element #e.

        Parameters
        ----------
        i
        e

        Returns
        -------

        """
        return self._derivative_method(i, e)

    def _default_ndarray_derivative(self, i, e):
        """When we provide ndarray, the derivative is just itself, as a ndarray.

        derivative of correspondence[i] for element #e.
        """
        assert 0 <= i <= len(self.correspondence), f"axis={i} is out of range."
        ndarray = self(e)
        return ndarray

    def _derivative_contribution(self, test_form, unknown, *known_pairs):
        """"""
        if self._direct_derivative_contribution:
            # ------ check the test_variable -----------------------------------------------
            assert test_form in self.correspondence, f"test_form is not found."
            # --if the unknown variable is not in the correspondence, return None --
            if unknown not in self.correspondence:
                return None

            # ------ we clear not relative pairs -----------------------------------------
            provided = list()
            amount_None = 0
            for cp in self.correspondence:
                the_cochain_form = None
                for kcp in known_pairs:
                    if cp is kcp[0]:
                        the_cochain_form = kcp[1]
                    else:
                        pass
                if the_cochain_form is None:
                    amount_None += 1
                provided.append(the_cochain_form)

            assert amount_None == len(self.correspondence) - len(known_pairs)

            MDM_2d = self.eliminate(provided)

            if MDM_2d.correspondence == [test_form, unknown]:
                return MDM_2d.matrix(transpose=False)
            elif MDM_2d.correspondence == [unknown, test_form]:
                return MDM_2d.matrix(transpose=True)
            else:
                raise Exception()
        else:
            raise NotImplementedError()

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def _evaluate(self, pairs):
        """"""
        provided_cochains = [None for _ in self.correspondence]
        for f_cochain in pairs:
            f, cochain = f_cochain
            if f in self.correspondence:
                provided_cochains[self.correspondence.index(f)] = cochain

        _evaluate = self.eliminate(provided_cochains)  # whose data must be dict
        data = _evaluate._data

        if data[0].ndim == 1:
            _2d_data = list()
            for e in data:
                _2d_data.append(data[e])

            _2d_data = np.array(_2d_data)

            return _2d_data

        else:
            raise NotImplementedError()


class _Matrix(Frozen):
    """"""
    def __init__(self, nop, transpose):
        """"""
        self._nop = nop
        self._T = transpose
        self._freeze()

    def __call__(self, e):
        dde = self._nop(e)
        assert isinstance(dde, np.ndarray) and dde.ndim == 2, f"data for element #{e} is not a 2d array."
        if self._T:
            return csr_matrix(dde).T
        else:
            return csr_matrix(dde)


class _RmulFactorHelper(Frozen):
    """"""
    def __init__(self, nop, factor):
        """"""
        self._nop = nop
        self._f = factor  # number
        self._freeze()

    def __call__(self, e, cochain_providers):
        """data of element #e."""
        raw_data = self._nop(e, cochain_providers)
        return self._f * raw_data

    def derivative(self, i, e):
        """derivative of correspondence[i] for element #e

        Parameters
        ----------
        i
        e

        Returns
        -------

        """
        raw_derivative = self._nop.derivative(i, e)
        return self._f * raw_derivative
