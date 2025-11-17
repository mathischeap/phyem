# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.gathering_matrix.main import MseHttSpaceGatheringMatrix
from phyem.msehtt.static.space.reduce.main import MseHttSpaceReduce
from phyem.msehtt.static.space.reconstruct.main import MseHttSpaceReconstruct
from phyem.msehtt.static.space.mass_matrix.main import MseHttSpaceMassMatrix
from phyem.msehtt.static.space.local_dofs.main import MseHttSpace_Local_Dofs
from phyem.msehtt.static.space.error.main import MseHttSpaceError
from phyem.msehtt.static.space.incidence_matrix.main import MseHttSpaceIncidenceMatrix
from phyem.msehtt.static.space.norm.main import MseHttSpaceNorm
from phyem.msehtt.static.space.reconstruction_matrix.main import MseHttSpaceReconstructionMatrix
from phyem.msehtt.static.space.reconstruction_matrix_for_element_face.main import MseHttSpace_RMef
from phyem.msehtt.static.space.integrate_matrix_over_sub_geometry.main import MseHttSpace_IntMatOverSubGeo
from phyem.msehtt.static.space.reconstruct_on_element_face.main import MseHttSpace_RConEF
from phyem.msehtt.static.space.inner_product.main import MseHttSpace_InnerProduct


def _distribute_IMPLEMENTATION_space(indicator, m, n, **kwargs):
    r""""""
    new_IMPLEMENTATION_abstract_space = _IMPLEMENTATION_AbstractSpace_(indicator, m, n, **kwargs)
    new_IMPLEMENTATION_space = MseHttSpace(new_IMPLEMENTATION_abstract_space)
    return new_IMPLEMENTATION_space


class _IMPLEMENTATION_AbstractSpace_(Frozen):
    r""""""
    def __init__(self, indicator, m, n, **kwargs):
        r""""""
        self._indicator = indicator
        self._m = m
        self._n = n
        if indicator == 'Lambda':
            assert 'k' in kwargs, f"must have input 'k' for Lambda spaces."
            self._k = kwargs['k']
            assert 'orientation' in kwargs, f"must have input 'orientation' for Lambda spaces."
            self._orientation = kwargs['orientation']
        else:
            raise NotImplementedError()
        self._freeze()

    def __repr__(self):
        r""""""
        ab_space_repr = super().__repr__().split(' at ')[1]
        return '<IMPLEMENTATION-abstract-space at ' + ab_space_repr

    @property
    def indicator(self):
        return self._indicator

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        r"""Not always make sense."""
        return self._k

    @property
    def orientation(self):
        r"""Not always make sense."""
        return self._orientation


class MseHttSpace(Frozen):
    r""""""

    def __init__(self, abstract_space):
        """"""
        self._abstract = abstract_space
        self._tpm = None

        self._gm = None
        self._rd = None
        self._rc = None
        self._mm = None
        self._im = None
        self._LDofs = None
        self._error = None
        self._norm = None
        self._rm = None
        self._RMef = None
        self._RConEF = None
        self._int_mat_over_sub_geo = None
        self._ip_ = None

        self._freeze()

    # ------------ IMPLEMENTATION FORM ---------------------------------------------------------------------

    @property
    def _IMPLEMENTATION_d_next_space_(self):
        r""""""
        d_next_space_indicator = self.d_space_str_indicator

        if d_next_space_indicator == 'm2n2k1_inner':
            indicator = 'Lambda'
            m, n, k, orientation = 2, 2, 1, 'inner'
            IMPLEMTATION_space = _distribute_IMPLEMENTATION_space(indicator, m, n, k=k, orientation=orientation)
            IMPLEMTATION_space._tpm = self._tpm
            return IMPLEMTATION_space

        elif d_next_space_indicator == 'm2n2k1_outer':
            indicator = 'Lambda'
            m, n, k, orientation = 2, 2, 1, 'outer'
            IMPLEMTATION_space = _distribute_IMPLEMENTATION_space(indicator, m, n, k=k, orientation=orientation)
            IMPLEMTATION_space._tpm = self._tpm
            return IMPLEMTATION_space

        elif d_next_space_indicator in ('m2n2k2', 'm3n3k1', 'm3n3k2', 'm3n3k3'):
            indicator = 'Lambda'
            m, n, k, orientation = self.m, self.n, int(d_next_space_indicator[-1]), self.orientation
            IMPLEMTATION_space = _distribute_IMPLEMENTATION_space(indicator, m, n, k=k, orientation=orientation)
            IMPLEMTATION_space._tpm = self._tpm
            return IMPLEMTATION_space

        else:
            raise NotImplementedError(d_next_space_indicator)

    # ======================================================================================================

    @property
    def tpm(self):
        if self._tpm is None:
            raise Exception(f"first set tpm: I am {self}!")
        return self._tpm

    @property
    def tgm(self):
        return self.tpm._tgm

    @property
    def abstract(self):
        """The abstract space of me."""
        return self._abstract

    @property
    def indicator(self):
        """The indicator showing what type of space I am."""
        return self.abstract.indicator

    @property
    def m(self):
        """the dimensions of the space I am living in."""
        return self.abstract.m

    @property
    def n(self):
        """the dimensions of the mesh I am living in."""
        return self.abstract.n

    @property
    def _imn_(self):
        """"""
        return self.indicator, self.m, self.n

    @property
    def mn(self):
        return self.m, self.n

    @property
    def str_indicator(self):
        """"""
        idc, m, n = self._imn_
        if idc == 'Lambda':
            k = self.abstract.k
            if m == n == 2 and k == 1:
                orientation = self.abstract.orientation
                return f"m{m}n{n}k{k}_{orientation}"
            else:
                return f"m{m}n{n}k{k}"
        else:
            raise NotImplementedError(idc)

    @property
    def d_space_str_indicator(self):
        """"""
        _, m, n = self._imn_
        indicator = self.str_indicator

        if m == n == 3:
            if indicator == 'm3n3k0':
                return 'm3n3k1'
            elif indicator == 'm3n3k1':
                return 'm3n3k2'
            elif indicator == 'm3n3k2':
                return 'm3n3k3'
            else:
                raise Exception(f"m3n3 for what? {indicator}")

        elif m == n == 2:
            if indicator == 'm2n2k0':
                orientation = self.orientation
                if orientation == 'inner':
                    return 'm2n2k1_inner'
                elif orientation == 'outer':
                    return 'm2n2k1_outer'
                else:
                    raise Exception()
            elif indicator == 'm2n2k1_inner':
                return 'm2n2k2'
            elif indicator == 'm2n2k1_outer':
                return 'm2n2k2'
            else:
                raise Exception(f"m2n2 for what? {indicator}")

        else:
            raise NotImplementedError(indicator)

    @property
    def orientation(self):
        """The orientation I am."""
        return self.abstract.orientation

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        return '<MseHtt ' + ab_space_repr + super().__repr__().split('object')[1]

    @property
    def gathering_matrix(self):
        """"""
        if self._gm is None:
            self._gm = MseHttSpaceGatheringMatrix(self)
        return self._gm

    @property
    def reduce(self):
        """"""
        if self._rd is None:
            self._rd = MseHttSpaceReduce(self)
        return self._rd

    @property
    def reconstruct(self):
        """"""
        if self._rc is None:
            self._rc = MseHttSpaceReconstruct(self)
        return self._rc

    @property
    def mass_matrix(self):
        """"""
        if self._mm is None:
            self._mm = MseHttSpaceMassMatrix(self)
        return self._mm

    @property
    def local_dofs(self):
        r"""Information of local dofs in elements."""
        if self._LDofs is None:
            self._LDofs = MseHttSpace_Local_Dofs(self)
        return self._LDofs

    @property
    def error(self):
        """"""
        if self._error is None:
            self._error = MseHttSpaceError(self)
        return self._error

    @property
    def incidence_matrix(self):
        """"""
        if self._im is None:
            self._im = MseHttSpaceIncidenceMatrix(self)
        return self._im

    @property
    def norm(self):
        """"""
        if self._norm is None:
            self._norm = MseHttSpaceNorm(self)
        return self._norm

    @property
    def reconstruction_matrix(self):
        """"""
        if self._rm is None:
            self._rm = MseHttSpaceReconstructionMatrix(self)
        return self._rm

    @property
    def RMef(self):
        r"""reconstruction matrix for element face/edge"""
        if self._RMef is None:
            self._RMef = MseHttSpace_RMef(self)
        return self._RMef

    @property
    def RoF(self):
        r"""reconstruction matrix for element face/edge"""
        if self._RConEF is None:
            self._RConEF = MseHttSpace_RConEF(self)
        return self._RConEF

    @property
    def iMsg(self):
        r"""integrate matrix over a sub geometry."""
        if self._int_mat_over_sub_geo is None:
            self._int_mat_over_sub_geo = MseHttSpace_IntMatOverSubGeo(self)
        return self._int_mat_over_sub_geo

    @property
    def inner_product(self):
        r""""""
        if self._ip_ is None:
            self._ip_ = MseHttSpace_InnerProduct(self)
        return self._ip_
