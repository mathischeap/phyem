# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class MseHttStatic_Local_Wrapper_NonlinearOperator_DataMaker(Frozen):
    """"""
    def __init__(self, correspondence):
        """"""
        for i, cf in enumerate(correspondence):

            for j, ocf in enumerate(correspondence):

                if i == j:
                    pass
                else:
                    assert cf is not ocf, f"correspondence form cannot be repeated."
        self._correspondence = correspondence  # being the forms.

        self._freeze()

    @property
    def correspondence(self):
        """"""
        return self._correspondence

    def reduce(self, provided_cochains):
        """"""
        assert len(provided_cochains) == len(self._correspondence), f"provided_cochains length wrong."
        num_not_provided = provided_cochains.count(None)
        if num_not_provided == 1:
            return self.reduce_to_vector(provided_cochains)
        else:
            raise NotImplementedError()

    def reduce_to_vector(self, provided_cochains):
        """"""
        raise NotImplementedError()


class MseHttStatic_Local_Wrapper_NonlinearOperator(Frozen):
    """This is only a wrapper, all functionalities should be provided by the data input instance. """

    def __init__(self, data_maker, correspondence):
        """"""
        assert issubclass(data_maker.__class__, MseHttStatic_Local_Wrapper_NonlinearOperator_DataMaker), \
            f"dm must be a subclass of DM wrapper."
        self._dm = data_maker

        for i, cf in enumerate(correspondence):

            for j, ocf in enumerate(correspondence):

                if i == j:
                    pass
                else:
                    assert cf is not ocf, f"correspondence form cannot be repeated."
        self._correspondence = correspondence
        # be the form static copies and a test form.
        # So, it is different from the correspondence of the dm.

        self._freeze()

    @property
    def correspondence(self):
        """"""
        return self._correspondence

    @property
    def dm(self):
        """The data maker."""
        return self._dm

    @property
    def data_maker(self):
        """The data maker."""
        return self._dm

    def _evaluate(self, provided_pairs):
        """"""
        provided_cochains = [None for _ in self.correspondence]
        for f_and_cochain in provided_pairs:
            f, cochain = f_and_cochain
            assert f is not None, f"must be!"
            assert cochain is not None, f'Cochain cannot be None.'
            if f in self.correspondence:
                provided_cochains[self.correspondence.index(f)] = cochain
            else:
                pass

        # noinspection PyArgumentList
        results = self._dm.reduce(provided_cochains)
        return results

    def __rmul__(self, other):
        """"""
        if isinstance(other, (float, int)):
            new_dm = ___Rmul_float_DM___(self._dm, other)
            return self.__class__(new_dm, self._correspondence)
        else:
            raise NotImplementedError()


class ___Rmul_float_DM___(MseHttStatic_Local_Wrapper_NonlinearOperator_DataMaker):
    """"""
    def __init__(self, original_dm, the_float):
        """"""
        super().__init__(original_dm.correspondence)
        self._melt()
        self._odm = original_dm
        self._f = the_float
        self._freeze()

    def reduce_to_vector(self, provided_cochains):
        """"""
        original_vector = self._odm.reduce_to_vector(provided_cochains)
        return self._f * original_vector
