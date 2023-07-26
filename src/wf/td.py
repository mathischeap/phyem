# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from src.tools.time_sequence import AbstractTimeSequence
from typing import Dict


class TemporalDiscretization(Frozen):
    """TemporalDiscretization"""

    def __init__(self, wf):
        """"""
        self._wf = wf
        self._initialize_odes()
        self._ats = None
        self._freeze()

    def _initialize_odes(self):
        """Initialize odes."""
        from src.ode.main import ode
        # must import locally here to avoid a circular import

        wf = self._wf
        valid_ode = dict()
        for i in wf._term_dict:
            terms = wf._term_dict[i]
            signs = wf._sign_dict[i]
            v_ode = ode(terms_and_signs=[terms, signs])
            valid_ode[i] = v_ode

        self._valid_ode = valid_ode

    def __getitem__(self, item):
        """Return the ode."""
        return self._valid_ode[item].discretize

    def __iter__(self):
        """iter over valid ode numbers."""
        for valid_ode_number in self._valid_ode:
            yield valid_ode_number

    def __call__(self, *args, **kwargs):
        """Return a new weak formulation by combining all equations."""
        wfs: Dict = dict()
        for i in self:
            wfs[i] = self[i]()
        for i in self._wf._term_dict:
            if i not in wfs:
                wfs[i] = {
                    '_term_dict': self._wf._term_dict[i],
                    '_sign_dict': self._wf._sign_dict[i]
                }
            else:
                pass
        wf = self._wf.__class__(self._wf._test_forms, merge=wfs)
        wf._bc = self._wf._bc
        # we get the new weak formulation by combining each pde.
        return wf

    @property
    def ts(self):
        """shortcut of `time_sequence`."""
        return self._ats

    @property
    def time_sequence(self):
        """The time sequence this discretization is working on."""
        return self._ats

    def set_time_sequence(self, ts=None):
        """The method of setting time sequence.

        Note that each wf only use one time sequence. If your time sequence is complex, you should carefully design
        it instead of trying to make multi time sequences.
        """
        if ts is None:  # make a new one
            ts = AbstractTimeSequence()
        else:
            pass
        assert ts.__class__.__name__ == 'AbstractTimeSequence', f"I need an abstract time sequence object."
        assert self._ats is None, f"time_sequence existing, change it may leads to unexpected issue."
        self._ats = ts
        for i in self:
            self[i].set_time_sequence(self._ats)

    def define_abstract_time_instants(self, *atis):
        """Define abstract time instants for all valid odes."""
        for i in self:
            self[i].define_abstract_time_instants(*atis)

    def differentiate(self, index, *args):
        """

        Parameters
        ----------
        index :
            The index of the weak formulation. So we parse it to locate the ode and the local index.
        args

        Returns
        -------

        """
        assert index in self._wf, f"index={index} is illegal, print representations to check the indices."
        i, j = index.split('-')
        ode_d = self[int(i)]
        ode_d.differentiate(j, *args)

    def average(self, index, *args):
        """

        Parameters
        ----------
        index :
            The index of the weak formulation. So we parse it to locate the ode and the local index.
        args

        Returns
        -------

        """
        assert index in self._wf, f"index={index} is illegal, print representations to check the indices."
        i, j = index.split('-')
        ode_d = self[int(i)]
        ode_d.average(j, *args)


if __name__ == '__main__':
    # python src/wf/td.py

    import __init__ as ph

    samples = ph.samples

    oph = samples.pde_canonical_pH(n=3, p=3)[0]
    oph.pr()
    a3, b2 = oph.unknowns

    wf = oph.test_with(oph.unknowns, sym_repr=[r'v^3', r'u^2'])
    wf = wf.derive.integration_by_parts('1-1')
    wf.pr(indexing=True)

    td = wf.td
    td.set_time_sequence()  # initialize a time sequence

    td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
    td.differentiate('0-0', 'k-1', 'k')
    td.average('0-1', b2, ['k-1', 'k'])

    td.differentiate('1-0', 'k-1', 'k')
    td.average('1-1', a3, ['k-1', 'k'])
    td.average('1-2', a3, ['k-1/2'])
    dt = td.time_sequence.make_time_interval('k-1', 'k')

    wf = td()
    wf.pr()
    wf.unknowns = [a3 @ td.time_sequence['k'], b2 @ td.time_sequence['k']]
    wf = wf.derive.split('0-0', 'f0',
                         [a3 @ td.ts['k'], a3 @ td.ts['k-1']],
                         ['+', '-'],
                         factors=[1/dt, 1/dt])
    wf = wf.derive.split('0-2', 'f0',
                         [ph.d(b2 @ td.ts['k-1']), ph.d(b2 @ td.ts['k'])],
                         ['+', '+'],
                         factors=[1/2, 1/2])
    wf = wf.derive.split('1-0', 'f0',
                         [b2 @ td.ts['k'], b2 @ td.ts['k-1']],
                         ['+', '-'],
                         factors=[1/dt, 1/dt])
    wf = wf.derive.split('1-2', 'f0',
                         [a3 @ td.ts['k-1'], a3 @ td.ts['k']],
                         ['+', '+'],
                         factors=[1/2, 1/2])
    wf = wf.derive.rearrange(
        {
            0: '0, 3 = 1, 2',
            1: '3, 0 = 2, 1, 4',
        }
    )

    ph.space.finite(3)

    ap = wf.ap

    wf.pr()

    # wf.pr()
    # print(wf.elementary_forms)
