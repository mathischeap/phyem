# -*- coding: utf-8 -*-
r"""
"""

import sys

if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
import traceback
from src.config import _parse_lin_repr
from src.config import _abstract_time_sequence_default_sym_repr
from src.config import _abstract_time_interval_default_sym_repr
from src.config import _abstract_time_sequence_default_lin_repr
from src.config import _check_sym_repr
from src.form.parameters import constant_scalar

_global_abstract_time_sequence = dict()
_global_abstract_time_interval = dict()


def abstract_time_sequence():
    """A wrapper of AbstractTimeSequence"""
    return AbstractTimeSequence()


class AbstractTimeSequence(Frozen):
    """"""

    def __init__(self):
        """"""
        num = str(len(_global_abstract_time_sequence))
        base_sym_repr = _abstract_time_sequence_default_sym_repr
        if num == '0':
            sym_repr = base_sym_repr
            lin_repr = _abstract_time_sequence_default_lin_repr
        else:
            sym_repr = base_sym_repr + '_{' + num + '}'
            lin_repr = _abstract_time_sequence_default_lin_repr + num
        sym_repr = _check_sym_repr(sym_repr)
        lin_repr, self._pure_lin_repr = _parse_lin_repr('abstract_time_sequence', lin_repr)
        self._sym_repr = sym_repr
        self._lin_repr = lin_repr
        _global_abstract_time_sequence[lin_repr] = self
        self._object = None
        self._my_abstract_time_instants = dict()  # cache all abstract time instants of this abstract time sequence.
        self._my_abstract_time_interval = dict()  # cache all abstract time intervals of this abstract time sequence.
        self._freeze()

    def specify(self, class_id, *args, **kwargs):
        """specify to a particular time sequence."""
        assert self._object is None, f"specific time sequence existing, we cannot replace it."
        assert class_id in _implemented_specific_time_sequences, f"Time sequence {class_id} is not implemented yet."
        class_body = _implemented_specific_time_sequences[class_id]
        self._object = class_body(*args, **kwargs)

    def __getitem__(self, k):
        """return t[k], not return t=k."""
        assert isinstance(k, str) and ' ' not in k, f"Can only set abstract time instant with str of no space."
        lin_repr = self._pure_lin_repr + r"[" + k + "]"
        lin_repr, pure_lin_repr = _parse_lin_repr('abstract_time_instant', lin_repr)
        if lin_repr in self._my_abstract_time_instants:
            return self._my_abstract_time_instants[lin_repr]
        else:
            ati = AbstractTimeInstant(self, k, lin_repr, pure_lin_repr)
            self._my_abstract_time_instants[lin_repr] = ati
            return ati

    def __repr__(self):
        """customized repr."""
        super_repr = super().__repr__().split('object')[1]
        return f"<AbstractTimeSequence" + super_repr

    def __eq__(self, other):
        """=="""
        return self is other

    def info(self):
        """Info myself in the console."""
        if self._object is None:
            print('abstract' + self._lin_repr)
        else:
            self._object.info()

    @staticmethod
    def _is_abstract_time_sequence():
        """A private tag."""
        return True

    def make_time_interval(self, ks, ke, sym_repr=None):
        """

        Parameters
        ----------
        ks
        ke
        sym_repr :
            The symbolic representation of the abstract time interval.

        Returns
        -------

        """
        if ks.__class__ is AbstractTimeInstant:
            ts = ks
        else:
            ts = self[ks]
        if ke.__class__ is AbstractTimeInstant:
            te = ke
        else:
            te = self[ke]

        lin_repr = self._pure_lin_repr + r"[" + ts.k + "," + te.k + "]"
        lin_repr, pure_lin_repr = _parse_lin_repr('abstract_time_interval', lin_repr)
        if lin_repr in self._my_abstract_time_interval:
            assert sym_repr is None, \
                f"The abstract time interval from {ts} to {te} exists, pls do not provide sym_repr!"
            return self._my_abstract_time_interval[lin_repr]
        else:
            ati = AbstractTimeInterval(ts, te, lin_repr, pure_lin_repr, sym_repr=sym_repr)
            self._my_abstract_time_interval[lin_repr] = ati
            return ati

    def pr(self, *args, **kwargs):
        """We print this time sequence."""
        assert self._object is not None, f"to print a time sequence, first specify it to a particular one."
        return self._object.pr(*args, **kwargs)


class TimeSequence(Frozen):
    """"""

    def __init__(self):
        self._t_0 = None
        self._t_max = None
        self._freeze()

    @property
    def t_0(self):
        return self._t_0

    @property
    def t_max(self):
        return self._t_max

    @staticmethod
    def _is_specific_time_sequence():
        """A private tag."""
        return True

    def __getitem__(self, k):
        """return t[k]"""
        raise NotImplementedError()

    @staticmethod
    def _is_time_sequence():
        """A private tag."""
        return True


class ConstantTimeSequence(TimeSequence):
    """Steps are all equal.

    """

    def __init__(self, t0_max_n, factor):
        """

        Parameters
        ----------
        t0_max_n
        factor
        """
        super().__init__()
        assert len(t0_max_n) == 3, f"I need a tuple of three numbers."
        t0, t_max, n = t0_max_n
        # n is equal to the number of time intervals between t0 and t_max.
        assert t_max > t0 and n % 1 == 0 and n > 0
        assert factor % 1 == 0 and factor > 0, f"`factor` needs to be a positive integer."

        self._t_0 = t0
        self._t_max = t_max
        self._melt()
        self._factor = factor  # in each step, we have factor - 1 intermediate time instances.
        self._dt = (t_max - t0) * factor / n
        self._k_max = n / factor
        assert self._k_max % 1 == 0 and self._k_max > 0, \
            (f"max time step must be positive integer. now it is {self._k_max}, "
             f"pls check inputs for constant time sequence.")
        self._k_max = int(self._k_max)
        self._n = n
        self._allowed_reminder = [round(1*i/factor, 8) for i in range(factor)]
        self._freeze()

    @property
    def dt(self):
        """time interval between tk and tk+1."""
        return self._dt

    @property
    def k_max(self):
        """the max valid k for t[k]."""
        return self._k_max

    def __getitem__(self, k):
        """return t[k], not return t=k.

        examples
        --------

            >>> t = ConstantTimeSequence([0, 5, 15], 3)
            >>> t = t[1+1/3]
            >>> print(t)  # doctest: +ELLIPSIS
            <TimeInstant t=1.3333...

        """
        assert isinstance(k, (int, float)), f"specific time sequence can not use number for time instant."
        time = self.t_0 + k * self._dt
        remainder = round(k % 1, 8)
        if time < self.t_0 - self._dt:  # leave a dt as lower margin
            raise TimeInstantError(
                f"t[{k}] = {time} is lower than t0:{self.t_0} - dt:{self._dt}.")
        elif time > self.t_max + self._dt:  # leave a dt as upper margin
            raise TimeInstantError(
                f"t[{k}] = {time} is higher than t_max: {self.t_max} + dt:{self._dt}.")
        elif remainder not in self._allowed_reminder:
            raise TimeInstantError(
                f"t[{k}] = {time} is not a valid time instance of the sequence.")
        else:
            return TimeInstant(time)

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<ConstantTimeSequence ({self.t_0}, {self.t_max}, {self._n}) " \
               f"@ k_max={self._k_max}, dt={self._dt}, factor={self._factor}" + \
            super_repr

    def info(self):
        """info myself in the console."""
        print(f" =constant= {self._t_0} -> ... -> {self._k_max} * " +
              f"%.5f -> ... -> {self.t_max}." % self._dt)

    def pr(self, obj=None):
        """print this constant interval time sequence together with an object."""
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
        })
        matplotlib.use('TkAgg')

        # ----------- we make data of the time sequence ----------
        time_instant_hierarchy = {}
        k_sequence = range(1, self._k_max+1)  # this is important, not range(0, k_max).
        time_instant_hierarchy[0] = np.array(
            range(0, self._k_max+1)
        ) * self._dt
        if self._factor == 1:
            pass
        else:
            array = np.array(
                [k_sequence]
            ).T - 1
            array = array * self._dt

            dt = self._dt / self._factor
            intermediate_time_instants = list()
            for i in range(1, self._factor):
                new_col = array + dt * i
                intermediate_time_instants.append(
                    new_col
                )
            intermediate_time_instants = np.hstack(intermediate_time_instants)
            intermediate_time_instants = np.vstack([
                intermediate_time_instants[0] - self._dt,  # thw lower margin
                intermediate_time_instants,
                intermediate_time_instants[-1] + self._dt,  # the upper margin
            ])
            time_instant_hierarchy[1] = intermediate_time_instants.ravel('C')

        major_nodes = time_instant_hierarchy[0]

        if len(major_nodes) > 6:
            print('Constant time sequence plot warning: too many time instants to be plotted, '
                  f'the figure will be messy, plotting cancelled. Reduce time steps {len(major_nodes)-1} '
                  f'to <=5.')
            return
        else:
            pass

        if obj is None:
            # we just plot the time_instant_hierarchy.
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.tick_params(left=False,
                            right=False,
                            labelleft=False,
                            labelbottom=False,
                            bottom=False)
            vertical_length = (self._t_max - self._t_0) * 0.02
            plt.plot(
                major_nodes[1:], 0*major_nodes[1:],
                '-s',
                color='k',
                linewidth=1.2,
            )
            plt.plot(
                [self._t_0-self._dt, self._t_max+self._dt], [0, 0],
                '-',
                color='k',
                linewidth=1.2,
            )
            right_end = self._t_max+self._dt
            plt.plot(   # start point
                [self._t_0, self._t_0], [-vertical_length, vertical_length],
                color='darkgreen',
                linewidth=1.8,
            )
            plt.plot(  # ending triangle
                [right_end-vertical_length, right_end, right_end-vertical_length],
                [vertical_length, 0, -vertical_length],
                color='k',
                linewidth=1,
            )
            for _k_, major_node in enumerate(major_nodes):
                if _k_ == 0:
                    plt.text(
                        major_node,
                        -3*vertical_length,
                        f'$t_{_k_}=%.1f$' % self._t_0,
                        c='darkgreen',
                        ha='center', va='center',
                        fontsize=15
                    )
                elif _k_ == len(major_nodes) - 1:
                    plt.text(
                        major_node,
                        -3*vertical_length,
                        f'$t_{_k_}=%.1f$' % self._t_max,
                        c='blue',
                        ha='center', va='center',
                        fontsize=15
                    )
                else:
                    plt.text(
                        major_node,
                        -3*vertical_length,
                        f'$t_{_k_}$',
                        ha='center', va='center',
                        fontsize=15
                    )
            if 1 in time_instant_hierarchy:
                minor_nodes = time_instant_hierarchy[1]
                plt.scatter(
                    minor_nodes, 0*minor_nodes, marker='x', color='red'
                )
            # ----- save -----------------------------------------------------
            plt.tight_layout()
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='constantTimeSequence')
            else:
                plt.show(block=_setting['block'])
            return fig

        else:
            assert hasattr(obj, "_pr_temporal_advancing"), \
                f"{obj} has no method `_pr_temporal_advancing`, implement it first!"
            obj._pr_temporal_advancing(
                self,
                time_instant_hierarchy
            )


class TimeInstantError(Exception):
    """Raise when we try to define new attribute for a frozen object."""


class TimeInstant(Frozen):
    """This is a time instance regardless of the sequence."""

    def __init__(self, time):
        self._t = time
        self._freeze()

    @property
    def time(self):
        return self._t

    def __call__(self):
        return self.time

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f'<TimeInstant t={self.time}' + super_repr

    def __eq__(self, other):
        return other.__class__.__name__ == 'TimeInstant' and other.time == self.time


class AbstractTimeInstant(Frozen):
    """"""

    def __init__(self, ts, k, lin_repr, pure_lin_repr):
        self._ts = ts
        k = self._parse_k(k)
        self._k = k
        sym_repr = ts._sym_repr + f'[{k}]'
        sym_repr = _check_sym_repr(sym_repr)
        self._sym_repr = sym_repr
        self._lin_repr = lin_repr
        self._pure_lin_repr = pure_lin_repr
        self._freeze()

    def _parse_k(self, k):
        """"""
        k = k.replace(' ', '')  # delete all space
        length = len(k)
        self._kwarg_keys = list()
        assert isinstance(k, str), f"abstract kwarg must be a str."
        for i, _ in enumerate(k):
            if _.isalpha():
                if i > 0:
                    assert not k[i-1].isalpha(), f"abstract time {k} illegal. A variable must contain one alpha-beta."
                if i < length - 1:
                    assert not k[i+1].isalpha(), f"abstract time {k} illegal. A variable must contain one alpha-beta."
                self._kwarg_keys.append(_)
        return k

    @property
    def time_sequence(self):
        return self._ts

    @property
    def k(self):
        return self._k

    def __eq__(self, other):
        """=="""
        return self.__class__.__name__ == other.__class__.__name__ and \
            self.time_sequence == other.time_sequence and \
            self.k == other.k

    def __call__(self, **kwargs):
        """call, return a TimeInstant object."""
        time_instance_str = self._k
        for key in kwargs:
            assert key in self._kwarg_keys, f"key={key} is not found for ati:{self}."
            time_instance_str = time_instance_str.replace(key, str(kwargs[key]))
        time = eval(time_instance_str)
        assert isinstance(time, (int, float)), f"format wrong, `eval` does not return a number."
        assert self.time_sequence._object is not None, \
            f"The abstract time sequence has no object (particular time sequence). Specify it firstly."
        try:
            ts = self.time_sequence._object[time]
        except TimeInstantError:
            tb = traceback.format_exc().split('TimeInstantError:')[1]
            key_str = ["'" + str(key) + "'" + '=' + str(kwargs[key]) for key in kwargs]
            local_error_message = f"t['{self.k}'] for {''.join(key_str)} leads to"
            raise TimeInstantError(local_error_message + tb)
        return ts

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<AbstractTimeInstant t['{self.k}']" + super_repr[:-1] + f" of {self.time_sequence}>"


class TimeInterval(Frozen):
    """Delta t"""

    def __init__(self, t_start, t_end):
        """

        Parameters
        ----------
        t_start : TimeInstant
            The start time.
        t_end : TimeInstant
            The end time.
        """
        assert t_start.__class__.__name__ == 'TimeInstant' and t_end.__class__.__name__ == 'TimeInstant', \
            f"t_start and t_end must be `TimeInstant` instances."
        self._t_start = t_start
        self._t_end = t_end
        self._dt = t_end() - t_start()
        assert self._dt > 0, f"time interval must be positive."
        self._freeze()

    @property
    def start(self):
        return self._t_start

    @property
    def end(self):
        return self._t_end

    def __call__(self):
        """dt"""
        return self._dt

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f'<TimeInterval {self()} from t={self.start()} to t={self.end()}' + super_repr


class AbstractTimeInterval(Frozen):
    """"""

    def __init__(self, t_start, t_end, lin_repr, pure_lin_repr, sym_repr=None):
        """

        Parameters
        ----------

        Parameters
        ----------
        t_start :
            The start abstract time instant.
        t_end :
            The end abstract time instant.
        lin_repr :
        pure_lin_repr :
        sym_repr :

        """
        assert t_start.__class__.__name__ == 'AbstractTimeInstant' and \
            t_end.__class__.__name__ == 'AbstractTimeInstant', \
            f"t_start and t_end must be `AbstractTimeInstant` instances."
        ts0 = t_start.time_sequence
        ts1 = t_end.time_sequence
        assert ts0 is ts1, f"time sequences of t_start, t_end does not match."
        self._ts = ts0
        self._t_start = t_start
        self._t_end = t_end
        self._lin_repr = lin_repr
        self._pure_lin_repr = pure_lin_repr
        num = len(_global_abstract_time_interval)
        base_sym_repr = _abstract_time_interval_default_sym_repr
        if sym_repr is None:
            if num == 0:
                sym_repr = base_sym_repr
            else:
                sym_repr = base_sym_repr + r"_{" + str(num) + r"}"
        else:
            pass
        sym_repr = _check_sym_repr(sym_repr)
        for _ in _global_abstract_time_interval:
            exist_sym_repr = _global_abstract_time_interval[_]
            assert sym_repr != exist_sym_repr, f"sym_repr = {exist_sym_repr} exist, use a new one!"
        _global_abstract_time_interval[num] = sym_repr
        self._sym_repr = sym_repr
        self._s = None
        self._freeze()

    @property
    def time_sequence(self):
        return self._ts

    @property
    def start(self):
        return self._t_start

    @property
    def end(self):
        return self._t_end

    def __call__(self, **kwargs):
        """dt, return a TimeInterval instance."""
        ts = self.start(**kwargs)
        te = self.end(**kwargs)
        return TimeInterval(ts, te)

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<AbstractTimeInterval from t['{self.start.k}]' to t['{self.end.k}'], {self._sym_repr}," + \
            super_repr[:-1] + f' of {self.time_sequence}>'

    def _as_scalar(self):
        if self._s is None:
            ati_sr = self._sym_repr
            ati_lr = self._pure_lin_repr
            self._s = constant_scalar(ati_sr, ati_lr)
            self._s._value = self.__call__
        return self._s

    def __rtruediv__(self, other):
        """other / self"""
        if isinstance(other, (int, float)):
            return constant_scalar(other) / self._as_scalar()
        else:
            return other / self._as_scalar()


_implemented_specific_time_sequences = {
    'constant': ConstantTimeSequence,
}


if __name__ == '__main__':
    # python src/time_sequence.py
    from doctest import testmod
    testmod()

    # ct = ConstantTimeSequence([0, 100, 100], 2)
    # t0 = ct[0]
    # t1 = ct[1]
    # ti = TimeInterval(t0, t1)
    # print(ti.start, ti.end, ti)

    at = AbstractTimeSequence()
    t0 = at['2']
    t1 = at['k']
    # ti = AbstractTimeInterval(t0, t1)
    # at.specify('constant', [0, 100, 100], 2)
    # # for k in range(1,10):
    # print(ti.start(k=1), ti.end(k=1), ti(k=1))

    # ti = at.make_time_interval('k+0', 'k+0.5')
    at.specify('constant', [0, 100, 100], 2)
    # for k in range(1,10):

    print(t0, t0._kwarg_keys, t0()())

    # print(ti.start(k=1)(), ti(k=1)(), t1(k=50)())
