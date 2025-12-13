# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK

if RANK == MASTER_RANK:
    import matplotlib.pyplot as plt
    import matplotlib
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
    })
    matplotlib.use('TkAgg')

else:
    pass


class MultiGridSchemeConfig(Frozen):
    r""""""

    def __init__(self, cycle_type, num_max_layers, **kwargs):
        r"""
        Parameters
        ----------
        cycle_type : str
            Like 'V,' 'N', 'W', etc.
        num_max_layers :
            The number of max layers for the scheme.
        kwargs :
            Other parameters for the particular cycle.
        """
        if cycle_type == 'V':
            self._scheme_ = self._parse_V_cycle_(num_max_layers)
        elif cycle_type == 'N':
            self._scheme_ = self._parse_N_cycle_(num_max_layers)
        elif cycle_type == 'W':
            self._scheme_ = self._parse_W_cycle_(num_max_layers, **kwargs)
        elif cycle_type == 'FMG':
            self._scheme_ = self._parse_FMG_cycle_(num_max_layers, **kwargs)
        else:
            raise NotImplementedError(f"MultiGridSchemeFile for cycle_type={cycle_type} is not implemented.")
        self.___renew_properties___(num_max_layers)
        self._freeze()

    def ___renew_properties___(self, num_max_layers):
        r""""""
        self._num_max_layers = num_max_layers
        self._pairs_ = None

    def visualize(self, saveto=None):
        r""""""
        if RANK != MASTER_RANK:
            return None
        else:
            pass
        dx = 0.1
        dy = 0.2
        X = list()
        Y = list()

        total_steps = len(self._scheme_)
        for i, step in enumerate(self._scheme_):
            x = i * dx
            y = step._layer * dy
            X.append(x)
            Y.append(y)
        last_step = self._scheme_[-1]
        if last_step.__class__ is InterpolationStep:
            X.append(len(self._scheme_) * dx)
            Y.append((last_step._layer + 1) * dy)
        else:
            raise NotImplementedError()

        x_range = [0, X[-1]]
        y_range = np.linspace(0, dy * (self._num_max_layers-1), self._num_max_layers)

        plt.rc('text', usetex=True)
        fig, ax = plt.subplots(figsize=(2 + total_steps, 1 + self._num_max_layers))
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        plt.plot(X, Y, 'o-', color='gray')

        for layer, x_line_y in enumerate(y_range):
            plt.plot(x_range, [x_line_y, x_line_y], '--', color='lightgray', linewidth=0.75)
            plt.text(-0.1 * dx, x_line_y, rf'layer\#{layer}', ha='right', va='center')

        for i, x in enumerate(X[:-1]):
            y = Y[i]
            x_next = X[i+1]
            y_next = Y[i+1]
            Xc = (x + x_next) / 2
            Yc = (y + y_next) / 2
            if self._scheme_[i].__class__ is RestrictionStep:
                color = 'red'
                text = r'\mathcal{R}'
            elif self._scheme_[i].__class__ is InterpolationStep:
                color = 'blue'
                text = r'\mathcal{I}'
            else:
                raise NotImplementedError()
            plt.text(Xc, Yc, rf'\#{i}:${text}$', color=color, ha='center', va='center')

        # save -----------------------------------------------------------
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight', pad_inches=0)
        else:
            from phyem.src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehtt_elements')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])
        plt.close()
        return None

    def __len__(self):
        r"""How many steps I have?"""
        return len(self._scheme_)

    def __iter__(self):
        r"""Go through all indices of steps. Not go through all steps. """
        for i in range(len(self)):
            yield i

    def __getitem__(self, i):
        r"""Return the `i`-th step."""
        return self._scheme_[i]

    def num_layers(self):
        r"""How many layers of meshes?"""
        return self._num_max_layers

    @property
    def pairs(self):
        r"""Produce the pairs. To find out which steps form v-cycles.

        We use the idea of stack to do this.

        For example:
            pair = {
                10: 9,      <- #10 step, an interpolation, is paired to step #9, a restriction.
                11: 8,      <- #10 step, an interpolation, is paired to step #8, a restriction.
                ...
                0: None,    <- #0 step, an interpolation, is not paired to restriction.
                1: None,    <- #1 step, an interpolation, is not paired to restriction.
            }
        where keys are indices of interpolations and values (if is not None) are the restrictions paired to
        the corresponding interpolations.

        If a value is None, then that interpolation is not paired.

        """
        if self._pairs_ is None:
            pass
        else:
            return self._pairs_

        repr_index_dict = {}
        for i in self:
            rp = self[i].__repr__()
            repr_index_dict[rp] = i

        pair_dict = {}
        stack = list()
        for step in self._scheme_[::-1]:
            if step.__class__ is InterpolationStep:
                stack.append(step)
            elif step.__class__ is RestrictionStep:
                interpolation = stack.pop()
                r_index = repr_index_dict[step.__repr__()]
                i_index = repr_index_dict[interpolation.__repr__()]
                pair_dict[i_index] = r_index

        for interpolation in stack[::-1]:
            i_index = repr_index_dict[interpolation.__repr__()]
            pair_dict[i_index] = None  # these interpolation is left unpaired.

        # noinspection PyAttributeOutsideInit
        self._pairs_ = pair_dict

        return self._pairs_

    @classmethod
    def _parse_V_cycle_(cls, num_max_layers):
        r""" Like if `num_max_layers = 4`, the cycle is:

            x-----------x          <- top layer, normally the most refined mesh.
             \         /
              x       x
               \     /
                x   x
                 \ /
                  x               <- bottom layer, normally the coarsest mesh.

        And we use `root_solver=direct` solver to solve systems on the coarsest mesh, i.e. the bottom layer on above
        diagram.

        Parameters
        ----------

        """
        scheme = []
        allowed_layers = range(num_max_layers)
        for layer in allowed_layers[::-1][:-1]:
            scheme.append(RestrictionStep(layer))
        for layer in allowed_layers[:-1]:
            scheme.append(InterpolationStep(layer))
        return scheme

    @classmethod
    def _parse_N_cycle_(cls, num_max_layers):
        r"""For example, if `num_max_layers = 4`, the cycle is:

        ------x-----------x          <- top layer, normally the most refined mesh.
             / \         /
            x   x       x
           /     \     /
          x       x   x
         /         \ /
        x           x                <- bottom layer, normally the coarsest mesh.

        And we use `root_solver=direct` solver to solve systems on the coarsest mesh, i.e. the bottom layer on above
        diagram.

        Parameters
        ----------

        """
        scheme = []
        allowed_layers = range(num_max_layers)
        for layer in allowed_layers[:-1]:
            scheme.append(InterpolationStep(layer))
        for layer in allowed_layers[::-1][:-1]:
            scheme.append(RestrictionStep(layer))
        for layer in allowed_layers[:-1]:
            scheme.append(InterpolationStep(layer))
        return scheme

    @classmethod
    def _parse_W_cycle_(cls, num_max_layers, sequences=''):
        r"""'W'-type cycle starts with the most refined mesh.

        For example, if `num_max_layers = 4, sequences = '3-1-1-2-2-3'`, the cycle is:

        x-----------------------x         <- top layer, normally the most refined mesh.
         \                     /
          x           x       x
           \         / \     /
            x   x   x   x   x
             \ / \ /     \ /
              x   x       x               <- bottom layer, normally the coarsest mesh.
          3R 1I 1R  2I 2R   3I

        where sequences = '3-1-1-2-2-3'. In a 'W'-type cycle, total restrictions must be equal to total interpolations.
                        R-I-R-I-R-I
        And it must be R -> I -> R -> I -> ... -> I. So, an interpolation follows each restriction. And the
        cycle starts with a restriction and ends with an interpolation.

        Parameters
        ----------
        sequences : str
            The sequence defines the 'W'-type cycle.

        """
        scheme = []
        sequences = sequences.split('-')
        c_layer = num_max_layers - 1
        c_type = 'R'
        for S in sequences:
            s = int(S)
            for i in range(s):
                if c_type == 'R':
                    scheme.append(RestrictionStep(c_layer))
                    c_layer -= 1
                elif c_type == 'I':
                    scheme.append(InterpolationStep(c_layer))
                    c_layer += 1

            if c_type == 'R':
                c_type = 'I'
            elif c_type == 'I':
                c_type = 'R'
            else:
                raise Exception()
        assert scheme[-1].__class__ is InterpolationStep and scheme[-1]._layer == num_max_layers - 2
        return scheme

    @classmethod
    def _parse_FMG_cycle_(cls, num_max_layers, sequences=''):
        r"""Full MultiGrid.

        'FMG'-type cycle starts with the coarsest mesh.

        For example, if `num_max_layers = 4, sequences = '3-1-1-2-2-3'`, the cycle is:

        --------------------------x         <- top layer, normally the most refined mesh.
                                 /
            x           x       x
           / \         / \     /
          x   x   x   x   x   x
         /     \ / \ /     \ /
        x       x   x       x               <- bottom layer, normally the coarsest mesh.
        2I  2R 1I 1R  2I 2R   3I

        where sequences = '2-2-1-1-2-2-3'. So total restrictions must be one less than interpolations.
                           I-R-I-R-I-R-I

        It starts with an interpolation and also end with an interpolation.

        Parameters
        ----------
        sequences : str
            The sequence defines the 'W'-type cycle.

        """
        scheme = []
        sequences = sequences.split('-')
        c_layer = 0
        c_type = 'I'
        for S in sequences:
            s = int(S)
            for i in range(s):
                if c_type == 'R':
                    scheme.append(RestrictionStep(c_layer))
                    c_layer -= 1
                elif c_type == 'I':
                    scheme.append(InterpolationStep(c_layer))
                    c_layer += 1

            if c_type == 'R':
                c_type = 'I'
            elif c_type == 'I':
                c_type = 'R'
            else:
                raise Exception()
        assert scheme[-1].__class__ is InterpolationStep and scheme[-1]._layer == num_max_layers - 2
        return scheme


class RestrictionStep(Frozen):
    r""""""
    def __init__(self, layer, rtype='regular'):
        r""""""
        self._layer = layer
        self._rtype = rtype
        self._freeze()

    def __repr__(self):
        r""""""
        super_repr_ = super().__repr__().split(' at ')[1]
        return rf"<{self._layer}>-R-<{self._layer-1}> at " + super_repr_


class InterpolationStep(Frozen):
    r""""""
    def __init__(self, layer, itype='regular'):
        r""""""
        self._layer = layer
        self._itype = itype
        self._freeze()

    def __repr__(self):
        r""""""
        super_repr_ = super().__repr__().split(' at ')[1]
        return rf"<{self._layer}>-I-<{self._layer+1}> at " + super_repr_


if __name__ == '__main__':
    scheme = MultiGridSchemeConfig('V', 4)
    for r in scheme.pairs:
        print(r, scheme.pairs[r])
    scheme.visualize()

    scheme = MultiGridSchemeConfig('N', 4)
    for r in scheme.pairs:
        print(r, scheme.pairs[r])
    scheme.visualize()

    scheme = MultiGridSchemeConfig('W', 4, sequences='3-1-1-2-2-3')
    for r in scheme.pairs:
        print(r, scheme.pairs[r])
    scheme.visualize()

    scheme = MultiGridSchemeConfig('N', 4, sequences='2-2-1-1-2-2-3')
    for r in scheme.pairs:
        print(r, scheme.pairs[r])
    scheme.visualize()
