# -*- coding: utf-8 -*-
r"""
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
import pandas as pd

from tools.miscellaneous.csv.visualize.main import CsvFilerVisualize
from src.config import SIZE


class CsvFiler(Frozen):
    """We'd better do not use this class in parallel, although it works."""
    def __init__(self, csv_filename: str, header='infer'):
        assert SIZE == 1, f"csvFiler better not work in parallel."
        assert isinstance(csv_filename, str), "csv filename must be a str."
        if csv_filename[-4:] != '.csv':
            csv_filename += '.csv'
        # noinspection PyTypeChecker
        self._df_ = pd.read_csv(csv_filename, index_col=0, header=header)
        self._visualize_ = CsvFilerVisualize(self)
        self._freeze()

    @property
    def df(self):
        return self._df_

    @property
    def columns(self):
        return self._df_.columns

    @property
    def visualize(self):
        return self._visualize_

    def drop(self, index):
        """delete a row indexed `index`.

        Parameters
        ----------
        index

        Returns
        -------

        """
        self._df_ = self.df.drop(index)


if __name__ == '__main__':
    # mpiexec -n 1 python tools/miscellaneous/csv/main.py
    import os
    current_dir = os.path.dirname(__file__)
    csv = CsvFiler(current_dir + '/csv_test')
    csv.drop(0)
    csv.df['enstrophy'] = csv.df['enstrophy'] - csv.df.loc[1, 'enstrophy']

    csv.visualize.plot(
        't', 'enstrophy', style='-',
        yticks=[-1e-13, 0, 1e-13],
        xlabel=r"$t$",
        ylabel=r'$\mathcal{E}^h$',
    )
