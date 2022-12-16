import numpy as np
import pandas as pd

from arenets.arekit.common.data.storages.base import BaseRowsStorage


class PandasBasedRowsStorage(BaseRowsStorage):
    """ Storage Kernel functions implementation, based on the pandas DataFrames.
    """

    def __init__(self, df=None):
        assert(isinstance(df, pd.DataFrame) or df is None)
        self._df = df

    @staticmethod
    def __create_empty(cols_with_types):
        """ cols_with_types: list of pairs ("name", dtype)
        """
        assert(isinstance(cols_with_types, list))
        data = np.empty(0, dtype=np.dtype(cols_with_types))
        return pd.DataFrame(data)

    def __filter(self, column_name, value):
        return self._df[self._df[column_name] == value]

    @staticmethod
    def __iter_rows_core(df):
        assert(isinstance(df, pd.DataFrame))
        for row_index, row in df.iterrows():
            yield row_index, row

    def __fill_with_blank_rows(self, row_id_column_name, rows_count):
        assert(isinstance(row_id_column_name, str))
        assert(isinstance(rows_count, int))
        self._df[row_id_column_name] = list(range(rows_count))
        self._df.set_index(row_id_column_name, inplace=True)

    # region protected methods

    def _iter_rows(self):
        for row_index, row in self.__iter_rows_core(self._df):
            yield row_index, row

    def _get_rows_count(self):
        return len(self._df)

    # endregion

    # region public methods

    def get_row(self, row_index):
        return self._df.iloc[row_index]

    def iter_column_values(self, column_name, dtype=None):
        values = self._df[column_name]
        if dtype is None:
            return values
        return values.astype(dtype)

    # endregion
