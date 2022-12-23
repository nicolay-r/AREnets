import pandas as pd

from arenets.arekit.common.data.storages.base import BaseRowsStorage


class PandasBasedRowsStorage(BaseRowsStorage):
    """ Storage Kernel functions implementation, based on the pandas DataFrames.
    """

    def __init__(self, df=None):
        assert(isinstance(df, pd.DataFrame) or df is None)
        self._df = df

    # region protected methods

    def _iter_rows(self):
        assert(isinstance(self._df, pd.DataFrame))
        for row_index, row in self._df.iterrows():
            yield row_index, row.to_dict()

    def _get_rows_count(self):
        return len(self._df)

    # endregion
