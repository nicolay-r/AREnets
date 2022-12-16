import logging

logger = logging.getLogger(__name__)


class BaseRowsStorage(object):

    # region abstract methods

    def _iter_rows(self):
        raise NotImplemented()

    def _get_rows_count(self):
        raise NotImplemented()

    def iter_column_values(self, column_name, dtype=None):
        raise NotImplemented()

    def get_row(self, row_index):
        raise NotImplemented()

    # endregion

    # endregion

    # region base methods

    def __iter__(self):
        return self._iter_rows()

    def __len__(self):
        return self._get_rows_count()

    # endregion
