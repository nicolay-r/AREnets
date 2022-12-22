import logging

logger = logging.getLogger(__name__)


class BaseRowsStorage(object):

    # region abstract methods

    def _iter_rows(self):
        raise NotImplemented()

    def _get_rows_count(self):
        raise NotImplemented()

    # endregion

    # endregion

    # region base methods

    def __iter__(self):
        return self._iter_rows()

    def __len__(self):
        return self._get_rows_count()

    # endregion
