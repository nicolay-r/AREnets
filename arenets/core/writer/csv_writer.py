import logging

from arenets.arekit.common.utils import create_dir_if_not_exists, progress_bar_iter
from arenets.core.writer.base_writer import BaseIterativeWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CsvContentWriter(BaseIterativeWriter):
    """ This writer saves information in a CSV-based format.
    """

    def __init__(self, separator=',', write_title=True):
        super(CsvContentWriter, self).__init__()
        self.__col_separator = separator
        self.__write_title = write_title
        self.__f = None

    def __write(self, params):
        line = "{}\n".format(self.__col_separator.join([str(p) for p in params]))
        self.__f.write(line)

    def write(self, title, contents_it):

        # Save title optionally.
        if self.__write_title:
            self.__write(title)

        wrapped_it = progress_bar_iter(iterable=contents_it,
                                       desc='Writing output',
                                       unit='rows')

        # Save contents.
        for contents in wrapped_it:
            self.__write(contents)

    # region base

    def __enter__(self):
        create_dir_if_not_exists(self._target)
        self.__f = open(self._target, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__f.close()
        logger.info("Saved: {}".format(self._target))

    # endregion
