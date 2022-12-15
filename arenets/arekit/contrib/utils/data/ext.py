from arenets.arekit.common.data.input.reader import BaseReader
from arenets.arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader

PANDAS_CSV_EXTENSION = ".tsv.gz"


def create_reader_extension(writer):
    assert(isinstance(writer, BaseReader))

    if isinstance(writer, PandasCsvReader):
        return PANDAS_CSV_EXTENSION

    raise NotImplementedError()
