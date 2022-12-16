from arenets.arekit.common.data.input.reader import BaseReader
from arenets.arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader


def create_reader_extension(writer):
    assert(isinstance(writer, BaseReader))

    if isinstance(writer, PandasCsvReader):
        return ".tsv.gz"

    raise NotImplementedError()
