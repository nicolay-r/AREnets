import unittest
from os.path import join, dirname

from arenets.arekit.common.data.row_ids.base import BaseIDProvider
from arenets.arekit.common.data.views.samples import LinkedSamplesStorageView
from arenets.external.readers.pandas_csv_reader import PandasCsvReader


class TestSamplesStorageView(unittest.TestCase):

    def __get_local_dir(self, local_filepath):
        return join(dirname(__file__), local_filepath)

    def test(self):
        samples_filepath = self.__get_local_dir("test_data/sample-train.tsv.gz")
        reader = PandasCsvReader()
        storage = reader.read(samples_filepath)
        samples_view = LinkedSamplesStorageView(row_ids_provider=BaseIDProvider())
        for data in samples_view.iter_from_storage(storage):
            print(type(data))