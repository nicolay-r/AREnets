from arenets.arekit.common.data.input.reader import BaseReader
from arenets.arekit.contrib.utils.data.storages.jsonl_based import JsonBasedRowsStorage


class JsonlReader(BaseReader):

    def read(self, target):
        rows = []
        with open(target, "r") as f:
            for line in f.readlines():
                rows.append(line)
        return JsonBasedRowsStorage(rows)

    def target_extension(self):
        return ".jsonl"
