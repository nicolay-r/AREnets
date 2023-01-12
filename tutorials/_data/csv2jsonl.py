import sys
import json

sys.path.append('../../')

from arenets.external.readers.pandas_csv_reader import PandasCsvReader

reader = PandasCsvReader()
for data in ["train", "test"]:
    storage = reader.read("sample-{}.tsv.gz".format(data))
    with open("sample-{}.jsonl".format(data), "w") as f:
        for _, row in storage:
            f.write("{}\n".format(json.dumps(row, ensure_ascii=False)))