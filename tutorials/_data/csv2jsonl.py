import sys
import json

from tests.readers.pandas_csv_reader import PandasCsvReader

sys.path.append('../')

reader = PandasCsvReader()
for data in ["train", "test"]:
    storage = reader.read("_data/sample-{}-0.tsv.gz".format(data))
    with open("_data/sample-{}-0.jsonl".format(data), "w") as f:
        for _, row in storage:
            f.write("{}\n".format(json.dumps(row, ensure_ascii=False)))