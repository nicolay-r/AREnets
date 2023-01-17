import sys
import json

sys.path.append('../../')

from arenets.external.readers.pandas_csv_reader import PandasCsvReader

reader = PandasCsvReader()
for data in ["train", "test"]:
    storage = reader.read("sample-{}.tsv.gz".format(data))
    with open("sample-{}.jsonl".format(data), "w") as f:
        for _, row in storage:
            del row["s_ind"]
            del row["t_ind"]
            del row["doc_id"]
            del row["sent_ind"]
            del row["entities"]
            del row["entity_types"]
            del row["entity_values"]
            f.write("{}\n".format(json.dumps(row, ensure_ascii=False)))