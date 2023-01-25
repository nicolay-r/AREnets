## Input Data Readers

For the `train.py` and `predict.py` scripts it is possible setup readers:

* `JSONL` utilized by default:
```python
from arenets.arekit.contrib.utils.data.readers.jsonl import JsonlReader
reader = JsonlReader()
```

* `CSV` reader: requires installation of the `pandas`:
```python
from arenets.external.readers.pandas_csv_reader import PandasCsvReader
reader = PandasCsvReader(sep=',',                   # Column separator
                         target_extension=".csv")   # Input file type
```