### Inference Output Customization

According to the inference quickstart call, which may look like this:

```python
from arenets.arekit.common.data_type import DataType
from arenets.arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arenets.core.predict.provider.id_and_binary_labels import IdAndBinaryLabelsPredictProvider
from arenets.core.writer.csv_writer import CsvContentWriter
from arenets.quickstart.predict import predict
from arenets.enum_name_types import ModelNames

predict(input_data_dir="_data", output_dir="_out",
        labels_count=3, bags_per_minibatch=32, unknown_term_index=0,
        model_name=ModelNames.CNN, data_type=DataType.Test, reader=JsonlReader(),
        ###############################################
        # Parameters below are responsible for output:
        ###############################################
        predict_provider=IdAndBinaryLabelsPredictProvider(),
        predict_writer=CsvContentWriter())
```
Predict Paramaters that might be manually implemented are as follows:
* Provider -- responsible for title and rows content is expected to be provided for writing, including
`row_id` and `labels`; we consider `IdAndBinaryLabelsPredictProvider` by default;
* Writer -- formatter of the provided rows; 
  by default we consider gzipped version of the csv, and `CsvContentWriter` class;
  