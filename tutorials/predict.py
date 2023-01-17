from arenets.arekit.common.data_type import DataType
from arenets.arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arenets.core.predict.provider.id_and_binary_labels import IdAndBinaryLabelsPredictProvider
from arenets.core.writer.csv_writer import CsvContentWriter
from arenets.quickstart.predict import predict
from arenets.enum_name_types import ModelNames
from nn_config import modify_config


predict(input_data_dir="_data",                              # Input data where all the information required for input is stored.
        output_dir="_out",                                   # Directory to save the results.
        labels_count=3,                                      # Task labels count.
        bags_per_minibatch=32,                               # Batch size
        model_name=ModelNames.CNN,                           # Name (enum type) from the list of the predefined models.
        data_type=DataType.Test,                             # Data to be tested.
        modify_config_func=modify_config,
        predict_provider=IdAndBinaryLabelsPredictProvider(),
        predict_writer=CsvContentWriter(),
        reader=JsonlReader(),
        unknown_term_index=0)
