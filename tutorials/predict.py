from arenets.arekit.common.data_type import DataType
from arenets.arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arenets.core.predict.provider.id_and_binary_labels import IdAndBinaryLabelsPredictProvider
from arenets.core.writer.csv_writer import CsvContentWriter
from arenets.quickstart.predict import predict
from arenets.enum_name_types import ModelNames
from nn_config import modify_config


predict(input_data_dir="_data",
        output_dir="_out",
        labels_count=3,
        bags_per_minibatch=32,
        model_name=ModelNames.CNN,
        data_type=DataType.Test,
        modify_config_func=modify_config,
        predict_provider=IdAndBinaryLabelsPredictProvider(),
        predict_writer=CsvContentWriter(),
        reader=JsonlReader(),
        unknown_term_index=0)
