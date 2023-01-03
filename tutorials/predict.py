from arenets.arekit.common.data_type import DataType
from arenets.quickstart.predict import predict
from arenets.enum_name_types import ModelNames
from nn_config import modify_config


predict(input_data_dir="_data",
        output_dir="_out",
        labels_count=2,
        bags_per_minibatch=32,
        model_name=ModelNames.CNN,
        data_type=DataType.Test,
        modify_config_func=modify_config,
        unknown_term_index=0)
