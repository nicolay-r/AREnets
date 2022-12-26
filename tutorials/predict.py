from arenets.arekit.common.data_type import DataType
from arenets.context.configurations.base.base import DefaultNetworkConfig
from arenets.quickstart.predict import predict
from arenets.enum_name_types import ModelNames


def modify_config(config):
    assert(isinstance(config, DefaultNetworkConfig))
    config.modify_terms_per_context(50)


predict(input_data_dir="_data", output_dir="_out",
        labels_count=3, model_name=ModelNames.RCNN, data_type=DataType.Dev,
        modify_config_func=modify_config)
