from arenets.arekit.common.data_type import DataType
from arenets.quickstart.predict import predict
from arenets.enum_name_types import ModelNames

predict(input_data_dir="_data", output_dir="_out", labels_count=3, model_name=ModelNames.CNN, data_type=DataType.Test)
