from arenets.enum_name_types import ModelNames
from arenets.quickstart.train import train

train(input_data_dir="_data", labels_count=3, model_name=ModelNames.CNN, epochs_count=10, train_acc_limit=1.1)
