from os.path import join

from arenets.arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arenets.core.callback.hidden import HiddenStatesWriterCallback
from arenets.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arenets.core.callback.train_limiter import TrainingLimiterCallback
from arenets.enum_name_types import ModelNames
from arenets.np_utils.writer import NpzDataWriter
from arenets.quickstart.train import train
from nn_config import modify_config


input_data_dir = "_data"
model_name = ModelNames.CNN

train(input_data_dir=input_data_dir,      # Input data where all the information required for input is stored.
      labels_count=3,                     # Amount of labels in the classification task.
      model_name=model_name,              # Name (enum type) from the list of the predefined models.
      epochs_count=20,                    # Total amount of epochs to consider for training.
      bags_per_minibatch=32,              # Batch size
      learning_rate=0.01,                 # Task learning rate.
      reader=JsonlReader(),               # Input format: JSONL by default.
      modify_config_func=modify_config,
      callbacks=[
          TrainingLimiterCallback(train_acc_limit=1.0),
          # HiddenStatesWriterCallback(log_dir=join(input_data_dir, model_name.value, "hidden"), writer=NpzDataWriter()),
          # InputHiddenStatesWriterCallback(log_dir=join(input_data_dir, model_name.value, "hidden"), writer=NpzDataWriter())
      ],
      unknown_term_index=0)

