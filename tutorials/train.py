from os.path import join

from arenets.context.configurations.base.base import DefaultNetworkConfig
from arenets.core.callback.hidden import HiddenStatesWriterCallback
from arenets.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arenets.core.callback.train_limiter import TrainingLimiterCallback
from arenets.enum_name_types import ModelNames
from arenets.np_utils.writer import NpzDataWriter
from arenets.quickstart.train import train


def modify_config(config):
    assert(isinstance(config, DefaultNetworkConfig))
    config.modify_terms_per_context(50)


input_data_dir = "_data"
model_name = ModelNames.RCNNAttPZhou

train(input_data_dir=input_data_dir, labels_count=3,
      model_name=model_name,
      epochs_count=10,
      bags_per_minibatch=32,
      learning_rate=0.01,
      modify_config_func=modify_config,
      callbacks=[
          TrainingLimiterCallback(train_acc_limit=1.0),
          HiddenStatesWriterCallback(log_dir=join(input_data_dir, model_name.value, "hidden"), writer=NpzDataWriter()),
          InputHiddenStatesWriterCallback(log_dir=join(input_data_dir, model_name.value, "hidden"), writer=NpzDataWriter())
      ],
      unknown_term_index=0)

