from os.path import join

from arenets.arekit.common.data_type import DataType
from arenets.arekit.common.pipeline.base import BasePipeline
from arenets.arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arenets.arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arenets.arekit.contrib.utils.io_utils.samples import SamplesIO
from arenets.core.callback.hidden import HiddenStatesWriterCallback
from arenets.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arenets.core.callback.stat import TrainingStatProviderCallback
from arenets.core.callback.train_limiter import TrainingLimiterCallback
from arenets.core.feeding.bags.collection.single import SingleBagsCollection
from arenets.core.model_io import TensorflowNeuralNetworkModelIO
from arenets.enum_input_types import ModelInputType
from arenets.enum_name_types import ModelNames
from arenets.factory import create_network_and_network_config_funcs
from arenets.np_utils.writer import NpzDataWriter
from arenets.pipelines.items.training import NetworksTrainingPipelineItem


def train(input_data_dir, labels_count, model_save_dir=None, model_log_dir=None,
          modify_config_func=None, model_name_suffix="model",
          vocab_filename="vocab.txt", embedding_npz_filename="term_embedding.npz",
          epochs_count=100, model_name=ModelNames.CNN,
          bags_per_minibatch=32, bag_size=1, terms_per_context=50,
          learning_rate=0.01, embedding_dropout_keep_prob=1.0,
          dropout_keep_prob=0.9, train_acc_limit=0.99,
          part_of_speech_types_count=100):
    """
        modify_config_func: func of None
            allows to declare and provide your function which modifies the contents of the config.
        model_save_dir: str
            Where to keep the Tensorflow-based serialized model state.
        model_log_dir: str
            Where to keep logging details during the model process training.
    """
    assert(callable(modify_config_func) or modify_config_func is None)
    assert(isinstance(input_data_dir, str))
    assert(isinstance(model_save_dir, str) or model_save_dir is None)
    assert(isinstance(model_log_dir, str) or model_log_dir is None)

    model_save_dir = input_data_dir if model_save_dir is None else model_save_dir
    model_log_dir = input_data_dir if model_log_dir is None else model_log_dir

    full_model_name = "-".join([model_name.value, model_name_suffix])
    model_target_dir = join(model_log_dir, full_model_name)
    model_io = TensorflowNeuralNetworkModelIO(model_name=full_model_name,
                                              target_dir=model_save_dir)
    data_writer = NpzDataWriter()

    network_func, network_config_func = create_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=ModelInputType.SingleInstance)

    network_callbacks = [
        TrainingLimiterCallback(train_acc_limit=train_acc_limit),
        TrainingStatProviderCallback(),
        HiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer),
        InputHiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer)
    ]

    # Configuration initialization.
    config = network_config_func()
    config.modify_classes_count(value=labels_count)
    config.modify_learning_rate(learning_rate)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(dropout_keep_prob)
    config.modify_bag_size(bag_size)
    config.modify_bags_per_minibatch(bags_per_minibatch)
    config.modify_embedding_dropout_keep_prob(embedding_dropout_keep_prob)
    config.modify_terms_per_context(terms_per_context)
    config.modify_use_entity_types_in_embedding(False)
    config.set_pos_count(part_of_speech_types_count)

    # Custom configuration modification function.
    if modify_config_func is not None:
        modify_config_func(config)

    pipeline_item = NetworksTrainingPipelineItem(
        load_model=True,
        model_io=model_io,
        labels_count=labels_count,
        create_network_func=network_func,
        samples_io=SamplesIO(target_dir=input_data_dir, reader=PandasCsvReader()),
        emb_io=NpEmbeddingIO(target_dir=input_data_dir,
                             vocab_filename=vocab_filename,
                             embedding_npz_filename=embedding_npz_filename),
        config=config,
        bags_collection_type=SingleBagsCollection,
        network_callbacks=network_callbacks,
        training_epochs=epochs_count)

    ppl = BasePipeline([pipeline_item])
    ppl.run(None, params_dict={"supported_data_types": [DataType.Train],
                               "data_type": DataType.Train})
