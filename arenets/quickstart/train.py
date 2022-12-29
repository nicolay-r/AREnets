from os.path import join

from arenets.arekit.common.data_type import DataType
from arenets.arekit.common.pipeline.base import BasePipeline
from arenets.arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arenets.arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arenets.arekit.contrib.utils.io_utils.samples import SamplesIO
from arenets.core.callback.stat import TrainingStatProviderCallback
from arenets.core.feeding.bags.collection.single import SingleBagsCollection
from arenets.core.model_io import TensorflowNeuralNetworkModelIO
from arenets.emb_converter import convert_text_embedding_if_needed
from arenets.enum_input_types import ModelInputType
from arenets.enum_name_types import ModelNames
from arenets.factory import create_network_and_network_config_funcs
from arenets.pipelines.items.training import NetworksTrainingPipelineItem


def train(input_data_dir, labels_count,
          model_dir=None, model_hstates_dir=None,
          model_input_type=ModelInputType.SingleInstance,
          modify_config_func=None, unknown_term_index=0,
          word2vec_txt_model_name="model.txt",
          reader=JsonlReader(), callbacks=None,
          epochs_count=100, model_name=ModelNames.CNN,
          bags_collection_type=SingleBagsCollection,
          bags_per_minibatch=32, bag_size=1, terms_per_context=50,
          learning_rate=0.01, embedding_dropout_keep_prob=1.0,
          dropout_keep_prob=0.9, part_of_speech_types_count=100):
    """
        model_input_type: enum
            Optional wrap over context-based network core which allows to consider multiple contexts as a single one.
            default: SingleInstance
        word2vec_txt_model_name: str
            this is a filename that declares word2vec model, saved as a text file, incuding vocabulary
            and vectors for every term in particular.
        modify_config_func: func of None
            allows to declare and provide your function which modifies the contents of the config.
        model_save_dir: str
            Where to keep the Tensorflow-based serialized model state.
        model_hstates_dir: str
            Where to keep hidden states during the model process training.
        bags_collection_type: enum
            How data is presented; for singe instance it denotes we deal with sequence of bags, while
            for multi-instance type, every bag contains a list of bags.
    """
    assert(callable(modify_config_func) or modify_config_func is None)
    assert(isinstance(input_data_dir, str))
    assert(isinstance(model_input_type, ModelInputType))
    assert(isinstance(model_dir, str) or model_dir is None)
    assert(isinstance(model_hstates_dir, str) or model_hstates_dir is None)
    assert(isinstance(callbacks, list) or callbacks is None)
    assert(isinstance(unknown_term_index, int))

    # Setup parameters.
    model_dir = input_data_dir if model_dir is None else model_dir
    callbacks = [] if callbacks is None else callbacks

    # Declaring embedding input/output parameters.
    embedding_io = NpEmbeddingIO(target_dir=input_data_dir,
                                 vocab_filename="vocab.txt",
                                 unknown_ind=unknown_term_index,
                                 embedding_npz_filename="term_embedding.npz")

    if not embedding_io.check_targets_existed():
        convert_text_embedding_if_needed(txt_embedding_filepath=join(input_data_dir, word2vec_txt_model_name),
                                         embedding_io=embedding_io)

    model_io = TensorflowNeuralNetworkModelIO(model_name=model_name.value,
                                              target_dir=model_dir)

    network_func, network_config_func = create_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=model_input_type)

    callbacks += [
        TrainingStatProviderCallback(),
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
        model_io=model_io,
        labels_count=labels_count,
        create_network_func=network_func,
        samples_io=SamplesIO(target_dir=input_data_dir, reader=reader),
        emb_io=embedding_io,
        config=config,
        bags_collection_type=bags_collection_type,
        network_callbacks=callbacks,
        training_epochs=epochs_count)

    ppl = BasePipeline([pipeline_item])
    ppl.run(None, params_dict={"supported_data_types": [DataType.Train],
                               "data_type": DataType.Train})
