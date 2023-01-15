from os.path import join

from arenets.arekit.common.data_type import DataType
from arenets.arekit.common.pipeline.base import BasePipeline
from arenets.arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arenets.arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arenets.arekit.contrib.utils.io_utils.samples import SamplesIO
from arenets.core.callback.hidden import HiddenStatesWriterCallback
from arenets.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arenets.core.feeding.bags.collection.single import SingleBagsCollection
from arenets.core.model_io import TensorflowNeuralNetworkModelIO
from arenets.core.predict.tsv_writer import TsvPredictWriter
from arenets.core.predict.provider.id_and_binary_labels import IdAndBinaryLabelsPredictProvider
from arenets.emb_converter import convert_text_embedding_if_needed
from arenets.enum_input_types import ModelInputType
from arenets.enum_name_types import ModelNames
from arenets.np_utils.writer import NpzDataWriter
from arenets.pipelines.items.infer import TensorflowNetworkInferencePipelineItem


def predict(input_data_dir, output_dir, labels_count,
            model_input_type=ModelInputType.SingleInstance,
            hstates_dir=None,
            modify_config_func=None,
            save_hidden_states=True,
            callbacks=None,
            bag_size=1,
            bags_collection_type=SingleBagsCollection,
            bags_per_minibatch=32,
            reader=JsonlReader(),
            predict_provider=IdAndBinaryLabelsPredictProvider(),
            model_name=ModelNames.CNN,
            data_type=DataType.Test,
            word2vec_txt_model_name="model.txt",
            unknown_term_index=0):
    """ Perform inference for dataset using a pre-trained collection
        This is a pipeline-based implementation, taken from
        the ARElight repository, see the following code for reference:
            https://github.com/nicolay-r/ARElight/blob/v0.22.0/arelight/pipelines/inference_nn.py

        model_input_type: enum
            Optional wrap over context-based network core which allows to consider multiple contexts as a single one.
            default: SingleInstance
        modify_config_func: func of None
            allows to declare and provide your function which modifies the contents of the config.
        word2vec_txt_model_name: str
            this is a filename that declares word2vec model, saved as a text file, incuding vocabulary
            and vectors for every term in particular.
        hstates_dir: str
            Where to keep hidden states during the model process training.
        bags_collection_type: enum
            How data is presented; for singe instance it denotes we deal with sequence of bags, while
            for multi-instance type, every bag contains a list of bags.
        predict_provider:
            rows provider involving labels suppose to be written.
    """
    assert(isinstance(input_data_dir, str))
    assert(isinstance(output_dir, str))
    assert(isinstance(callbacks, list) or callbacks is None)
    assert(isinstance(unknown_term_index, int))

    # Declaring model io, based on Tensorflow API.
    model_io = TensorflowNeuralNetworkModelIO(model_name=model_name.value, source_dir=input_data_dir)

    # Declaring embedding input/output parameters.
    embedding_io = NpEmbeddingIO(target_dir=input_data_dir,
                                 vocab_filename="vocab.txt",
                                 unknown_ind=unknown_term_index,
                                 embedding_npz_filename="term_embedding.npz")

    if not embedding_io.check_targets_existed():
        convert_text_embedding_if_needed(txt_embedding_filepath=join(input_data_dir, word2vec_txt_model_name),
                                         embedding_io=embedding_io)

    # Setup callbacks.
    callbacks = [] if callbacks is None else callbacks
    if save_hidden_states:
        data_writer = NpzDataWriter()
        hstates_dir = join(model_io.get_model_source_path_tf_prefix(), "hidden") if hstates_dir is None else hstates_dir
        callbacks += [
            HiddenStatesWriterCallback(log_dir=hstates_dir, writer=data_writer),
            InputHiddenStatesWriterCallback(log_dir=hstates_dir, writer=data_writer)
        ]

    ppl = BasePipeline(pipeline=[
        TensorflowNetworkInferencePipelineItem(
            data_type=data_type,
            bag_size=bag_size,
            bags_per_minibatch=bags_per_minibatch,
            model_name=model_name,
            bags_collection_type=bags_collection_type,
            model_input_type=model_input_type,
            predict_writer=TsvPredictWriter(),
            callbacks=callbacks,
            modify_config_func=modify_config_func,
            labels_count=labels_count,
            predict_provider=predict_provider,
            nn_io=model_io)
    ])

    input_data = {
        "samples_io": SamplesIO(target_dir=input_data_dir, reader=reader),
        "emb_io": embedding_io,
        "predict_root": output_dir
    }

    ppl.run(input_data=input_data, params_dict={"full_model_name": model_name.value})
