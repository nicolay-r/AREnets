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
from arenets.enum_input_types import ModelInputType
from arenets.enum_name_types import ModelNames
from arenets.np_utils.writer import NpzDataWriter
from arenets.pipelines.items.infer import TensorflowNetworkInferencePipelineItem


def predict(input_data_dir, output_dir, labels_count,
            hstates_dir=None,
            modify_config_func=None,
            save_hidden_states=True,
            model_name_suffix="model",
            callbacks=None,
            bag_size=1,
            bags_per_minibatch=32,
            reader=JsonlReader(),
            model_name=ModelNames.CNN,
            data_type=DataType.Test,
            vocab_filename="vocab.txt",
            unknown_term_index=-1,
            embedding_npz_filename="term_embedding.npz"):
    """ Perform inference for dataset using a pre-trained collection
        This is a pipeline-based implementation, taken from
        the ARElight repository, see the following code for reference:
            https://github.com/nicolay-r/ARElight/blob/v0.22.0/arelight/pipelines/inference_nn.py

        modify_config_func: func of None
            allows to declare and provide your function which modifies the contents of the config.
        hstates_dir: str
            Where to keep hidden states during the model process training.
    """
    assert(isinstance(input_data_dir, str))
    assert(isinstance(output_dir, str))
    assert(isinstance(callbacks, list) or callbacks is None)
    assert(isinstance(unknown_term_index, int))

    model_io = TensorflowNeuralNetworkModelIO(
        model_name="-".join([model_name.value, model_name_suffix]),
        source_dir=output_dir)

    # Setup callbacks.
    callbacks = [] if callbacks is None else callbacks
    if save_hidden_states:
        data_writer = NpzDataWriter()
        hstates_dir = join(output_dir, "hidden") if hstates_dir is None else hstates_dir
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
            bags_collection_type=SingleBagsCollection,
            model_input_type=ModelInputType.SingleInstance,
            predict_writer=TsvPredictWriter(),
            callbacks=callbacks,
            modify_config_func=modify_config_func,
            labels_count=labels_count,
            nn_io=model_io)
    ])

    input_data = {
        "samples_io": SamplesIO(target_dir=input_data_dir, reader=reader),
        "emb_io": NpEmbeddingIO(target_dir=input_data_dir,
                                vocab_filename=vocab_filename,
                                unknown_ind=unknown_term_index,
                                embedding_npz_filename=embedding_npz_filename),
        "predict_root": output_dir
    }

    ppl.run(input_data=input_data, params_dict={"full_model_name": model_name.value})
