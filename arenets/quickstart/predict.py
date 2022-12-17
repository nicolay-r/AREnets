from arenets.arekit.common.data_type import DataType
from arenets.arekit.common.pipeline.base import BasePipeline
from arenets.arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arenets.arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arenets.arekit.contrib.utils.io_utils.samples import SamplesIO
from arenets.core.feeding.bags.collection.single import SingleBagsCollection
from arenets.core.model_io import TensorflowNeuralNetworkModelIO
from arenets.core.predict.tsv_writer import TsvPredictWriter
from arenets.enum_input_types import ModelInputType
from arenets.enum_name_types import ModelNames
from arenets.pipelines.items.infer import TensorflowNetworkInferencePipelineItem


def predict(input_data_dir, output_dir, labels_scaler,
            modify_config_func=None,
            model_name_suffix="model",
            bag_size=1,
            bags_per_minibatch=32,
            model_name=ModelNames.CNN,
            data_type=DataType.Test,
            vocab_filename="vocab.txt",
            embedding_npz_filename="term_embedding.npz"):
    """ Perform inference for dataset using a pre-trained collection
        This is a pipeline-based implementation, taken from
        the ARElight repository, see the following code for reference:
            https://github.com/nicolay-r/ARElight/blob/v0.22.0/arelight/pipelines/inference_nn.py

        modify_config_func: func of None
            allows to declare and provide your function which modifies the contents of the config.
    """
    assert(isinstance(output_dir, str))
    assert(isinstance(input_data_dir, str))

    model_io = TensorflowNeuralNetworkModelIO(
        model_name="-".join([model_name.value, model_name_suffix]),
        source_dir=output_dir)

    ppl = BasePipeline(pipeline=[
        TensorflowNetworkInferencePipelineItem(
            data_type=data_type,
            bag_size=bag_size,
            bags_per_minibatch=bags_per_minibatch,
            model_name=model_name,
            bags_collection_type=SingleBagsCollection,
            model_input_type=ModelInputType.SingleInstance,
            predict_writer=TsvPredictWriter(),
            callbacks=[],
            modify_config_func=modify_config_func,
            labels_scaler=labels_scaler,
            nn_io=model_io)
    ])

    input_data = {
        "samples_io": SamplesIO(target_dir=input_data_dir, reader=PandasCsvReader()),
        "emb_io": NpEmbeddingIO(target_dir=input_data_dir,
                                vocab_filename=vocab_filename,
                                embedding_npz_filename=embedding_npz_filename),
        "predict_root": output_dir
    }

    ppl.run(input_data=input_data, params_dict={"full_model_name": model_name.value})
