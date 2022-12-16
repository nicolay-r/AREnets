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


def predict_nn(output_dir, embedding_dir, samples_dir, data_folding_name="fixed", bag_size=1,
               bags_per_minibatch=32, model_name=ModelNames.CNN, data_type=DataType.Test):
    """ Perform inference for dataset using a pre-trained collection
        This is a pipeline-based implementation, taken from
        the ARElight repository, see the following code for reference:
            https://github.com/nicolay-r/ARElight/blob/v0.22.0/arelight/pipelines/inference_nn.py
    """
    assert(isinstance(output_dir, str))
    assert(isinstance(embedding_dir, str))
    assert(isinstance(samples_dir, str))

    model_io = TensorflowNeuralNetworkModelIO(
        model_name="-".join([data_folding_name, model_name.value]),
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
            labels_scaler=PosNegNeuRelationsLabelScaler(),
            nn_io=model_io)
    ])

    input_data = {
        "samples_io": SamplesIO(target_dir=samples_dir, reader=PandasCsvReader()),
        "emb_io": NpEmbeddingIO(target_dir=embedding_dir),
        "predict_root": output_dir
    }

    ppl.run(input_data=input_data, params_dict={"full_model_name": model_name.value})
