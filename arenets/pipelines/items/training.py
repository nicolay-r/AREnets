import gc
import logging
import os

from arenets.arekit.common.data.row_ids.base import BaseIDProvider
from arenets.arekit.common.data.views.samples import LinkedSamplesStorageView
from arenets.arekit.common.data_type import DataType
from arenets.arekit.common.pipeline.context import PipelineContext
from arenets.arekit.common.pipeline.items.base import BasePipelineItem
from arenets.arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arenets.arekit.contrib.utils.io_utils.samples import SamplesIO
from arenets.context.configurations.base.base import DefaultNetworkConfig
from arenets.core.ctx_inference import InferenceContext
from arenets.core.feeding.bags.collection.base import BagsCollection
from arenets.core.model import BaseTensorflowModel
from arenets.core.model_ctx import TensorflowModelContext
from arenets.core.params import NeuralNetworkModelParams
from arenets.core.pipeline.item_fit import MinibatchFittingPipelineItem
from arenets.core.pipeline.item_keep_hidden import MinibatchHiddenFetcherPipelineItem
from arenets.core.pipeline.item_predict import EpochLabelsPredictorPipelineItem
from arenets.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arenets.shapes import NetworkInputShapes
from arenets.utils import rm_dir_contents


class NetworksTrainingPipelineItem(BasePipelineItem):

    def __init__(self, bags_collection_type, model_io, samples_io, emb_io,
                 config, create_network_func, training_epochs,
                 labels_count, network_callbacks, prepare_model_root=True, seed=None):
        assert(callable(create_network_func))
        assert(isinstance(samples_io, SamplesIO))
        assert(isinstance(emb_io, NpEmbeddingIO))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(seed, int) or seed is None)
        assert(isinstance(training_epochs, int))
        assert(isinstance(network_callbacks, list))
        assert(isinstance(labels_count, int))

        super(NetworksTrainingPipelineItem, self).__init__()

        self.__logger = self.__create_logger()
        self.__samples_io = samples_io
        self.__emb_io = emb_io
        self.__clear_model_root_before_experiment = prepare_model_root
        self.__config = config
        self.__create_network_func = create_network_func
        self.__bags_collection_type = bags_collection_type
        self.__network_callbacks = network_callbacks
        self.__training_epochs = training_epochs
        self.__labels_count = labels_count
        self.__model_io = model_io
        self.__seed = seed

    @staticmethod
    def __create_logger():
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        return logger

    def __prepare_model(self):
        # Clear model root before training optionally
        if self.__clear_model_root_before_experiment:
            rm_dir_contents(dir_path=self.__model_io.get_model_saving_dir(),
                            logger=self.__logger)

        # Disable tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Notify other subscribers that initialization process has been completed.
        self.__config.init_initializers()

    def __check_targets_existed(self, data_types_iter):
        """ Check that all the required resources existed.
        """

        if not self.__samples_io.check_targets_existed(data_types_iter=data_types_iter):
            return False

        if not self.__emb_io.check_targets_existed():
            return False

        return True

    def __handle_iteration(self, supported_data_types, data_type):
        assert(isinstance(supported_data_types, list))
        assert(isinstance(data_type, DataType))

        targets_existed = self.__check_targets_existed(data_types_iter=iter(supported_data_types))

        if not targets_existed:
            raise Exception("Data has not been initialized/serialized!")

        # Reading embedding.
        embedding_data = self.__emb_io.load_embedding()
        self.__config.set_term_embedding(embedding_data)
        self.__samples_io.create_target(data_type=data_type)

        # Performing samples reading process.
        inference_ctx = InferenceContext.create_empty()
        inference_ctx.initialize(
            dtypes=iter(supported_data_types),
            load_target_func=lambda dtype: self.__samples_io.create_target(data_type=dtype),
            samples_view=LinkedSamplesStorageView(row_ids_provider=BaseIDProvider()),
            samples_reader=self.__samples_io.Reader,
            is_external_vocab=not self.__model_io.IsPretrainedStateProvided,
            labels_count=self.__labels_count,
            terms_vocab=self.__emb_io.load_vocab(),
            bags_collection_type=self.__bags_collection_type,
            input_shapes=NetworkInputShapes(iter_pairs=[
                (NetworkInputShapes.FRAMES_PER_CONTEXT, self.__config.FramesPerContext),
                (NetworkInputShapes.TERMS_PER_CONTEXT, self.__config.TermsPerContext),
                (NetworkInputShapes.SYNONYMS_PER_CONTEXT, self.__config.SynonymsPerContext),
            ]),
            bag_size=self.__config.BagSize)

        if inference_ctx.HasNormalizedWeights:
            weights = inference_ctx.calc_normalized_weights(labels_count=self.__labels_count)
            self.__config.set_class_weights(weights)

        # Update parameters after iteration preparation has been completed.
        self.__config.reinit_config_dependent_parameters()

        # Initialize network and model.
        network = self.__create_network_func()
        model = BaseTensorflowModel(
            context=TensorflowModelContext(
                network=network,
                config=self.__config,
                inference_ctx=inference_ctx,
                bags_collection_type=self.__bags_collection_type,
                nn_io=self.__model_io),
            callbacks=self.__network_callbacks,
            predict_pipeline=[
                EpochLabelsPredictorPipelineItem(),
                EpochLabelsCollectorPipelineItem(),
                MinibatchHiddenFetcherPipelineItem()
            ],
            fit_pipeline=[
                MinibatchFittingPipelineItem(),
                MinibatchHiddenFetcherPipelineItem()
            ])

        # Initialize model params instance.
        model_params = NeuralNetworkModelParams(epochs_count=self.__training_epochs)

        model.fit(model_params=model_params, seed=self.__seed, data_type=data_type)

        del network
        del model

        gc.collect()

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("supported_data_types" in pipeline_ctx)

        # Prepare all the required data.
        self.__prepare_model()
        supported_data_types = pipeline_ctx.provide("supported_data_types")
        data_type = pipeline_ctx.provide_or_none("data_type")

        self.__handle_iteration(supported_data_types=supported_data_types,
                                data_type=data_type if data_type is not None else DataType.Train)
