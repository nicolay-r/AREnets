from os.path import join

from arenets.arekit.common.data.row_ids.base import BaseIDProvider
from arenets.arekit.common.data.views.samples import LinkedSamplesStorageView
from arenets.arekit.common.data_type import DataType
from arenets.arekit.common.experiment.api.base_samples_io import BaseSamplesIO
from arenets.arekit.common.pipeline.context import PipelineContext
from arenets.arekit.common.pipeline.items.base import BasePipelineItem
from arenets.core.callback.writer import PredictResultWriterCallback
from arenets.core.ctx_inference import InferenceContext
from arenets.core.embedding_io import BaseEmbeddingIO
from arenets.core.model import BaseTensorflowModel
from arenets.core.model_ctx import TensorflowModelContext
from arenets.core.pipeline.item_fit import MinibatchFittingPipelineItem
from arenets.core.pipeline.item_keep_hidden import MinibatchHiddenFetcherPipelineItem
from arenets.core.pipeline.item_predict import EpochLabelsPredictorPipelineItem
from arenets.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arenets.core.predict.base_writer import BasePredictWriter
from arenets.factory import create_network_and_network_config_funcs
from arenets.shapes import NetworkInputShapes


class TensorflowNetworkInferencePipelineItem(BasePipelineItem):

    def __init__(self, model_name, bags_collection_type, model_input_type, predict_writer,
                 data_type, bag_size, bags_per_minibatch, nn_io, labels_count, callbacks,
                 modify_config_func=None, part_of_speech_types_count=100):
        assert(isinstance(callbacks, list))
        assert(isinstance(bag_size, int))
        assert(isinstance(predict_writer, BasePredictWriter))
        assert(isinstance(data_type, DataType))
        assert(callable(modify_config_func) or modify_config_func is None)

        # Create network an configuration.
        network_func, config_func = create_network_and_network_config_funcs(
            model_name=model_name, model_input_type=model_input_type)

        # setup network and config parameters.
        self.__network = network_func()
        self.__config = config_func()
        self.__config.modify_classes_count(labels_count)
        self.__config.modify_bag_size(bag_size)
        self.__config.modify_bags_per_minibatch(bags_per_minibatch)
        self.__config.set_class_weights([1] * labels_count)
        self.__config.set_pos_count(part_of_speech_types_count)
        self.__config.reinit_config_dependent_parameters()

        # Custom configuration modification function.
        if modify_config_func is not None:
            modify_config_func(self.__config)

        # intialize model context.
        self.__create_model_ctx = lambda inference_ctx: TensorflowModelContext(
            nn_io=nn_io,
            network=self.__network,
            config=self.__config,
            inference_ctx=inference_ctx,
            bags_collection_type=bags_collection_type)

        self.__callbacks = callbacks + [
            PredictResultWriterCallback(labels_count=labels_count, writer=predict_writer)
        ]

        self.__writer = predict_writer
        self.__bags_collection_type = bags_collection_type
        self.__data_type = data_type

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("emb_io" in input_data)
        assert("samples_io" in input_data)
        assert("predict_root" in input_data)

        emb_io = input_data["emb_io"]
        samples_io = input_data["samples_io"]
        predict_root = input_data["predict_root"]
        assert(isinstance(emb_io, BaseEmbeddingIO))
        assert(isinstance(samples_io, BaseSamplesIO))

        # Setup predicted result writer.
        full_model_name = pipeline_ctx.provide_or_none("full_model_name")
        tgt = join(predict_root, "predict-{fmn}-{dtype}.tsv.gz".format(
            fmn=full_model_name, dtype=str(self.__data_type).lower().split('.')[-1]))

        # Fetch other required in further information from input_data.
        embedding = emb_io.load_embedding()
        terms_vocab = emb_io.load_vocab()
        unknown_term_index = emb_io.UnknownTermIndex

        # Setup config parameters.
        self.__config.set_term_embedding(embedding)

        inference_ctx = InferenceContext.create_empty()
        inference_ctx.initialize(
            dtypes=[self.__data_type],
            bags_collection_type=self.__bags_collection_type,
            load_target_func=lambda data_type: samples_io.create_target(data_type=data_type),
            samples_reader=samples_io.Reader,
            samples_view=LinkedSamplesStorageView(row_ids_provider=BaseIDProvider()),
            is_external_vocab=True,
            terms_vocab=terms_vocab,
            unknown_term_index=unknown_term_index,
            labels_count=self.__config.ClassesCount,
            input_shapes=NetworkInputShapes(iter_pairs=[
                (NetworkInputShapes.FRAMES_PER_CONTEXT, self.__config.FramesPerContext),
                (NetworkInputShapes.TERMS_PER_CONTEXT, self.__config.TermsPerContext),
                (NetworkInputShapes.SYNONYMS_PER_CONTEXT, self.__config.SynonymsPerContext),
            ]),
            bag_size=self.__config.BagSize)

        # Model preparation.
        model = BaseTensorflowModel(
            context=self.__create_model_ctx(inference_ctx),
            callbacks=self.__callbacks,
            predict_pipeline=[
                EpochLabelsPredictorPipelineItem(),
                EpochLabelsCollectorPipelineItem(),
                MinibatchHiddenFetcherPipelineItem()
            ],
            fit_pipeline=[MinibatchFittingPipelineItem()])

        self.__writer.set_target(tgt)

        model.predict(data_type=self.__data_type, do_compile=True)


