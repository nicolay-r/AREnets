import logging

from arenets.core.callback.base import NetworkCallback
from arenets.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arenets.core.predict.provider.base import BasePredictProvider
from arenets.core.utils import get_item_from_pipeline
from arenets.core.writer.base_writer import BaseIterativeWriter

logger = logging.getLogger(__name__)


class PredictResultWriterCallback(NetworkCallback):

    def __init__(self, labels_count, predict_provider, writer):
        assert(isinstance(predict_provider, BasePredictProvider))
        assert(isinstance(writer, BaseIterativeWriter))
        self.__labels_count = labels_count
        self.__predict_provider = predict_provider
        self.__writer = writer

    def on_predict_finished(self, pipeline):
        super(PredictResultWriterCallback, self).on_predict_finished(pipeline)

        item = get_item_from_pipeline(pipeline=pipeline, item_type=EpochLabelsCollectorPipelineItem)
        labeled_samples = item.LabeledSamples

        with self.__writer:
            title, contents_it = self.__predict_provider.provide(
                sample_id_with_uint_labels_iter=labeled_samples.iter_non_duplicated_labeled_sample_row_ids(),
                labels_count=self.__labels_count)

            self.__writer.write(title=title, contents_it=contents_it)
