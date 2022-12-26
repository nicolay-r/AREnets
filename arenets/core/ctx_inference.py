import collections
import logging

from arenets.arekit.common.data.input.reader import BaseReader
from arenets.arekit.common.data.storages.base import BaseRowsStorage
from arenets.arekit.common.data.views.samples import LinkedSamplesStorageView
from arenets.arekit.common.data_type import DataType
from arenets.arekit.model.labeling.stat import calculate_labels_distribution_stat
from arenets.core.feeding.bags.collection.base import BagsCollection
from arenets.core.input.rows_parser import ParsedSampleRow
from arenets.sample import InputSample

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InferenceContext(object):

    def __init__(self, sample_label_pairs_dict, bags_collections_dict):
        assert(isinstance(sample_label_pairs_dict, dict))
        assert(isinstance(bags_collections_dict, dict))
        self.__sample_label_pairs_dict = sample_label_pairs_dict
        self.__bags_collections_dict = bags_collections_dict
        self.__train_stat_uint_labeled_sample_row_ids = None

    # region Properties

    @property
    def BagsCollections(self):
        return self.__bags_collections_dict

    @property
    def SampleIdAndLabelPairs(self):
        return self.__sample_label_pairs_dict

    @property
    def HasNormalizedWeights(self):
        return self.__train_stat_uint_labeled_sample_row_ids is not None

    # endregion

    @classmethod
    def create_empty(cls):
        return cls(sample_label_pairs_dict={}, bags_collections_dict={})

    def initialize(self, dtypes, load_target_func, samples_view, samples_reader, is_external_vocab,
                   terms_vocab, unknown_term_index, labels_count, bags_collection_type, bag_size, input_shapes):
        """
        Perform reading information from the serialized experiment inputs.
        Initializing core configuration.
        """
        assert(isinstance(dtypes, collections.Iterable))
        assert(isinstance(is_external_vocab, bool))
        assert(isinstance(labels_count, int) and labels_count > 0)
        assert(isinstance(samples_view, LinkedSamplesStorageView))
        assert(isinstance(samples_reader, BaseReader))
        assert(callable(load_target_func))

        # Reading from serialized information
        for data_type in dtypes:

            # Load Samples Storage.
            storage = samples_reader.read(load_target_func(data_type))

            # Extracting such information from serialized files.
            bags_collection = self.__read_for_data_type(
                linked_samples_iter=samples_view.iter_from_storage(storage),
                is_external_vocab=is_external_vocab,
                bags_collection_type=bags_collection_type,
                terms_vocab=terms_vocab,
                bag_size=bag_size,
                input_shapes=input_shapes,
                desc="Filling bags collection [{}]".format(data_type),
                unknown_term_index=unknown_term_index)

            uint_labeled_sample_row_ids = self.__get_labeled_sample_row_ids(storage)

            # Saving into dictionaries.
            self.__bags_collections_dict[data_type] = bags_collection
            self.__sample_label_pairs_dict[data_type] = uint_labeled_sample_row_ids

            if data_type == DataType.Train:
                self.__train_stat_uint_labeled_sample_row_ids = uint_labeled_sample_row_ids

    def calc_normalized_weights(self, labels_count):
        assert(isinstance(labels_count, int) and labels_count > 0)

        if self.__train_stat_uint_labeled_sample_row_ids is None:
            return

        normalized_label_stat, _ = calculate_labels_distribution_stat(
            uint_labeled_sample_row_ids=self.__train_stat_uint_labeled_sample_row_ids,
            classes_count=labels_count)

        return normalized_label_stat

    # region private methods

    @staticmethod
    def __read_for_data_type(linked_samples_iter, is_external_vocab, bags_collection_type,
                             terms_vocab, unknown_term_index, bag_size, input_shapes, desc=""):
        assert(issubclass(bags_collection_type, BagsCollection))

        return bags_collection_type.from_formatted_samples(
            linked_samples_iter=linked_samples_iter,
            desc=desc,
            bag_size=bag_size,
            shuffle=True,
            create_empty_sample_func=lambda: InputSample.create_empty(input_shapes),
            create_sample_func=lambda row: InputSample.create_from_parameters(
                input_sample_id=row.SampleID,
                terms=row.Terms,
                entity_inds=row.EntityInds,
                is_external_vocab=is_external_vocab,
                subj_ind=row.SubjectIndex,
                obj_ind=row.ObjectIndex,
                terms_vocab=terms_vocab,
                frame_inds=row.TextFrameVariantIndices,
                frame_connotations=row.TextFrameConnotations,
                syn_obj_inds=row.SynonymObjectInds,
                syn_subj_inds=row.SynonymSubjectInds,
                input_shapes=input_shapes,
                pos_tags=row.PartOfSpeechTags,
                unknown_term_index=unknown_term_index))

    @staticmethod
    def __get_labeled_sample_row_ids(storage):
        assert(isinstance(storage, BaseRowsStorage))
        rows_list = []
        for _, row in storage:
            labeled_row = InferenceContext.__extract_labeled_rows(row)
            rows_list.append(labeled_row)
        return rows_list

    @staticmethod
    def __extract_labeled_rows(row):
        parsed_row = ParsedSampleRow(row)
        return parsed_row.SampleID, parsed_row.UintLabel

    # endregion
