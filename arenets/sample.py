import random

import numpy as np

from arenets.arekit.common.data.input.sample import InputSampleBase
from arenets.context.configurations.base.base import DefaultNetworkConfig
from arenets.features.pointers import PointersFeature
from arenets.features.sample_dist import DistanceFeatures
from arenets.features.term_connotation import FrameConnotationFeatures
from arenets.features.term_indices import IndicesFeature
from arenets.features.term_types import calculate_term_types
from arenets.features.utils import pad_right_or_crop_inplace
from arenets.shapes import NetworkInputShapes


class InputSample(InputSampleBase):
    """
    Base sample which is a part of a Bag
    It provides a to_network_input method which
    generates an input info in an appropriate way
    """

    # It is important to name with 'I_' prefix
    I_X_INDS = "x_indices"
    I_SYN_SUBJ_INDS = "syn_subj_inds"
    I_SYN_OBJ_INDS = "syn_obj_inds"
    I_SUBJ_IND = "subj_inds"
    I_OBJ_IND = "obj_inds"
    I_SUBJ_DISTS = "subj_dist"
    I_NEAREST_SUBJ_DISTS = "nearest_subj_dist"
    I_NEAREST_OBJ_DISTS = "nearest_obj_dist"
    I_OBJ_DISTS = "obj_dist"
    I_POS_INDS = "pos_inds"
    I_TERM_TYPE = "term_type"
    I_FRAME_INDS = 'frame_inds'
    I_FRAME_CONNOTATIONS = 'frame_connotations'

    # TODO: Should be -1, but now it is not supported
    FRAME_SENT_ROLES_PAD_VALUE = 0
    FRAMES_PAD_VALUE = 0
    POS_PAD_VALUE = 0
    X_PAD_VALUE = 0
    TERM_TYPE_PAD_VALUE = -1
    SYNONYMS_PAD_VALUE = 0

    def __init__(self,
                 X,
                 subj_ind,
                 obj_ind,
                 syn_subj_inds,
                 syn_obj_inds,
                 dist_from_subj,
                 dist_from_obj,
                 dist_nearest_subj,
                 dist_nearest_obj,
                 pos_indices,
                 term_type,
                 frame_indices,
                 frame_connotations,
                 input_sample_id,
                 shift_index_dbg=0):
        assert(isinstance(X, np.ndarray))
        assert(isinstance(subj_ind, int))
        assert(isinstance(obj_ind, int))
        assert(isinstance(syn_obj_inds, np.ndarray))
        assert(isinstance(syn_subj_inds, np.ndarray))
        assert(isinstance(dist_from_subj, np.ndarray))
        assert(isinstance(dist_from_obj, np.ndarray))
        assert(isinstance(dist_nearest_subj, np.ndarray))
        assert(isinstance(dist_nearest_obj, np.ndarray))
        assert(isinstance(pos_indices, np.ndarray))
        assert(isinstance(term_type, np.ndarray))
        assert(isinstance(frame_indices, np.ndarray))
        assert(isinstance(frame_connotations, np.ndarray))
        assert(isinstance(shift_index_dbg, int))

        values = [(InputSample.I_X_INDS, X),
                  (InputSample.I_SUBJ_IND, subj_ind),
                  (InputSample.I_OBJ_IND, obj_ind),
                  (InputSample.I_SYN_OBJ_INDS, syn_obj_inds),
                  (InputSample.I_SYN_SUBJ_INDS, syn_subj_inds),
                  (InputSample.I_SUBJ_DISTS, dist_from_subj),
                  (InputSample.I_OBJ_DISTS, dist_from_obj),
                  (InputSample.I_NEAREST_SUBJ_DISTS, dist_nearest_subj),
                  (InputSample.I_NEAREST_OBJ_DISTS, dist_nearest_obj),
                  (InputSample.I_POS_INDS, pos_indices),
                  (InputSample.I_FRAME_INDS, frame_indices),
                  (InputSample.I_FRAME_CONNOTATIONS, frame_connotations),
                  (InputSample.I_TERM_TYPE, term_type)]

        super(InputSample, self).__init__(shift_index_dbg=shift_index_dbg,
                                          input_sample_id=input_sample_id,
                                          values=values)

    # region class methods

    @classmethod
    def create_empty(cls, input_shapes):
        assert(isinstance(input_shapes, NetworkInputShapes))

        blank_synonyms = np.zeros(input_shapes.get_shape(input_shapes.SYNONYMS_PER_CONTEXT))
        blank_terms = np.zeros(input_shapes.get_shape(input_shapes.TERMS_PER_CONTEXT))
        blank_frames = np.full(shape=input_shapes.get_shape(input_shapes.FRAMES_PER_CONTEXT),
                               fill_value=cls.FRAMES_PAD_VALUE)
        return cls(X=blank_terms,
                   subj_ind=0,
                   obj_ind=1,
                   syn_subj_inds=blank_synonyms,
                   syn_obj_inds=blank_synonyms,
                   dist_from_subj=blank_terms,
                   dist_from_obj=blank_terms,
                   pos_indices=blank_terms,
                   term_type=blank_terms,
                   dist_nearest_subj=blank_terms,
                   dist_nearest_obj=blank_terms,
                   frame_connotations=blank_terms,
                   frame_indices=blank_frames,
                   input_sample_id="1")

    # TODO. Refactoring #199.
    @classmethod
    def _generate_test(cls, config):
        assert(isinstance(config, DefaultNetworkConfig))
        blank_synonyms = np.zeros(config.SynonymsPerContext)
        blank_terms = np.random.randint(0, 3, config.TermsPerContext)
        blank_frames = np.full(shape=config.FramesPerContext,
                               fill_value=cls.FRAMES_PAD_VALUE)
        return cls(X=blank_terms,
                   subj_ind=random.randint(0, 3),
                   obj_ind=random.randint(0, 3),
                   syn_subj_inds=blank_synonyms,
                   syn_obj_inds=blank_synonyms,
                   dist_from_subj=blank_terms,
                   dist_from_obj=blank_terms,
                   pos_indices=np.random.randint(0, 5, config.TermsPerContext),
                   term_type=np.random.randint(0, 3, config.TermsPerContext),
                   dist_nearest_subj=blank_terms,
                   dist_nearest_obj=blank_terms,
                   frame_connotations=blank_terms,
                   frame_indices=blank_frames,
                   input_sample_id="1")

    @classmethod
    def __get_index_by_term(cls, term, terms_vocab, is_external_vocab, unknown_term_index):

        if not is_external_vocab:
            # Since we consider that all the existed terms presented in vocabulary
            # we obtain the related index without any additional checks
            return terms_vocab[term]

        # In case of non-native vocabulary, we consider an additional
        # placeholed when the related term has not been found in vocabulary.
        return terms_vocab[term] if term in terms_vocab else unknown_term_index

    @staticmethod
    def calc_dist_between_text_opinion_end_indices(pos1_ind, pos2_ind):
        return abs(pos1_ind - pos2_ind)

    @classmethod
    def create_from_parameters(cls,
                               input_sample_id,  # row_id
                               terms,  # list of terms, that might be found in terms_vocab
                               entity_inds,
                               is_external_vocab,
                               subj_ind,
                               obj_ind,
                               terms_vocab,  # for indexing input (all the vocabulary, obtained from offsets.py)
                               input_shapes,
                               syn_subj_inds,
                               syn_obj_inds,
                               frame_inds,
                               pos_tags,
                               frame_connotations,
                               unknown_term_index):
        """
        Here we first need to perform indexing of terms. Therefore, mark entities, frame_variants among them.
        None parameters considered as optional.
        """
        assert(isinstance(terms, list))
        assert(isinstance(terms_vocab, dict))
        assert((isinstance(subj_ind, int) and 0 <= subj_ind < len(terms)) or subj_ind is None)
        assert((isinstance(obj_ind, int) and 0 <= obj_ind < len(terms)) or obj_ind is None)
        assert(isinstance(input_shapes, NetworkInputShapes))
        assert(isinstance(entity_inds, list) or entity_inds is None)
        assert((isinstance(syn_subj_inds, list) and len(syn_subj_inds) > 0) or syn_subj_inds is None)
        assert((isinstance(syn_obj_inds, list) and len(syn_obj_inds) > 0) or syn_obj_inds is None)
        assert(isinstance(frame_inds, list) or frame_inds is None)
        assert(isinstance(pos_tags, list) or pos_tags is None)
        assert(isinstance(frame_connotations, list) or frame_connotations is None)
        assert(isinstance(unknown_term_index, int))

        def shift_index(ind):
            return ind - get_start_offset()

        def get_start_offset():
            return x_feature.StartIndex

        def get_end_offset():
            return x_feature.EndIndex

        if len(terms) < 2:
            raise Exception("AREnets does not support the input amount of tokens less than 2 due "
                            "to assignation of the [subj] and [obj] towards the different terms"
                            "in text in case of the most part of the models")

        # Setup default parameters.
        subj_ind = 0 if subj_ind is None else subj_ind
        obj_ind = 1 if obj_ind is None else obj_ind
        entity_inds = [] if entity_inds is None else entity_inds
        frame_inds = [] if frame_inds is None else frame_inds
        frame_connotations = [] if frame_connotations is None else frame_connotations
        syn_subj_inds = [subj_ind] if syn_subj_inds is None else syn_subj_inds
        syn_obj_inds = [obj_ind] if syn_obj_inds is None else syn_obj_inds
        pos_tags = [0] * len(terms) if pos_tags is None else pos_tags

        # * Check the compatibility of the provided pos_tags with respect to the given terms.
        # * Check that we do not organize the loop relation.
        assert(len(terms) == len(pos_tags))
        assert(subj_ind != obj_ind)

        # Setup entities set.
        entities_set = set(entity_inds)

        # Composing vectors
        x_indices = np.array([cls.__get_index_by_term(term, terms_vocab, is_external_vocab, unknown_term_index)
                              for term in terms])

        terms_per_context = input_shapes.get_shape(input_shapes.TERMS_PER_CONTEXT)
        synonyms_per_context = input_shapes.get_shape(input_shapes.SYNONYMS_PER_CONTEXT)
        frames_per_context = input_shapes.get_shape(input_shapes.FRAMES_PER_CONTEXT)

        # Check an ability to create sample by analyzing required window size.
        window_size = terms_per_context
        dist_between_entities = cls.calc_dist_between_text_opinion_end_indices(pos1_ind=subj_ind, pos2_ind=obj_ind)

        if not cls._check_ends_could_be_fitted_in_window(dist_between_entities, window_size):
            # In some cases we may encounter with mismatched of tpc (terms per context parameter)
            # utilized during serialization stage, and the one utilized in training process.
            # If the windows size is lower in the latter case, we need to notify in order to prevent
            # from the infinite loop.
            raise Exception("Bounds for sample_id='{sample_id}' with "
                            "positions obj={obj_ind}, subj={subj_ind} "
                            "(diff={dist}) could not be fit in window, "
                            "size of {window}".format(sample_id=input_sample_id,
                                                      obj_ind=obj_ind,
                                                      subj_ind=subj_ind,
                                                      dist=dist_between_entities,
                                                      window=window_size))

        x_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=x_indices,
            e1_ind=subj_ind,
            e2_ind=obj_ind,
            expected_size=window_size,
            filler=cls.X_PAD_VALUE)

        pos_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=np.array(pos_tags),
            e1_ind=subj_ind,
            e2_ind=obj_ind,
            expected_size=window_size,
            filler=cls.POS_PAD_VALUE)

        term_type_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=calculate_term_types(terms=terms,
                                              entity_inds_set=entities_set),
            e1_ind=subj_ind,
            e2_ind=obj_ind,
            expected_size=window_size,
            filler=cls.TERM_TYPE_PAD_VALUE)

        frame_connotations_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=FrameConnotationFeatures.to_input(frame_inds=frame_inds,
                                                           frame_connotations=frame_connotations,
                                                           size=len(terms),
                                                           filler=cls.FRAME_SENT_ROLES_PAD_VALUE),
            e1_ind=subj_ind,
            e2_ind=obj_ind,
            expected_size=window_size,
            filler=cls.FRAME_SENT_ROLES_PAD_VALUE)

        frames_feature = PointersFeature.create_shifted_and_fit(
            original_value=frame_inds,
            start_offset=get_start_offset(),
            end_offset=get_end_offset(),
            expected_size=frames_per_context,
            filler=cls.FRAMES_PAD_VALUE)

        syn_subj_inds_feature = PointersFeature.create_shifted_and_fit(
            original_value=syn_subj_inds,
            start_offset=get_start_offset(),
            end_offset=get_end_offset(),
            filler=cls.SYNONYMS_PAD_VALUE)

        syn_obj_inds_feature = PointersFeature.create_shifted_and_fit(
            original_value=syn_obj_inds,
            start_offset=get_start_offset(),
            end_offset=get_end_offset(),
            filler=cls.SYNONYMS_PAD_VALUE)

        shifted_subj_ind = shift_index(subj_ind)
        shifted_obj_ind = shift_index(obj_ind)

        dist_from_subj = DistanceFeatures.distance_feature(position=shifted_subj_ind, size=terms_per_context)
        dist_from_obj = DistanceFeatures.distance_feature(position=shifted_obj_ind, size=terms_per_context)

        dist_nearest_subj = DistanceFeatures.distance_abs_nearest_feature(
            positions=syn_subj_inds_feature.ValueVector,
            size=terms_per_context)

        dist_nearest_obj = DistanceFeatures.distance_abs_nearest_feature(
            positions=syn_obj_inds_feature.ValueVector,
            size=terms_per_context)

        pad_right_or_crop_inplace(lst=syn_subj_inds_feature.ValueVector,
                                  pad_size=synonyms_per_context,
                                  filler=cls.SYNONYMS_PAD_VALUE)

        pad_right_or_crop_inplace(lst=syn_obj_inds_feature.ValueVector,
                                  pad_size=synonyms_per_context,
                                  filler=cls.SYNONYMS_PAD_VALUE)

        return cls(X=np.array(x_feature.ValueVector),
                   subj_ind=shifted_subj_ind,
                   obj_ind=shifted_obj_ind,
                   syn_subj_inds=np.array(syn_subj_inds_feature.ValueVector),
                   syn_obj_inds=np.array(syn_obj_inds_feature.ValueVector),
                   dist_from_subj=dist_from_subj,
                   dist_from_obj=dist_from_obj,
                   dist_nearest_subj=dist_nearest_subj,
                   dist_nearest_obj=dist_nearest_obj,
                   pos_indices=np.array(pos_feature.ValueVector),
                   term_type=np.array(term_type_feature.ValueVector),
                   frame_indices=np.array(frames_feature.ValueVector),
                   frame_connotations=np.array(frame_connotations_feature.ValueVector),
                   input_sample_id=input_sample_id,
                   shift_index_dbg=get_start_offset())

    # endregion

    @staticmethod
    def iter_parameters():
        for var_name in dir(InputSample):
            if not var_name.startswith('I_'):
                continue
            yield getattr(InputSample, var_name)
