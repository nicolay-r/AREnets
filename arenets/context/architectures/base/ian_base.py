import tensorflow as tf

from arenets.attention.helpers import embedding
from arenets.context.architectures.base.fc_single import FullyConnectedLayer
from arenets.arekit.common.data_type import DataType
from arenets.tf_helpers import sequence
from arenets.context.configurations.base.ian_base import StatesAggregationModes, IANBaseConfig
from arenets.sample import InputSample
from arenets.tf_helpers import filtering
from arenets.tf_helpers.sequence import get_cell


class IANBase(FullyConnectedLayer):
    """
    Title: Interactive Attention Networks for Aspect-Level Sentiment Classification
    Paper: https://arxiv.org/pdf/1709.00893.pdf
    Author: Peiqin Lin
    Code: https://github.com/lpq29743/IAN/blob/master/model.py
    """
    ASPECT_W = 'W_a'
    CONTEXT_W = 'W_c'

    ASPECT_B = 'B_a'
    CONTEXT_B = 'B_c'

    def __init__(self):
        super(IANBase, self).__init__()

        # Hidden states
        self.__w_a = None
        self.__w_c = None
        self.__w_l = None
        self.__b_a = None
        self.__b_c = None
        self.__b_l = None

        # Input dependent parameters
        self.__aspect_att = None
        self.__context_att = None

        self.__dropout_rnn_keep_prob = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return self.Config.HiddenSize * 2

    def get_aspect_input(self):
        raise NotImplementedError()

    # endregion

    # region public 'set' methods

    def set_input_rnn_keep_prob(self, value):
        self.__dropout_rnn_keep_prob = value

    # endregion

    # region public 'init' methods

    def init_input(self):
        super(IANBase, self).init_input()
        self.__dropout_rnn_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32,
                                                                name='ctx_' + "dropout_rnn_keep_prob")

    def init_body_dependent_hidden_states(self):
        assert(isinstance(self.Config, IANBaseConfig))

        self.__w_a = tf.compat.v1.get_variable(
            name=self.ASPECT_W,
            shape=[self.Config.HiddenSize, self.Config.HiddenSize],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

        self.__w_c = tf.compat.v1.get_variable(
            name=self.CONTEXT_W,
            shape=[self.Config.HiddenSize, self.Config.HiddenSize],
            initializer=self.Config.WeightInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

        self.__b_a = tf.compat.v1.get_variable(
            name=self.ASPECT_B,
            shape=[self.Config.MaxAspectLength, 1],
            initializer=self.Config.BiasInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

        self.__b_c = tf.compat.v1.get_variable(
            name=self.CONTEXT_B,
            shape=[self.Config.MaxContextLength, 1],
            initializer=self.Config.BiasInitializer,
            regularizer=self.Config.LayerRegularizer,
            trainable=True)

    def init_embedded_input(self):
         context_embedded = super(IANBase, self).init_embedded_input()
         aspect_embedded = self.__compose_all_parameters()
         return [context_embedded, aspect_embedded]

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(embedded_terms, list))

        context_embedded, aspects_embedded = embedded_terms

        with tf.name_scope('dynamic_rnn'):

            # Prepare cells
            aspect_cell = get_cell(hidden_size=self.Config.HiddenSize,
                                   cell_type=self.Config.CellType,
                                   dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            context_cell = get_cell(hidden_size=self.Config.HiddenSize,
                                    cell_type=self.Config.CellType,
                                    dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            # Calculate input lengths
            aspect_lens = sequence.calculate_sequence_length(
                sequence=self.get_aspect_input(),
                is_neg_placeholder=InputSample.FRAMES_PAD_VALUE < 0)

            aspect_lens_casted = tf.cast(x=tf.maximum(aspect_lens, 1), dtype=tf.int32)

            context_lens = sequence.calculate_sequence_length(
                sequence=self.get_input_parameter(InputSample.I_X_INDS))

            context_lens_casted = tf.cast(x=tf.maximum(context_lens, 1), dtype=tf.int32)

            # Receive aspect output
            aspect_outputs, _ = sequence.rnn(cell=aspect_cell,
                                             inputs=aspects_embedded,
                                             sequence_length=aspect_lens_casted,
                                             dtype=tf.float32,
                                             scope='aspect_outputs')
            aspect_avg = self.__aggreagate(self.Config,
                                           outputs=aspect_outputs,
                                           length=aspect_lens_casted)

            # Receive context output
            context_outputs, _ = sequence.rnn(cell=context_cell,
                                              inputs=context_embedded,
                                              sequence_length=context_lens_casted,
                                              dtype=tf.float32,
                                              scope='context_outputs')
            context_avg = self.__aggreagate(self.Config,
                                            outputs=context_outputs,
                                            length=context_lens_casted)

            # Attention for aspects
            self.__aspect_att = tf.nn.softmax(
                tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.__w_a,
                                     tf.expand_dims(context_avg, -1)) + self.__b_a),
                axis=1)
            aspect_rep = tf.reduce_sum(self.__aspect_att * aspect_outputs, axis=1)

            # Attention for context
            self.__context_att = tf.nn.softmax(
                tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', context_outputs, self.__w_c,
                                     tf.expand_dims(aspect_avg, -1)) + self.__b_c),
                axis=1)
            context_rep = tf.reduce_sum(self.__context_att * context_outputs, axis=1)

            return tf.concat([context_rep, aspect_rep], 1)

    # endregion

    # region public 'iter' methods

    def iter_hidden_parameters(self):
        for name, value in super(IANBase, self).iter_hidden_parameters():
            yield name, value

        if self.__w_a is not None:
            yield self.ASPECT_W, self.__w_a

        if self.__w_c is not None:
            yield self.CONTEXT_W, self.__w_c

        if self.__b_a is not None:
            yield self.ASPECT_B, self.__b_a

        if self.__b_c is not None:
            yield self.CONTEXT_B, self.__b_c

    def iter_input_dependent_hidden_parameters(self):
        for key, value in super(IANBase, self).iter_input_dependent_hidden_parameters():
            yield key, value

        yield 'aspect_att', self.__aspect_att
        yield 'context_att', self.__context_att
        yield 'aspects', self.get_aspect_input()

    # endregion

    # region public 'create' methods

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IANBase, self).create_feed_dict(input=input, data_type=data_type)
        feed_dict[self.__dropout_rnn_keep_prob] = self.Config.DropoutRNNKeepProb if data_type == DataType.Train else 1.0
        return feed_dict

    # endregion

    # region private methods

    @staticmethod
    def __aggreagate(config, outputs, length):
        assert(isinstance(config, IANBaseConfig))
        if config.StatesAggregationMode == StatesAggregationModes.AVERAGE:
            return tf.reduce_mean(outputs, 1)
        if config.StatesAggregationMode == StatesAggregationModes.LAST_IN_SEQUENCE:
            return sequence.select_last_relevant_in_sequence(outputs, length)
        else:
            raise Exception('"{}" type does not supported'.format(config.StatesAggregationMode))

    def __compose_all_parameters(self):
        e_term_type_indices = tf.constant(value=1.0,
                                          shape=[self.Config.BatchSize,
                                                 self.Config.MaxAspectLength,
                                                 self.TermTypeEmbeddingSize])

        aspect_input = tf.concat(values=self.__get_embedded_parameters() + [e_term_type_indices],
                                 axis=-1)

        aspect_embedded = self.process_embedded_data(
            embedded=aspect_input,
            dropout_keep_prob=self.EmbeddingDropoutKeepProb)

        aspect_embedded = tf.reshape(aspect_embedded, [self.Config.BagsPerMinibatch,
                                                       self.Config.MaxAspectLength,
                                                       self.TermEmbeddingSize])

        return aspect_embedded

    def __get_embedded_parameters(self):
        embedded_params = []
        for e, v in embedding.get_ev(self):
            param_inds = filtering.filter_batch_elements(elements=v,
                                                         inds=self.get_aspect_input(),
                                                         handler=filtering.select_entity_related_elements)
            emb_param = tf.nn.embedding_lookup(e, param_inds)
            embedded_params.append(emb_param)

        return embedded_params

    # endregion
