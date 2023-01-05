import tensorflow as tf

from arenets.arekit.common.data_type import DataType

from arenets.context.configurations.base.base import DefaultNetworkConfig
from arenets.core.feeding.batch.base import MiniBatch
from arenets.core.nn import NeuralNetwork
from arenets.tf_helpers.initialization import init_weighted_cost, init_accuracy
from arenets.sample import InputSample


class SingleInstanceNeuralNetwork(NeuralNetwork):

    def __init__(self):
        self.__cfg = None

        self.__labels = None
        self.__scaled_logits = None

        self.__term_emb = None
        self.__dist_emb = None
        self.__pos_emb = None
        self.__sent_emb = None

        self.__cost = None
        self.__accuracy = None

        self.__input = {}

        self.__y = None
        self.__dropout_keep_prob = None
        self.__embedding_dropout_keep_prob = None

    # region property

    @property
    def Config(self):
        return self.__cfg

    @property
    def Labels(self):
        return self.__labels

    @property
    def ScaledLogits(self):
        return self.__scaled_logits

    @property
    def Accuracy(self):
        return self.__accuracy

    @property
    def Cost(self):
        return self.__cost

    @property
    def TermEmbeddingSize(self):
        size = self.__cfg.TermEmbeddingShape[1] + \
               4 * self.__cfg.DistanceEmbeddingSize + \
               self.__cfg.SentimentEmbeddingSize + \
               self.TermTypeEmbeddingSize

        if self.__cfg.UsePOSEmbedding:
            size += self.__cfg.PosEmbeddingSize

        return size

    @property
    def TermTypeEmbeddingSize(self):
        return 1

    @property
    def EmbeddingDropoutKeepProb(self):
        return self.__embedding_dropout_keep_prob

    @property
    def DropoutKeepProb(self):
        return self.__dropout_keep_prob

    @property
    def ContextEmbeddingSize(self):
        raise NotImplementedError()

    @property
    def TermEmbedding(self):
        return self.__term_emb

    @property
    def DistanceEmbedding(self):
        return self.__dist_emb

    @property
    def POSEmbedding(self):
        return self.__pos_emb

    @property
    def SentimentEmbedding(self):
        return self.__sent_emb

    # endregion

    def has_input_parameter(self, param):
        return param in self.__input

    def get_input_parameter(self, param):
        return self.__input[param]

    def get_input_labels(self):
        return self.__y

    def set_input_parameter(self, param, value):
        self.__input[param] = value

    def set_input_dropout_keep_prob(self, value):
        self.__dropout_keep_prob = value

    def set_input_embedding_dropout_keep_prob(self, value):
        self.__embedding_dropout_keep_prob = value

    def set_input_rnn_keep_prob(self, value):
        """
        Specific dropout only recurrent neural core
        """
        pass

    def update_network_specific_parameters(self):
        pass

    # region body

    def compile_hidden_states_only(self, config):
        """
        Utilized in multiinstance models.
        """
        self.__cfg = config
        self.__init_embedding_hidden_states()
        self.init_body_dependent_hidden_states()

    def compile(self, config, reset_graph, graph_seed=None, eager=False):
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(reset_graph, bool))
        assert(isinstance(graph_seed, int) or graph_seed is None)

        self.__cfg = config

        if eager is False:
            tf.compat.v1.disable_eager_execution()
        else:
            tf.compat.v1.enable_eager_execution()

        if reset_graph:
            tf.compat.v1.reset_default_graph()

        if graph_seed is not None:
            tf.compat.v1.set_random_seed(graph_seed)

        self.init_input()
        self.__init_embedding_hidden_states()
        self.init_body_dependent_hidden_states()
        self.init_logits_hidden_states()

        embedded_terms = self.init_embedded_input()
        context_embedding = self.init_context_embedding(embedded_terms)

        logits_unscaled, logits_unscaled_dropped = self.init_logits_unscaled(context_embedding)

        # Get output for each sample
        output = tf.nn.softmax(logits_unscaled)
        scaled_logits = self.__to_mean_of_bag(output)
        mean_output = tf.argmax(scaled_logits, axis=1)

        # Create labeling only for whole bags
        self.__labels = tf.cast(mean_output, tf.int32)
        self.__scaled_logits = scaled_logits

        self.__cost = self.init_cost(logits_unscaled_dropped)

        self.__accuracy = self.init_accuracy()

    # endregion

    # region init

    def init_body_dependent_hidden_states(self):
        """
        States that assumes to be utilized in model body.
        """
        raise NotImplementedError()

    def init_logits_hidden_states(self):
        """
        States that assumes to be in final fully connection layer
        """
        raise NotImplementedError()

    def init_context_embedding(self, embedded_terms):
        """
        Important: output considered as vector, i.e. flattened embedding presentation.
        embedded_terms: Tensor
            tensor shape of ()
        """
        raise NotImplementedError()

    def init_logits_unscaled(self, context_embedding):
        raise NotImplementedError()

    def init_embedded_input(self):
        return self.process_embedded_data(
            embedded=self.__init_embedded_terms(),
            dropout_keep_prob=self.__embedding_dropout_keep_prob)

    def init_input(self):
        """
        Input placeholders
        """
        prefix = 'ctx_'

        self.__input[InputSample.I_X_INDS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_X_INDS)

        self.__input[InputSample.I_SUBJ_DISTS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_SUBJ_DISTS)

        self.__input[InputSample.I_SYN_SUBJ_INDS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.SynonymsPerContext],
            name=prefix + InputSample.I_SYN_SUBJ_INDS)

        self.__input[InputSample.I_NEAREST_SUBJ_DISTS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_NEAREST_SUBJ_DISTS)

        self.__input[InputSample.I_OBJ_DISTS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_OBJ_DISTS)

        self.__input[InputSample.I_SYN_OBJ_INDS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.SynonymsPerContext],
            name=prefix + InputSample.I_SYN_OBJ_INDS)

        self.__input[InputSample.I_NEAREST_OBJ_DISTS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_NEAREST_OBJ_DISTS)

        self.__input[InputSample.I_TERM_TYPE] = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_TERM_TYPE)

        self.__input[InputSample.I_POS_INDS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_POS_INDS)

        self.__input[InputSample.I_SUBJ_IND] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize],
            name=prefix + InputSample.I_SUBJ_IND)

        self.__input[InputSample.I_OBJ_IND] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize],
            name=prefix + InputSample.I_OBJ_IND)

        self.__input[InputSample.I_FRAME_CONNOTATIONS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_FRAME_CONNOTATIONS)

        self.__input[InputSample.I_FRAME_INDS] = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.FramesPerContext],
            name=prefix + InputSample.I_FRAME_INDS)

        self.__y = tf.compat.v1.placeholder(dtype=tf.int32,
                                            shape=[self.__cfg.BagsPerMinibatch],
                                            name=prefix + MiniBatch.I_LABELS)

        self.__dropout_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32,
                                                            name="ctx_dropout_keep_prob")

        self.__embedding_dropout_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32,
                                                                      name="cxt_emb_dropout_keep_prob")

    def init_cost(self, logits_unscaled_dropped):
        with tf.name_scope("cost"):
            cost = init_weighted_cost(
                logits_unscaled_dropout=self.__to_mean_of_bag(logits_unscaled_dropped),
                true_labels=self.__y,
                config=self.Config)
        return cost

    def init_accuracy(self):
        with tf.name_scope("accuracy"):
            accuracy = init_accuracy(labels=self.Labels, true_labels=self.__y)
        return accuracy

    # endregion

    def create_feed_dict(self, input, data_type):
        assert(isinstance(input, dict))

        feed_dict = {}
        for param in InputSample.iter_parameters():
            if param not in self.__input:
                continue
            feed_dict[self.__input[param]] = input[param]

        feed_dict[self.__y] = input[MiniBatch.I_LABELS]
        feed_dict[self.__dropout_keep_prob] = self.__cfg.DropoutKeepProb if data_type == DataType.Train else 1.0
        feed_dict[self.__embedding_dropout_keep_prob] = self.__cfg.EmbeddingDropoutKeepProb if data_type == DataType.Train else 1.0

        return feed_dict

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(SingleInstanceNeuralNetwork, self).iter_input_dependent_hidden_parameters():
            yield name, value

        # TODO. This should be a part of the sample.
        yield 'x', self.__input[InputSample.I_X_INDS]
        yield 'obj_ind', self.__input[InputSample.I_OBJ_IND]
        yield 'subj_ind', self.__input[InputSample.I_SUBJ_IND]
        yield 'frame_inds', self.__input[InputSample.I_FRAME_INDS]
        yield 'frame_connotation_inds', self.__input[InputSample.I_FRAME_CONNOTATIONS]

        # Provide base input paramaters.
        yield 'y_labels', self.Labels
        yield 'y_scaled_logits', self.ScaledLogits
        yield 'y_etalon_labels', self.__y

    # region static methods

    @staticmethod
    def process_embedded_data(embedded, dropout_keep_prob):
        return tf.nn.dropout(embedded, keep_prob=dropout_keep_prob)

    # endregion

    # region private methods

    def __to_mean_of_bag(self, logits):
        loss = tf.reshape(logits, [self.__cfg.BagsPerMinibatch, self.__cfg.BagSize, self.__cfg.ClassesCount])
        return tf.reduce_mean(loss, axis=1)

    def __init_embedded_terms(self):

        term_types = tf.reshape(
            tensor=self.__input[InputSample.I_TERM_TYPE],
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext, self.TermTypeEmbeddingSize])

        embedded_terms = tf.concat(
            [tf.nn.embedding_lookup(self.__term_emb, self.__input[InputSample.I_X_INDS]),
             tf.nn.embedding_lookup(self.__dist_emb, self.__input[InputSample.I_SUBJ_DISTS]),
             tf.nn.embedding_lookup(self.__dist_emb, self.__input[InputSample.I_OBJ_DISTS]),
             tf.nn.embedding_lookup(self.__dist_emb, self.__input[InputSample.I_NEAREST_SUBJ_DISTS]),
             tf.nn.embedding_lookup(self.__dist_emb, self.__input[InputSample.I_NEAREST_OBJ_DISTS]),
             tf.nn.embedding_lookup(self.__sent_emb, self.__input[InputSample.I_FRAME_CONNOTATIONS]),
             term_types],
            axis=-1)

        if self.__cfg.UsePOSEmbedding:
            embedded_terms = tf.concat([embedded_terms,
                                        tf.nn.embedding_lookup(self.__pos_emb, self.__input[InputSample.I_POS_INDS])],
                                       axis=-1)

        return embedded_terms

    def __init_embedding_hidden_states(self):
        assert(self.__cfg.TermsPerContext is not None)
        assert(self.__cfg.DistanceEmbeddingSize is not None)
        assert(self.__cfg.PosCount is not None)
        assert(self.__cfg.SentimentEmbeddingSize is not None)
        assert(self.__cfg.TermEmbeddingShape is not None)

        self.__term_emb = tf.constant(value=self.__cfg.TermEmbeddingMatrix,
                                      dtype=tf.float32,
                                      shape=self.__cfg.TermEmbeddingShape)

        self.__dist_emb = tf.compat.v1.get_variable(
            dtype=tf.float32,
            initializer=self.__cfg.EmbeddingInitializer,
            shape=[self.__cfg.TermsPerContext, self.__cfg.DistanceEmbeddingSize],
            trainable=True,
            name="dist_emb")

        self.__pos_emb = tf.compat.v1.get_variable(
            dtype=tf.float32,
            initializer=self.__cfg.EmbeddingInitializer,
            shape=[self.__cfg.PosCount, self.__cfg.PosEmbeddingSize],
            trainable=True,
            name="pos_emb")

        self.__sent_emb = tf.compat.v1.get_variable(
            dtype=tf.float32,
            initializer=self.__cfg.EmbeddingInitializer,
            shape=[3, self.__cfg.SentimentEmbeddingSize],
            trainable=True,
            name="sent_emb")

    # endregion