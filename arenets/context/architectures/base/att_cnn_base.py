import tensorflow as tf

from arenets.attention import common
from arenets.attention.helpers import embedding
from arenets.context.architectures.cnn import VanillaCNN


class AttentionCNNBase(VanillaCNN):
    """
    Author: Yatian Shen, Xuanjing Huang
    Paper: https://www.aclweb.org/anthology/C16-1238

    Represents a base (abstract) class with attention scope.
    Usage:
        implement `get_att_input` method in nested class.
        configuration should include AttentionModel.
    """

    def __init__(self):
        super(AttentionCNNBase, self).__init__()
        self.__att_weights = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionCNNBase, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    # endregion

    def set_att_weights(self, weights):
        self.__att_weights = weights

    def get_att_input(self):
        """
        This is an abstract method which is considered to be implemented in nested class.
        """
        raise NotImplementedError()

    # region public 'init' methods

    def init_body_dependent_hidden_states(self):
        super(AttentionCNNBase, self).init_body_dependent_hidden_states()
        with tf.variable_scope(common.ATTENTION_SCOPE_NAME):
            self.Config.AttentionModel.init_term_embedding_size(p_names_with_sizes=embedding.get_ns(self))
            self.Config.AttentionModel.init_hidden()

    def init_context_embedding_core(self, embedded_terms):
        g = super(AttentionCNNBase, self).init_context_embedding_core(embedded_terms)

        att_e, att_weights = embedding.init_mlp_attention_embedding(
            ctx_network=self,
            mlp_att=self.Config.AttentionModel,
            keys=self.get_att_input())

        self.set_att_weights(att_weights)

        return tf.concat([g, att_e], axis=-1)

    # endregion

    # region public 'iter' methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionCNNBase, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_weights

    # endregion
