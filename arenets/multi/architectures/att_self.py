import tensorflow as tf

from arenets.attention import common
from arenets.attention.architectures.self_p_zhou import self_attention_by_peng_zhou, \
    calculate_sequential_attentive_weights_by_peng_zhou
from arenets.multi.architectures.base.base_single_mlp import BaseMultiInstanceSingleMLP


class AttSelfOverSentences(BaseMultiInstanceSingleMLP):
    """
    Utilize sequence-based attention architectures.
    """

    def __init__(self, context_network):
        super(AttSelfOverSentences, self).__init__(context_network)
        self.__att_alphas = None

    @property
    def EmbeddingSize(self):
        return self.ContextNetwork.ContextEmbeddingSize

    def init_multiinstance_embedding(self, context_outputs):
        """ input:
                context_outputs: Tensor
                    [batches, sentences, embedding]
            output: Tensor
                tensor of shape [batches, embedding]
        """
        context_outputs = tf.reshape(context_outputs, [self.Config.BatchSize,
                                                       self.Config.BagSize,
                                                       self.ContextNetwork.ContextEmbeddingSize])

        with tf.variable_scope("mi_{}".format(common.ATTENTION_SCOPE_NAME)):
            self.__att_alphas = self_attention_by_peng_zhou(context_outputs)

            # (B, T, D) -> (B, D)
            att_output = calculate_sequential_attentive_weights_by_peng_zhou(inputs=context_outputs,
                                                                             alphas=self.__att_alphas)

        return att_output

    # region 'iter' methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttSelfOverSentences, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_alphas

    # endregion
