import tensorflow as tf
from arenets.attention import common
from arenets.context.architectures.cnn import VanillaCNN


class SelfAttentionCNN(VanillaCNN):
    """ Self-attention application, based on the
        convolved input information by a set of filters
    """

    def __init__(self):
        super(SelfAttentionCNN, self).__init__()
        self.__att_alphas = None

    def get_attention_alphas(self, input_data):
        raise NotImplementedError()

    # region public methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(SelfAttentionCNN, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_alphas

    # endregion

    def convolved_transformation_optional(self, value):
        self.__att_alphas = self.get_attention_alphas(value)
        return value * tf.expand_dims(self.__att_alphas, -1)
