from arenets.context.configurations.cnn import CNNConfig


class AttentionSelfZYangCNNConfig(CNNConfig):

    __attention_size = 100

    def __init__(self):
        super(AttentionSelfZYangCNNConfig, self).__init__()

    @property
    def AttentionSize(self):
        return self.__attention_size

    def modify_terms_per_context(self, value):
        """ We make attention parameter dependent on the input terms count parameter.
        """
        super(AttentionSelfZYangCNNConfig, self).modify_terms_per_context(value)
        self.__attention_size = value
