from arenets.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig


class AttentionSelfZYangBiLSTMConfig(AttentionSelfPZhouBiLSTMConfig):

    __attention_size = 100

    def __init__(self):
        super(AttentionSelfZYangBiLSTMConfig, self).__init__()

    @property
    def AttentionSize(self):
        return self.__attention_size

    def modify_terms_per_context(self, value):
        """ We make attention parameter dependent on the input terms count parameter.
        """
        super(AttentionSelfZYangBiLSTMConfig, self).modify_terms_per_context(value)
        self.__attention_size = value
