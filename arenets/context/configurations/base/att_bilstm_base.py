from arenets.attention.architectures.mlp_interactive import InteractiveMLPAttention
from arenets.attention.configurations.mlp_interactive import InteractiveMLPAttentionConfig
from arenets.context.configurations.bilstm import BiLSTMConfig


class AttentionBiLSTMBaseConfig(BiLSTMConfig):
    """
    Based on Interactive attention model
    """

    def __init__(self, keys_count, att_support_zero_length):
        super(AttentionBiLSTMBaseConfig, self).__init__()
        assert(isinstance(att_support_zero_length, bool))
        self.__attention = None
        self.__attention_config = InteractiveMLPAttentionConfig(keys_count=keys_count)
        self.__att_support_zero_length = att_support_zero_length

    # region properties

    @property
    def AttentionModel(self):
        return self.__attention

    # endregion

    # region public methods

    def reinit_config_dependent_parameters(self):
        super(AttentionBiLSTMBaseConfig, self).reinit_config_dependent_parameters()

        self.__attention = InteractiveMLPAttention(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext,
            support_zero_length=self.__att_support_zero_length)

    def _internal_get_parameters(self):
        parameters = super(AttentionBiLSTMBaseConfig, self)._internal_get_parameters()
        parameters += self.__attention_config.get_parameters()
        return parameters

    # endregion
