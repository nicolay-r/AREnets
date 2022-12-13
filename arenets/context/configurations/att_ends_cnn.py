from arenets.attention.architectures.mlp import MLPAttention
from arenets.attention.configurations.mlp import MLPAttentionConfig
from arenets.context.configurations.base.att_cnn_base import AttentionCNNBaseConfig


class AttentionEndsCNNConfig(AttentionCNNBaseConfig):

    def __init__(self):
        super(AttentionEndsCNNConfig, self).__init__()
        self.__attention = None
        self.__attention_config = MLPAttentionConfig()

    # region properties

    @property
    def AttentionModel(self):
        return self.__attention

    # endregion

    # region public methods

    def get_attention_parameters(self):
        return self.__attention_config.get_parameters()

    def reinit_config_dependent_parameters(self):
        super(AttentionEndsCNNConfig, self).reinit_config_dependent_parameters()

        self.__attention = MLPAttention(
            cfg=self.__attention_config,
            batch_size=self.BatchSize,
            terms_per_context=self.TermsPerContext)

    # endregion
