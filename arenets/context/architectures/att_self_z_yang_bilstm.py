from arenets.attention.architectures.self_z_yang import self_attention_by_z_yang, \
    calculate_sequential_attentive_weights_by_z_yang
from arenets.context.architectures.base.att_self_bilstm_base import AttentionSelfBiLSTMBase
from arenets.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig


class AttentionSelfZYangBiLSTM(AttentionSelfBiLSTMBase):

    def get_attention_output_with_alphas(self, rnn_outputs):
        assert(isinstance(self.Config, AttentionSelfZYangBiLSTMConfig))
        alphas = self_attention_by_z_yang(inputs=rnn_outputs, attention_size=self.Config.AttentionSize)
        attentive_output = calculate_sequential_attentive_weights_by_z_yang(inputs=rnn_outputs, alphas=alphas)
        return attentive_output, alphas
