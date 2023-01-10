from arenets.attention.architectures.self_z_yang import self_attention_by_z_yang
from arenets.context.architectures.att_self_cnn import SelfAttentionCNN


class AttentionSelfZYangCNN(SelfAttentionCNN):

    def get_attention_alphas(self, input_data):
        return self_attention_by_z_yang(inputs=input_data, attention_size=self.Config.AttentionSize)
