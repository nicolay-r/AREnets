from arenets.attention.architectures.self_p_zhou import self_attention_by_peng_zhou
from arenets.context.architectures.att_self_cnn import SelfAttentionCNN


class AttentionSelfPZhouCNN(SelfAttentionCNN):

    def get_attention_alphas(self, input_data):
        return self_attention_by_peng_zhou(input_data)
