from arenets.attention.architectures.self_p_zhou import self_attention_by_peng_zhou
from arenets.context.architectures.att_self_rcnn import AttentionSelfRCNN


class AttentionSelfPZhouRCNN(AttentionSelfRCNN):

    def get_attention_alphas(self, rnn_outputs):
        return self_attention_by_peng_zhou(rnn_outputs)
