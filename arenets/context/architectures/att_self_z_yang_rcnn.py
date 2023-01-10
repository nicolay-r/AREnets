from arenets.attention.architectures.self_z_yang import self_attention_by_z_yang
from arenets.context.architectures.att_self_rcnn import AttentionSelfRCNN


class AttentionSelfZYangRCNN(AttentionSelfRCNN):

    def get_attention_alphas(self, rnn_outputs):
        alphas = self_attention_by_z_yang(inputs=rnn_outputs, attention_size=self.Config.AttentionSize)
        return alphas
