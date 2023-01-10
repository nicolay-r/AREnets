from arenets.attention.architectures.self_p_zhou import self_attention_by_peng_zhou
from arenets.context.architectures.att_self_cnn import SelfAttentionCNN


class AttentionSelfPZhouCNN(SelfAttentionCNN):
    """
    Authors: Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, Bo Xu
    Paper: https://www.aclweb.org/anthology/P16-2034
    Code author: SeoSangwoo (c), https://github.com/SeoSangwoo
    Code: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
    """

    def get_attention_alphas(self, input_data):
        return self_attention_by_peng_zhou(input_data)
