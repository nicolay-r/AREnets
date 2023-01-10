from arenets.context.architectures.att_ends_cnn import AttentionEndsCNN
from arenets.context.architectures.att_ends_pcnn import AttentionEndsPCNN
from arenets.context.architectures.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTM
from arenets.context.architectures.att_self_p_zhou_cnn import AttentionSelfPZhouCNN
from arenets.context.architectures.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNN
from arenets.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arenets.context.architectures.att_self_z_yang_cnn import AttentionSelfZYangCNN
from arenets.context.architectures.att_self_z_yang_rcnn import AttentionSelfZYangRCNN
from arenets.context.architectures.bilstm import BiLSTM
from arenets.context.architectures.cnn import VanillaCNN
from arenets.context.architectures.ian_ends import IANEndsBased
from arenets.context.architectures.pcnn import PiecewiseCNN
from arenets.context.architectures.rcnn import RCNN
from arenets.context.architectures.rnn import RNN
from arenets.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arenets.context.configurations.att_ends_cnn import AttentionEndsCNNConfig
from arenets.context.configurations.att_ends_pcnn import AttentionEndsPCNNConfig
from arenets.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arenets.context.configurations.att_self_p_zhou_cnn import AttentionSelfPZhouCNNConfig
from arenets.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arenets.context.configurations.att_self_z_yang_cnn import AttentionSelfZYangCNNConfig
from arenets.context.configurations.bilstm import BiLSTMConfig
from arenets.context.configurations.cnn import CNNConfig
from arenets.context.configurations.ian_ends import IANEndsBasedConfig
from arenets.context.configurations.rcnn import RCNNConfig
from arenets.context.configurations.rnn import RNNConfig
from arenets.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arenets.enum_input_types import ModelInputType
from arenets.enum_name_types import ModelNames
from arenets.multi.architectures.att_self import AttSelfOverSentences
from arenets.multi.architectures.base.base import BaseMultiInstanceNeuralNetwork
from arenets.multi.architectures.max_pooling import MaxPoolingOverSentences
from arenets.multi.configurations.att_self import AttSelfOverSentencesConfig
from arenets.multi.configurations.base import BaseMultiInstanceConfig
from arenets.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig


def create_network_and_network_config_funcs(model_name, model_input_type):
    assert(isinstance(model_name, ModelNames))
    assert(isinstance(model_input_type, ModelInputType))

    ctx_network_func, ctx_config_func = __get_network_with_config_types(model_name)

    if model_input_type == ModelInputType.SingleInstance:
        # In case of a single instance model, there is no need to perform extra wrapping
        # since all the base models assumes to work with a single context (input).
        return ctx_network_func, ctx_config_func

    # Compose multi-instance neural network and related configuration
    # in a form of a wrapper over context-based neural network and configuration respectively.
    mi_network, mi_config = __get_mi_network_with_config(model_input_type)
    assert(issubclass(mi_network, BaseMultiInstanceNeuralNetwork))
    assert(issubclass(mi_config, BaseMultiInstanceConfig))
    return lambda: mi_network(context_network=ctx_network_func()), \
           lambda: mi_config(context_config=ctx_config_func())


def __get_mi_network_with_config(model_input_type):
    assert(isinstance(model_input_type, ModelInputType))
    if model_input_type == ModelInputType.MultiInstanceMaxPooling:
        return MaxPoolingOverSentences, MaxPoolingOverSentencesConfig
    if model_input_type == ModelInputType.MultiInstanceWithSelfAttention:
        return AttSelfOverSentences, AttSelfOverSentencesConfig


def __get_network_with_config_types(model_name):
    assert(isinstance(model_name, ModelNames))
    if model_name == ModelNames.SelfAttentionBiLSTM:
        return SelfAttentionBiLSTM, SelfAttentionBiLSTMConfig
    if model_name == ModelNames.AttSelfPZhouBiLSTM:
        return AttentionSelfPZhouBiLSTM, AttentionSelfPZhouBiLSTMConfig
    if model_name == ModelNames.AttSelfZYangBiLSTM:
        return AttentionSelfZYangBiLSTM, AttentionSelfZYangBiLSTMConfig
    if model_name == ModelNames.BiLSTM:
        return BiLSTM, BiLSTMConfig
    if model_name == ModelNames.CNN:
        return VanillaCNN, CNNConfig
    if model_name == ModelNames.CNNAttSelfPZhou:
        return AttentionSelfPZhouCNN, AttentionSelfPZhouCNNConfig
    if model_name == ModelNames.CNNAttSelfZYang:
        return AttentionSelfZYangCNN, AttentionSelfZYangCNNConfig
    if model_name == ModelNames.LSTM:
        return RNN, RNNConfig
    if model_name == ModelNames.PCNN:
        return PiecewiseCNN, CNNConfig
    if model_name == ModelNames.RCNN:
        return RCNN, RCNNConfig
    if model_name == ModelNames.RCNNAttZYang:
        return AttentionSelfZYangRCNN, RCNNConfig
    if model_name == ModelNames.RCNNAttPZhou:
        return AttentionSelfPZhouRCNN, RCNNConfig
    if model_name == ModelNames.IANEnds:
        return IANEndsBased, IANEndsBasedConfig
    if model_name == ModelNames.AttEndsPCNN:
        return AttentionEndsPCNN, AttentionEndsPCNNConfig
    if model_name == ModelNames.AttEndsCNN:
        return AttentionEndsCNN, AttentionEndsCNNConfig
    raise NotImplementedError(u"config was not implemented for `{}` model name".format(model_name))
