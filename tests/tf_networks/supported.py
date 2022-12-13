from arenets.context.architectures.att_ef_bilstm import AttentionEndsAndFramesBiLSTM
from arenets.context.architectures.att_ef_cnn import AttentionEndsAndFramesCNN
from arenets.context.architectures.att_ef_pcnn import AttentionEndsAndFramesPCNN
from arenets.context.architectures.att_ends_cnn import AttentionEndsCNN
from arenets.context.architectures.att_frames_bilstm import AttentionFramesBiLSTM
from arenets.context.architectures.att_frames_cnn import AttentionFramesCNN
from arenets.context.architectures.att_se_bilstm import AttentionSynonymEndsBiLSTM
from arenets.context.architectures.att_se_cnn import AttentionSynonymEndsCNN
from arenets.context.architectures.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTM
from arenets.context.architectures.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNN
from arenets.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arenets.context.architectures.att_self_z_yang_rcnn import AttentionSelfZYangRCNN
from arenets.context.architectures.bilstm import BiLSTM
from arenets.context.architectures.ian_ef import IANEndsAndFrames
from arenets.context.architectures.ian_ends import IANEndsBased
from arenets.context.architectures.ian_frames import IANFrames
from arenets.context.architectures.ian_se import IANSynonymEndsBased
from arenets.context.architectures.att_ends_pcnn import AttentionEndsPCNN
from arenets.context.architectures.att_frames_pcnn import AttentionFramesPCNN
from arenets.context.architectures.att_se_pcnn import AttentionSynonymEndsPCNN
from arenets.context.architectures.pcnn import PiecewiseCNN
from arenets.context.architectures.rcnn import RCNN
from arenets.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arenets.context.configurations.att_ef_bilstm import AttentionEndsAndFramesBiLSTMConfig
from arenets.context.configurations.att_ef_cnn import AttentionEndsAndFramesCNNConfig
from arenets.context.configurations.att_ef_pcnn import AttentionEndsAndFramesPCNNConfig
from arenets.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arenets.context.configurations.att_ends_cnn import AttentionEndsCNNConfig
from arenets.context.configurations.bilstm import BiLSTMConfig
from arenets.context.configurations.att_ends_pcnn import AttentionEndsPCNNConfig
from arenets.context.configurations.att_frames_bilstm import AttentionFramesBiLSTMConfig
from arenets.context.configurations.att_frames_cnn import AttentionFramesCNNConfig
from arenets.context.configurations.att_frames_pcnn import AttentionFramesPCNNConfig
from arenets.context.configurations.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNNConfig
from arenets.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arenets.context.configurations.att_se_bilstm import AttentionSynonymEndsBiLSTMConfig
from arenets.context.configurations.att_se_cnn import AttentionSynonymEndsCNNConfig
from arenets.context.configurations.att_se_pcnn import AttentionSynonymEndsPCNNConfig
from arenets.context.configurations.att_self_z_yang_rcnn import AttentionSelfZYangRCNNConfig
from arenets.context.configurations.ian_ef import IANEndsAndFramesConfig
from arenets.context.configurations.ian_ends import IANEndsBasedConfig
from arenets.context.configurations.ian_se import IANSynonymEndsBasedConfig
from arenets.context.configurations.ian_frames import IANFramesConfig
from arenets.context.configurations.rcnn import RCNNConfig
from arenets.context.configurations.rnn import RNNConfig
from arenets.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arenets.context.configurations.cnn import CNNConfig
from arenets.context.architectures.cnn import VanillaCNN
from arenets.context.architectures.rnn import RNN


def get_supported():

    return [# Self-attention
            (SelfAttentionBiLSTMConfig(), SelfAttentionBiLSTM()),
            (AttentionSelfPZhouBiLSTMConfig(), AttentionSelfPZhouBiLSTM()),
            (AttentionSelfZYangBiLSTMConfig(), AttentionSelfZYangBiLSTM()),

            # CNN based
            (CNNConfig(), VanillaCNN()),
            (CNNConfig(), PiecewiseCNN()),

            # RNN-based
            (RNNConfig(), RNN()),
            (BiLSTMConfig(), BiLSTM()),

            # RCNN-based models (Recurrent-CNN)
            (RCNNConfig(), RCNN()),
            (AttentionSelfPZhouRCNNConfig(), AttentionSelfPZhouRCNN()),
            (AttentionSelfZYangRCNNConfig(), AttentionSelfZYangRCNN()),

            # IAN (Interactive attention networks)
            (IANFramesConfig(), IANFrames()),
            (IANEndsAndFramesConfig(), IANEndsAndFrames()),
            (IANEndsBasedConfig(), IANEndsBased()),
            (IANSynonymEndsBasedConfig(), IANSynonymEndsBased()),

            # MLP-Attention-based
            (AttentionEndsAndFramesBiLSTMConfig(), AttentionEndsAndFramesBiLSTM()),
            (AttentionFramesBiLSTMConfig(), AttentionFramesBiLSTM()),
            (AttentionSynonymEndsBiLSTMConfig(), AttentionSynonymEndsBiLSTM()),
            (AttentionEndsAndFramesPCNNConfig(), AttentionEndsAndFramesPCNN()),
            [AttentionEndsAndFramesCNNConfig(), AttentionEndsAndFramesCNN()],
            (AttentionEndsCNNConfig(), AttentionEndsCNN()),
            (AttentionEndsPCNNConfig(), AttentionEndsPCNN()),
            (AttentionSynonymEndsPCNNConfig(), AttentionSynonymEndsPCNN()),
            (AttentionSynonymEndsCNNConfig(), AttentionSynonymEndsCNN()),
            (AttentionFramesCNNConfig(), AttentionFramesCNN()),
            (AttentionFramesPCNNConfig(), AttentionFramesPCNN())]