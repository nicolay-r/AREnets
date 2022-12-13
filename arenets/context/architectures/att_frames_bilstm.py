from arenets.context.architectures.base.att_bilstm_base import AttentionBiLSTMBase
from arenets.sample import InputSample


class AttentionFramesBiLSTM(AttentionBiLSTMBase):
    """
    Based on a frames in context.
    """

    def get_att_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)
