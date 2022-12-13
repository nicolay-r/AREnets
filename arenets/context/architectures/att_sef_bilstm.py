import tensorflow as tf
from arenets.context.architectures.base.att_bilstm_base import AttentionBiLSTMBase
from arenets.sample import InputSample


class AttentionSynonymEndsAndFramesBiLSTM(AttentionBiLSTMBase):

    def get_att_input(self):
        combined = tf.concat([self.get_input_parameter(InputSample.I_SYN_SUBJ_INDS),
                              self.get_input_parameter(InputSample.I_SYN_OBJ_INDS),
                              self.get_input_parameter(InputSample.I_FRAME_INDS)],
                             axis=-1)
        return tf.contrib.framework.sort(combined, direction='DESCENDING')
