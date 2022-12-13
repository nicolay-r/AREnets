import tensorflow as tf
from arenets.context.architectures.base.att_cnn_base import AttentionCNNBase
from arenets.sample import InputSample


class AttentionEndsCNN(AttentionCNNBase):

    def get_att_input(self):
        return tf.stack([self.get_input_parameter(InputSample.I_SUBJ_IND),
                         self.get_input_parameter(InputSample.I_OBJ_IND)],
                        axis=-1)

