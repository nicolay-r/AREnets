import tensorflow as tf
from arenets.context.architectures.base.ian_base import IANBase
from arenets.sample import InputSample


class IANEndsBased(IANBase):
    """
    Based on a pair of attitude ends: <Object, Subject>
    """

    def get_aspect_input(self):
        return tf.stack([self.get_input_parameter(InputSample.I_SUBJ_IND),
                         self.get_input_parameter(InputSample.I_OBJ_IND)],
                        axis=-1)
