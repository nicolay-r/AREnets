from arenets.context.architectures.base.ian_base import IANBase
from arenets.sample import InputSample


class IANFrames(IANBase):
    """
    Based on a frames in context.
    """

    def get_aspect_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)
