import tensorflow as tf
from arenets.context.configurations.cnn import CNNConfig


class AttentionSelfPZhouCNNConfig(CNNConfig):

    def __init__(self):
        super(AttentionSelfPZhouCNNConfig, self).__init__()

    # region properties

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()

    # endregion
