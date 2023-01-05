import random
import sys
import unittest
import tensorflow as tf
import logging

sys.path.append('../')

from tests.labels import TestThreeLabelScaler
from tests.tf_networks.supported import get_supported
from tests.pos import PartOfSpeechTypesService
from tests.tf_networks.utils import init_config

from arenets.arekit.common.labels.scaler import BaseLabelScaler
from arenets.arekit.common.data_type import DataType
from arenets.context.configurations.base.base import DefaultNetworkConfig
from arenets.sample import InputSample
from arenets.core.feeding.bags.bag import Bag
from arenets.core.feeding.batch.base import MiniBatch
from arenets.core.nn import NeuralNetwork


class TestContextNetworkFeeding(unittest.TestCase):

    @staticmethod
    def init_session():
        init_op = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init_op)
        return sess

    @staticmethod
    def __create_minibatch(config, labels_scaler):
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(labels_scaler, BaseLabelScaler))

        bags = []
        for i in range(config.BagsPerMinibatch):
            # Generate labels withing the following region: [0, labels_count)
            uint_label = random.randint(0, labels_scaler.LabelsCount - 1)
            bag = Bag(uint_label=uint_label)
            for j in range(config.BagSize):
                bag.add_sample(InputSample._generate_test(config))
            bags.append(bag)

        return MiniBatch(bags=bags, batch_id=None)

    @staticmethod
    def run_feeding(network, network_config, create_minibatch_func, logger,
                    labels_scaler,
                    display_hidden_values=True,
                    display_idp_values=True):
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(network_config, DefaultNetworkConfig))
        assert(isinstance(labels_scaler, BaseLabelScaler))
        assert(callable(create_minibatch_func))

        pos_items_count = PartOfSpeechTypesService.get_mystem_pos_count()

        # Init config.
        init_config(config=network_config,
                    pos_items_count=pos_items_count)

        # Init network.
        network.compile(config=network_config, reset_graph=True, graph_seed=42, eager=False)
        minibatch = create_minibatch_func(config=network_config, labels_scaler=labels_scaler)

        network_optimiser = network_config.Optimiser.minimize(network.Cost)

        with TestContextNetworkFeeding.init_session() as sess:
            # Save graph
            writer = tf.summary.FileWriter("output", sess.graph)
            # Init feed dict
            feed_dict = network.create_feed_dict(input=minibatch.to_network_input(provide_labels=True),
                                                 data_type=DataType.Train)

            hidden_list = list(network.iter_hidden_parameters())
            idp_list = list(network.iter_input_dependent_hidden_parameters())

            hidden_names = [name for name, _ in hidden_list]
            idp_names = [name for name, _ in idp_list]

            fetches_hidden = [tensor for _, tensor in hidden_list]
            fetches_idp = [tensor for _, tensor in idp_list]

            fetches_default = [network_optimiser, network.Cost, network.Accuracy]

            # feed
            result = sess.run(fetches=fetches_default + fetches_hidden + fetches_idp,
                              feed_dict=feed_dict)

            # Printing graph
            print(result)
            writer.close()

            # Show hidden parameters
            hidden_values = result[len(fetches_default):len(fetches_default) + len(fetches_hidden)]
            for i, value in enumerate(hidden_values):
                if display_hidden_values:
                    logger.info('Value type: {}'.format(type(value)))
                    logger.info('Hidden parameter "{}": {}'.format(hidden_names[i], value))

            # Show idp parameters
            idp = result[len(fetches_default) + len(fetches_hidden):]
            for i, value in enumerate(idp):
                if display_idp_values:
                    logger.info('i: {}'.format(i))
                    logger.info('IDP: {}'.format(type(value)))
                    logger.info('IDP shape: {}'.format(value.shape))
                    logger.info('IDP param/value "{}": {}'.format(idp_names[i], value))

    def test(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)

        labels_scaler = TestThreeLabelScaler()

        for cfg, network in get_supported():
            logger.debug("Feed to the network: {}".format(type(network)))
            self.run_feeding(network=network,
                             network_config=cfg,
                             create_minibatch_func=self.__create_minibatch,
                             labels_scaler=labels_scaler,
                             logger=logger)


if __name__ == '__main__':
    unittest.main()
