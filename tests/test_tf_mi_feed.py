import logging
import sys
import unittest

sys.path.append('../../../')

from tests.test_tf_ctx_feed import TestContextNetworkFeeding
from tests.tf_networks.supported import get_supported
from tests.labels import TestThreeLabelScaler

from arenets.arekit.common.labels.scaler import BaseLabelScaler
from arenets.shapes import NetworkInputShapes
from arenets.multi.configurations.att_self import AttSelfOverSentencesConfig
from arenets.multi.architectures.att_self import AttSelfOverSentences
from arenets.core.feeding.bags.bag import Bag
from arenets.core.feeding.batch.multi import MultiInstanceMiniBatch
from arenets.multi.configurations.max_pooling import MaxPoolingOverSentencesConfig
from arenets.context.configurations.base.base import DefaultNetworkConfig
from arenets.sample import InputSample
from arenets.multi.architectures.max_pooling import MaxPoolingOverSentences


class TestMultiInstanceFeed(unittest.TestCase):

    @staticmethod
    def __create_minibatch(config, labels_scaler):
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(labels_scaler, BaseLabelScaler))

        bags = []
        no_label = labels_scaler.get_no_label_instance()

        shapes = NetworkInputShapes(iter_pairs=[
            (NetworkInputShapes.FRAMES_PER_CONTEXT, config.FramesPerContext),
            (NetworkInputShapes.TERMS_PER_CONTEXT, config.TermsPerContext),
            (NetworkInputShapes.SYNONYMS_PER_CONTEXT, config.SynonymsPerContext)])

        empty_sample = InputSample.create_empty(shapes)

        for i in range(config.BagsPerMinibatch):
            bag = Bag(labels_scaler.label_to_uint(no_label))
            for j in range(config.BagSize):
                bag.add_sample(empty_sample)
            bags.append(bag)

        return MultiInstanceMiniBatch(bags=bags, batch_id=None)

    @staticmethod
    def multiinstances_supported(ctx_config, ctx_network):
        return [
            (MaxPoolingOverSentencesConfig(ctx_config), MaxPoolingOverSentences(ctx_network)),
            (AttSelfOverSentencesConfig(ctx_config), AttSelfOverSentences(ctx_network))
        ]

    def test(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        labels_scaler = TestThreeLabelScaler()

        for ctx_config, ctx_network in get_supported():
            for config, network in self.multiinstances_supported(ctx_config, ctx_network):
                logger.info(type(network))
                logger.info('\t-> {}'.format(type(ctx_network)))
                TestContextNetworkFeeding.run_feeding(network=network,
                                                      network_config=config,
                                                      create_minibatch_func=self.__create_minibatch,
                                                      logger=logger,
                                                      labels_scaler=labels_scaler,
                                                      display_idp_values=False)


if __name__ == '__main__':
    unittest.main()
