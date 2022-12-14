#!/usr/bin/python
import logging
import sys
import unittest

sys.path.append('../')

from tests.pos import PartOfSpeechTypesService
from tests.tf_networks.utils import init_config
from tests.tf_networks.supported import get_supported
from arenets.context.configurations.base.base import DefaultNetworkConfig
from arenets.multi.architectures.max_pooling import MaxPoolingOverSentences
from arenets.multi.configurations.base import BaseMultiInstanceConfig


class TestMultiInstanceCompile(unittest.TestCase):

    @staticmethod
    def mpmi(context_config, context_network):
        assert(isinstance(context_config, DefaultNetworkConfig))

        context_config.modify_classes_count(3)

        logging.info("TEST: {}".format(context_network))
        config = BaseMultiInstanceConfig(context_config)

        config.modify_classes_count(3)

        network = MaxPoolingOverSentences(context_network=context_network)

        pos_items_count = PartOfSpeechTypesService.get_mystem_pos_count()
        init_config(config=config,
                    pos_items_count=pos_items_count)

        network.compile(config, reset_graph=True, graph_seed=42)

    def test(self):
        logging.basicConfig(level=logging.INFO)

        for config, network in get_supported():
            self.mpmi(config, network)


if __name__ == '__main__':
    unittest.main()
