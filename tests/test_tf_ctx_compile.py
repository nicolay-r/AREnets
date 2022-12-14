import logging
import sys
import unittest

sys.path.append('../')

from tests.pos import PartOfSpeechTypesService
from tests.tf_networks.supported import get_supported
from tests.tf_networks.utils import init_config
from arenets.context.configurations.base.base import DefaultNetworkConfig


class TestContextNetworkCompilation(unittest.TestCase):

    def test(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        pos_items_count = PartOfSpeechTypesService.get_mystem_pos_count()

        for config, network in get_supported():
            assert(isinstance(config, DefaultNetworkConfig))
            config.modify_classes_count(3)
            config.set_pos_count(pos_items_count)

            logger.info("Compile: {}".format(type(network)))
            logger.info("Clases count: {}".format(config.ClassesCount))

            init_config(config=config, pos_items_count=pos_items_count)
            network.compile(config, reset_graph=True, graph_seed=42)


if __name__ == '__main__':
    unittest.main()
