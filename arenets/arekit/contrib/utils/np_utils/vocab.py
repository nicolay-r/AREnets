import logging

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VocabRepositoryUtils(object):

    @staticmethod
    def load(source):
        vocab = np.loadtxt(source, dtype=str)
        logger.info("Loading vocabulary [size={size}]: {filepath}".format(size=len(vocab), filepath=source))
        return vocab
