import logging
import numpy as np

from arenets.arekit.common.utils import progress_bar_iter
from arenets.arekit.contrib.utils.np_utils.embedding import NpzEmbeddingHelper
from arenets.arekit.contrib.utils.np_utils.vocab import VocabRepositoryUtils
from arenets.core.embedding_io import BaseEmbeddingIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def convert_text_embedding_if_needed(txt_embedding_filepath, embedding_io):
    """ This is a formatter for the Word2Vec model, which generates the
        vocabulary and term embedding for a given model name presented
        in a form of the text. Usually, this is a file, named as `model.txt`

        Designed for processing TXT-based formats for the following resource:
        http://vectors.nlpl.eu/repository/
    """
    assert(isinstance(txt_embedding_filepath, str))
    assert(isinstance(embedding_io, BaseEmbeddingIO))

    vocab = []
    vectors = []
    shape = None

    logger.info("Converting original text-based embedding: {}".format(txt_embedding_filepath))
    with open(txt_embedding_filepath, "r") as f:
        for line_index, line in enumerate(progress_bar_iter(f.readlines(), unit="words")):

            args = line.split()
            if line_index == 0:
                shape = (int(args[0]), int(args[1]))
                continue

            word = args[0]
            assert(word != "")
            vector = [float(i) for i in args[1:]]
            vocab.append([word, line_index])
            vectors.append(vector)

    # Save the formatted versions.
    VocabRepositoryUtils.save(data=vocab, target=embedding_io.get_vocab_filepath())
    NpzEmbeddingHelper.save_embedding(data=np.concatenate(vectors).reshape(shape),
                                      target=embedding_io.get_embedding_filepath())