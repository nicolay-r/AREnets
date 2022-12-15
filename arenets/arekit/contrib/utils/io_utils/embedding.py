from os.path import join

from arenets.arekit.contrib.utils.io_utils.utils import check_targets_existence
from arenets.arekit.contrib.utils.np_utils.embedding import NpzEmbeddingHelper
from arenets.arekit.contrib.utils.np_utils.vocab import VocabRepositoryUtils
from arenets.core.embedding_io import BaseEmbeddingIO


class NpEmbeddingIO(BaseEmbeddingIO):
    """ Npz-based IO utils for embedding and text-based for vocabulary.
        This format represents a archived version of the numpy math data, i.e. vectors, numbers, etc.

        Provides additional Input/Output paths generation functions for:
            - embedding matrix;
            - embedding vocabulary.
    """

    TERM_EMBEDDING_FILENAME_TEMPLATE = 'term_embedding-{cv_index}'
    VOCABULARY_FILENAME_TEMPLATE = "vocab-{cv_index}.txt"

    def __init__(self, target_dir):
        assert(isinstance(target_dir, str))
        self.__target_dir = target_dir

    # region Embedding-related data

    def load_vocab(self):
        source = self.___get_vocab_source()
        return dict(VocabRepositoryUtils.load(source))

    def load_embedding(self):
        source = self.__get_term_embedding_source()
        return NpzEmbeddingHelper.load_embedding(source)

    def check_targets_existed(self):
        targets = [
            self.__get_default_vocab_filepath(),
            self.__get_term_embedding_target()
        ]
        return check_targets_existence(targets=targets)

    # endregion

    # region embedding-related data

    def ___get_vocab_source(self):
        """ It is possible to load a predefined embedding from another experiment
            using the related filepath provided by model_io.
        """
        return self.__get_default_vocab_filepath()

    def __get_term_embedding_target(self):
        return self.__get_default_embedding_filepath()

    def __get_term_embedding_source(self):
        return self.__get_default_embedding_filepath()

    def __get_default_vocab_filepath(self):
        return join(self.__target_dir,
                    self.VOCABULARY_FILENAME_TEMPLATE.format(cv_index=0))

    def __get_default_embedding_filepath(self):
        return join(self.__target_dir,
                    self.TERM_EMBEDDING_FILENAME_TEMPLATE.format(cv_index=0) + '.npz')

    # endregion
