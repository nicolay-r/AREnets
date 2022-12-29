class BaseEmbeddingIO(object):
    """ API for loading and saving embedding and vocabulary related data.
    """

    @property
    def UnknownTermIndex(self):
        raise NotImplementedError()

    def load_vocab(self):
        raise NotImplementedError()

    def load_embedding(self):
        raise NotImplementedError()

    def check_targets_existed(self):
        raise NotImplementedError()

    def get_vocab_filepath(self):
        raise NotImplementedError()

    def get_embedding_filepath(self):
        raise NotImplementedError()
