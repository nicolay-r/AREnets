class BaseEmbeddingIO(object):
    """ API for loading and saving embedding and vocabulary related data.
    """

    def load_vocab(self):
        raise NotImplementedError()

    def load_embedding(self):
        raise NotImplementedError()

    def check_targets_existed(self):
        raise NotImplementedError()
