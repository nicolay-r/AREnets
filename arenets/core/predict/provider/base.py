class BasePredictProvider(object):

    def provide(self, sample_id_with_uint_labels_iter, labels_count):
        raise NotImplementedError()
