import collections

from arenets.arekit.common.data import const
from arenets.core.predict.provider.base import BasePredictProvider


class IdAndBinaryLabelsPredictProvider(BasePredictProvider):
    """ This is a default output provider which returns t
        the row identifier and labels in a form of the one-hot
        vector, in which all values are zeros instead of the
        i-th -- the related class.
    """

    @staticmethod
    def __iter_contents(sample_id_with_uint_labels_iter, labels_count):
        assert(isinstance(labels_count, int))

        for sample_id, uint_label in sample_id_with_uint_labels_iter:
            assert(isinstance(uint_label, int))

            # Composing row contents.
            contents = [sample_id]

            # Composing labels.
            labels = ['0'] * labels_count
            labels[uint_label] = '1'

            # Providing row labels.
            contents.extend(labels)
            yield contents

    def provide(self, sample_id_with_uint_labels_iter, labels_count):
        assert(isinstance(sample_id_with_uint_labels_iter, collections.Iterable))
        assert(isinstance(labels_count, int))

        # Provide contents.
        contents_it = self.__iter_contents(sample_id_with_uint_labels_iter=sample_id_with_uint_labels_iter,
                                           labels_count=labels_count)

        # Provide title.
        title = [const.ID]
        title.extend([str(label) for label in range(labels_count)])

        return title, contents_it
