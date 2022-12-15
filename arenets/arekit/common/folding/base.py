import collections


class BaseDataFolding(object):
    """ Describes and provides API on how to handle doc_ids during experiment,
        i.e. how many states does nested folding algorithm supports,
        how to perform folding for a particular state (current),
        and how to such state into string.
    """

    def __init__(self, doc_ids_to_fold, supported_data_types):
        assert(isinstance(doc_ids_to_fold, collections.Iterable))
        assert(isinstance(supported_data_types, list))
        self._doc_ids_to_fold_set = set(doc_ids_to_fold)
        self._supported_data_types = supported_data_types

    def iter_supported_data_types(self):
        """ Iterates through data_types, supported in a related experiment
            Note:
            In CV-split algorithm, the first part corresponds to a LARGE split,
            Jand second to small; therefore, the correct sequence is as follows:
            DataType.Train, DataType.Test.
        """
        return iter(self._supported_data_types)
