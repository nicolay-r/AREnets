from collections import OrderedDict


class InputSampleBase(object):
    """
    Description of a single sample (context) of a model
    """

    def __init__(self, shift_index_dbg, input_sample_id, values):
        assert(isinstance(shift_index_dbg, int))
        assert(isinstance(input_sample_id, str))
        assert(isinstance(values, list))
        self._shift_index_dbg = shift_index_dbg
        self.__input_sample_id = input_sample_id
        self.__values = OrderedDict(values)

    # region properties

    @property
    def ID(self):
        return self.__input_sample_id

    # endregion

    @staticmethod
    def _check_ends_could_be_fitted_in_window(actual_dist, window):
        return actual_dist < window

    def __iter__(self):
        for key, value in self.__values.items():
            yield key, value
