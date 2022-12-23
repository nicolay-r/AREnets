from os.path import join

from arenets.arekit.common.data.input.reader import BaseReader
from arenets.arekit.common.experiment.api.base_samples_io import BaseSamplesIO
from arenets.arekit.contrib.utils.io_utils.utils import check_targets_existence, filename_template


class SamplesIO(BaseSamplesIO):
    """ Samples default IO utils for samples.
        Sample is a text part which include pair of attitude participants.
        This class allows to provide saver and loader for such entries, dubbed as samples.
        Samples required for machine learning training/inferring.
    """

    def __init__(self, target_dir, reader=None, prefix="sample"):
        assert(isinstance(target_dir, str))
        assert(isinstance(prefix, str))
        assert(isinstance(reader, BaseReader) or reader is None)
        self.__target_dir = target_dir
        self.__prefix = prefix
        self.__reader = reader

    # region public methods

    @property
    def Reader(self):
        return self.__reader

    def create_target(self, data_type):
        return self.__get_input_sample_target(data_type)

    def check_targets_existed(self, data_types_iter):
        for data_type in data_types_iter:

            targets = [
                self.__get_input_sample_target(data_type=data_type),
            ]

            if not check_targets_existence(targets=targets):
                return False
        return True

    # endregion

    def __get_input_sample_target(self, data_type):
        template = filename_template(data_type=data_type)
        return self.__get_filepath(out_dir=self.__target_dir,
                                   template=template,
                                   prefix=self.__prefix,
                                   extension=self.__reader.target_extension())

    @staticmethod
    def __get_filepath(out_dir, template, prefix, extension):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        assert(isinstance(extension, str))
        return join(out_dir, "{prefix}-{template}{extension}".format(
            prefix=prefix, template=template, extension=extension))
