import collections
import logging
from os.path import exists

from arenets.arekit.common.data_type import DataType
from arenets.arekit.common.folding.base import BaseDataFolding
from arenets.arekit.contrib.utils_folding import experiment_iter_index

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_targets_existence(targets):
    assert (isinstance(targets, collections.Iterable))

    result = True
    for filepath in targets:
        assert(isinstance(filepath, str))

        existed = exists(filepath)
        logger.info("Check existence [{is_existed}]: {fp}".format(is_existed=existed, fp=filepath))
        if not existed:
            result = False

    return result


def filename_template(data_type, data_folding):
    assert(isinstance(data_type, DataType))
    assert(isinstance(data_folding, BaseDataFolding))
    return "{data_type}-{iter_index}".format(data_type=data_type.name.lower(),
                                             iter_index=experiment_iter_index(data_folding))
