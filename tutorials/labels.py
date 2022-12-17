from collections import OrderedDict

from arenets.arekit.common.labels.base import Label
from arenets.arekit.common.labels.scaler import BaseLabelScaler


class NoLabel(Label):
    pass


class PositiveTo(Label):
    pass


class NegativeTo(Label):
    pass


class PosNegNeuRelationsLabelScaler(BaseLabelScaler):

    def __init__(self):
        self.__int_to_label_dict = OrderedDict([(NoLabel(), 0), (PositiveTo(), 1), (NegativeTo(), -1)])
        self.__uint_to_label_dict = OrderedDict([(NoLabel(), 0), (PositiveTo(), 1), (NegativeTo(), 2)])
        super(PosNegNeuRelationsLabelScaler, self).__init__(
            int_dict=self.__int_to_label_dict, uint_dict=self.__uint_to_label_dict)
