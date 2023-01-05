from collections import OrderedDict

from arenets.arekit.common.labels.base import Label
from arenets.arekit.common.labels.scaler import BaseLabelScaler


class NoLabel(Label):
    pass


class TestNeutralLabel(NoLabel):
    pass


class TestPositiveLabel(Label):
    pass


class TestNegativeLabel(Label):
    pass


class SentimentLabelScaler(BaseLabelScaler):

    def __init__(self):
        int_to_label = OrderedDict([(TestNeutralLabel(), 0), (TestPositiveLabel(), 1), (TestNegativeLabel(), -1)])
        uint_to_label = OrderedDict([(TestNeutralLabel(), 0), (TestPositiveLabel(), 1), (TestNegativeLabel(), 2)])
        super(SentimentLabelScaler, self).__init__(int_to_label, uint_to_label)


class TestThreeLabelScaler(SentimentLabelScaler):

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)

    def get_no_label_instance(self):
        return self.int_to_label(0)
