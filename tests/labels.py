from collections import OrderedDict

from arekit.common.labels.scaler.sentiment import SentimentLabelScaler


class Label(object):

    def __eq__(self, other):
        assert(isinstance(other, Label))
        return type(self) == type(other)

    def __ne__(self, other):
        assert(isinstance(other, Label))
        return type(self) != type(other)

    def __hash__(self):
        return hash(self.to_class_str())

    def to_class_str(self):
        return self.__class__.__name__


class NoLabel(Label):
    pass


class TestNeutralLabel(NoLabel):
    pass


class TestPositiveLabel(Label):
    pass


class TestNegativeLabel(Label):
    pass


class TestThreeLabelScaler(SentimentLabelScaler):

    def __init__(self):

        uint_labels = [(TestNeutralLabel(), 0),
                       (TestPositiveLabel(), 1),
                       (TestNegativeLabel(), 2)]

        int_labels = [(TestNeutralLabel(), 0),
                      (TestPositiveLabel(), 1),
                      (TestNegativeLabel(), -1)]

        super(TestThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                                   int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)
