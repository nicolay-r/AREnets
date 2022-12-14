from collections import OrderedDict

from arenets.arekit.common.data.input.sample import InputSampleBase
from arenets.core.debug import DebugKeys
from arenets.core.feeding.bags.bag import Bag
from arenets.core.feeding.batch.base import MiniBatch


class MultiInstanceMiniBatch(MiniBatch):

    def __init__(self, bags, batch_id=None):
        super(MultiInstanceMiniBatch, self).__init__(bags, batch_id)

    def to_network_input(self, provide_labels):
        assert(isinstance(provide_labels, bool))

        result = OrderedDict()

        for bag_index, bag in enumerate(self.iter_by_bags()):
            assert(isinstance(bag, Bag))
            for sample_index, sample in enumerate(bag):
                assert(isinstance(sample, InputSampleBase))
                for arg, value in sample:
                    if arg not in result:
                        result[arg] = [[None] * len(bag) for _ in range(len(self.Bags))]
                    result[arg][bag_index][sample_index] = value

        for bag in self.iter_by_bags():
            if self.I_LABELS not in result:
                result[self.I_LABELS] = []
            result[self.I_LABELS].append(bag.UintBagLabel if provide_labels else 0)

        if DebugKeys.MiniBatchShow:
            MiniBatch.debug_output(result)

        return result
