from arenets.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arenets.core.feeding.bags.collection.single import SingleBagsCollection
from arenets.core.feeding.batch.base import MiniBatch
from arenets.core.feeding.batch.multi import MultiInstanceMiniBatch


def create_batch_by_bags_group(bags_collection_type, bags_group):
    if issubclass(bags_collection_type, SingleBagsCollection):
        return MiniBatch(bags_group)
    if issubclass(bags_collection_type, MultiInstanceBagsCollection):
        return MultiInstanceMiniBatch(bags_group)
