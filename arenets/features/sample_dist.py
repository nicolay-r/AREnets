from arenets.features.utils import create_zeros


class DistanceFeatures(object):
    """
    Distance features for Relation Extraction allows to account a distance between attitude
    ends and other words in context
    Proposed in paper: https://www.aclweb.org/anthology/C14-1220/
    """

    @staticmethod
    def distance_feature(position, size):
        result = create_zeros(size)
        for i in range(len(result)):
            result[i] = i - position if i - position >= 0 else i - position + size
        return result

    @staticmethod
    def distance_abs_nearest_feature(positions, size):
        result = create_zeros(size)
        for i in range(len(result)):
            result[i] = min([abs(i - p) for p in positions])
        return result
