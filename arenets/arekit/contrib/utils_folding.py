from arenets.arekit.contrib.two_class import TwoClassCVFolding


def folding_iter_states(folding):
    if isinstance(folding, TwoClassCVFolding):
        for state in folding.iter_states():
            yield state
    yield 0
