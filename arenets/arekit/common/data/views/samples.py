from arenets.arekit.common.data import const
from arenets.arekit.common.data.storages.base import BaseRowsStorage


class LinkedSamplesStorageView(object):

    def iter_from_storage(self, storage):
        assert(isinstance(storage, BaseRowsStorage))
        undefined = -1

        linked = []
        current_opinion_id = undefined

        for row_index, row_dict in storage:

            # We consider that every row is unique, so we do not have linked opinions
            # in case of the absence of OPINION_INDEX parameter.
            opinion_id = row_index if const.OPINION_INDEX not in row_dict else row_dict[const.OPINION_INDEX]

            if current_opinion_id != undefined:
                if opinion_id != current_opinion_id:
                    yield linked
                    linked = []
                    current_opinion_id = opinion_id
            else:
                current_opinion_id = opinion_id

            linked.append(row_dict)

        if len(linked) > 0:
            yield linked