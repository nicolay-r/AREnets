from arenets.arekit.common.data import const
from arenets.arekit.common.data.row_ids.base import BaseIDProvider
from arenets.arekit.common.data.storages.base import BaseRowsStorage


class LinkedSamplesStorageView(object):

    def __init__(self, row_ids_provider):
        assert(isinstance(row_ids_provider, BaseIDProvider))
        self.__row_ids_provider = row_ids_provider

    def iter_from_storage(self, storage):
        assert(isinstance(storage, BaseRowsStorage))
        undefined = -1

        linked = []
        current_opinion_id = undefined

        for _, row_dict in storage:
            sample_id = str(row_dict[const.ID])
            opinion_id = self.__row_ids_provider.parse_opinion_in_sample_id(sample_id)
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