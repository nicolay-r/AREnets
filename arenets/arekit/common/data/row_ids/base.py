class BaseIDProvider(object):
    """
    Opinion in text is a sequence of opinions in context
    o1, o2, o3, ..., on

    o1 -- first_text_opinion
    i -- index in lined (for example: i=3 => 03)

    # TODO. This should be definitely refactored. This implementation
      TODO. combines opinion-based and sample-based data sources, which allows
      TODO. us to bypass such connection via external foreign keys.

      Since we are head to remove opinions, there is a need to refactor so in a
      way of an additional column that provides such information for further connection
      between rows of different storages.
    """

    SEPARATOR = '_'
    OPINION = "o{}" + SEPARATOR

    # region 'parse' methods

    @staticmethod
    def _parse(row_id, pattern):
        assert(isinstance(pattern, str))

        _from = row_id.index(pattern[0]) + 1
        _to = row_id.index(BaseIDProvider.SEPARATOR, _from, len(row_id))

        return int(row_id[_from:_to])

    @staticmethod
    def parse_opinion_in_sample_id(sample_id):
        assert(isinstance(sample_id, str))
        return BaseIDProvider._parse(sample_id, BaseIDProvider.OPINION)

    # endregion