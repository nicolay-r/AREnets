from enum import IntEnum


class PartOfSpeechType(IntEnum):

    NOUN = 1
    ADV = 2
    ADVPRO = 3
    ANUM = 4
    APRO = 5
    COM = 6
    CONJ = 7
    INTJ = 8
    NUM = 9
    PART = 10
    PR = 11
    ADJ = 12
    SPRO = 13
    VERB = 14

    Unknown = 15

    Empty = 16


class PartOfSpeechTypesService(object):

    __pos_names = {
        "S": PartOfSpeechType.NOUN,
        "ADV": PartOfSpeechType.ADV,
        "ADVPRO": PartOfSpeechType.ADVPRO,
        "ANUM": PartOfSpeechType.ANUM,
        "APRO": PartOfSpeechType.APRO,
        "COM": PartOfSpeechType.COM,
        "CONJ": PartOfSpeechType.CONJ,
        "INTJ": PartOfSpeechType.INTJ,
        "NUM": PartOfSpeechType.NUM,
        "PART": PartOfSpeechType.PART,
        "PR": PartOfSpeechType.PR,
        "A": PartOfSpeechType.ADJ,
        "SPRO": PartOfSpeechType.SPRO,
        "V": PartOfSpeechType.VERB,
        "UNKN": PartOfSpeechType.Unknown,
        "EMPTY": PartOfSpeechType.Empty}

    @staticmethod
    def get_mystem_pos_count():
        return len(PartOfSpeechTypesService.__pos_names)

