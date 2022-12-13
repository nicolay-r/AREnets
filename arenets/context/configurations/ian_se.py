from arenets.context.configurations.base.ian_base import IANBaseConfig


class IANSynonymEndsBasedConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return self.SynonymsPerContext * 2