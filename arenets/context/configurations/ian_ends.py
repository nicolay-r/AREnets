from arenets.context.configurations.base.ian_base import IANBaseConfig


class IANEndsBasedConfig(IANBaseConfig):

    @property
    def MaxAspectLength(self):
        return 2
