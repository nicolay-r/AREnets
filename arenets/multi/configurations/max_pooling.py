from arenets.multi.configurations.base import BaseMultiInstanceConfig


class MaxPoolingOverSentencesConfig(BaseMultiInstanceConfig):

    def __init__(self, context_config):
        super(MaxPoolingOverSentencesConfig, self).__init__(context_config)

