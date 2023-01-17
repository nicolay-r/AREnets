from arenets.context.configurations.base.base import DefaultNetworkConfig


def modify_config(config):
    """ This function allows you to customize parameters of
        the DefaultNetworkConfig class and moreover, all the
        derivatives classes from it, if necessary.
    """
    assert(isinstance(config, DefaultNetworkConfig))
    config.modify_terms_per_context(200)                # Input size.
    config.modify_dropout_keep_prob(0.8)                # Dropout probability for keeping in case of output layer.
