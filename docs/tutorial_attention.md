# Declaring your model with attention mechanism

In this tutorial illustrates implementation of `CNN` model with self-attention mechanism.

Every model in AREnets instantiate the `SingleInstanceNeuralNetwork` class which 
leaves some methods as non-declared.
The common principle of the custom model development is to
declare:
* `init_body_dependent_hidden_states` -- list of hidden states utilized in networks body.
* `init_logits_hidden_states` -- declaration of the output computation
* `init_context_embedding` -- body and architecture declaration
* `init_logits_unscaled` -- implementation of the output based on the declared hidden states

Every model presented by **architecture** and **configuration**.
Let's pick the default CNN model, and import them:

```python
from arenets.context.architectures.cnn import VanillaCNN
from arenets.context.configurations.cnn import CNNConfig
```

Next, in order to adopt attention, we declare a derived class from the CNN.
There is a method `convolved_transformation_optional` which could be 
overriden by usage and calculation of alpha weights as follows:

```python
import tensorflow as tf
from arenets.attention import common
from arenets.attention.architectures.self_p_zhou import self_attention_by_peng_zhou


class SelfAttentionCNN(VanillaCNN):

    def __init__(self):
        super(SelfAttentionCNN, self).__init__()
        # Declare alphas for further output logging purposes.
        self.__att_alphas = None

    def get_attention_alphas(self, input_data):
        # Using Peng-Zhou predefined Self-attention mechanism.
        return self_attention_by_peng_zhou(input_data)
        
    def iter_input_dependent_hidden_parameters(self):
        # Provide attention for logging.
        for name, value in super(SelfAttentionCNN, self).iter_input_dependent_hidden_parameters():
            yield name, value
        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_alphas

    def convolved_transformation_optional(self, value):
        # Calculating attention alphas based on the convolved information of the transformed input.
        self.__att_alphas = self.get_attention_alphas(value)
        return value * tf.expand_dims(self.__att_alphas, -1)
```

Then there is a need to provide the related configuration file.

```python
import tensorflow as tf

class AttentionSelfPZhouCNNConfig(CNNConfig):

    def __init__(self):
        super(AttentionSelfPZhouCNNConfig, self).__init__()

    @property
    def BiasInitializer(self):
        return tf.constant_initializer(0.1)

    @property
    def WeightInitializer(self):
        return tf.contrib.layers.xavier_initializer()
```

Finally the developed model and configuration could be passed into `train(...)` function in oder to use it.