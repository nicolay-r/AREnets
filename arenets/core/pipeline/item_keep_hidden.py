from arenets.arekit.common.pipeline.context import PipelineContext
from arenets.core.idhv_collection import NetworkInputDependentVariables
from arenets.core.pipeline.item_base import EpochHandlingPipelineItem


class MinibatchHiddenFetcherPipelineItem(EpochHandlingPipelineItem):

    def __init__(self):
        super(MinibatchHiddenFetcherPipelineItem, self).__init__()
        self.__idh_names = None
        self.__idh_tensors = None
        self.__input_dependent_params = None

    @property
    def InputDependentParams(self):
        return self.__input_dependent_params

    def before_epoch(self, model_context, data_type):
        super(MinibatchHiddenFetcherPipelineItem, self).before_epoch(
            model_context=model_context, data_type=data_type)

        self.__idh_names, self.__idh_tensors = map(
            list, zip(*self._context.Network.iter_input_dependent_hidden_parameters()))

        self.__input_dependent_params = NetworkInputDependentVariables()

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        minibatch = input_data
        feed_dict = self._context.create_feed_dict(minibatch=minibatch, data_type=self._data_type)
        idh_values = self._context.Session.run(self.__idh_tensors, feed_dict=feed_dict)

        if not (len(self.__idh_names) > 0 and len(idh_values) > 0):
            return

        self.__input_dependent_params.add_input_dependent_values(
            names_list=self.__idh_names,
            tensor_values_list=idh_values,
            text_opinion_ids=[sample.ID for sample in minibatch.iter_by_samples()],
            bags_per_minibatch=self._context.Config.BagsPerMinibatch,
            bag_size=self._context.Config.BagSize)
