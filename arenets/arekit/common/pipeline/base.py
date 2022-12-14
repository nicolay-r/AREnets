from arenets.arekit.common.pipeline.context import PipelineContext
from arenets.arekit.common.pipeline.items.base import BasePipelineItem


class BasePipeline(object):

    def __init__(self, pipeline):
        assert(isinstance(pipeline, list))
        self.__pipeline = pipeline

    def run(self, input_data, params_dict=None):
        assert(isinstance(params_dict, dict) or params_dict is None)

        pipeline_ctx = PipelineContext(params_dict if params_dict is not None else dict())

        for item in filter(lambda itm: itm is not None, self.__pipeline):
            assert(isinstance(item, BasePipelineItem))
            input_data = item.apply(input_data=input_data, pipeline_ctx=pipeline_ctx)

        return input_data

    def append(self, item):
        assert(isinstance(item, BasePipelineItem))
        self.__pipeline.append(item)
