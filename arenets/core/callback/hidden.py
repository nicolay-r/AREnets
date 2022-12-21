from os.path import join

from arenets.core.base_writer import BaseWriter
from arenets.core.callback.base import NetworkCallback


class HiddenStatesWriterCallback(NetworkCallback):

    def __init__(self, log_dir, writer):
        assert(isinstance(writer, BaseWriter))
        super(HiddenStatesWriterCallback, self).__init__()

        self.__epochs_passed = 0
        self.__log_dir = log_dir
        self.__writer = writer

    @staticmethod
    def __target_provider(log_dir, name, epoch_index=None):
        target_parts = ["hparams", name]
        if epoch_index is not None:
            target_parts += [f'e{epoch_index}']
        return join(log_dir, "_".join(target_parts))

    def __write(self, pipeline, epoch_index=None):
        model_ctx = pipeline[0].ModelContext
        names, tensors = map(list, zip(*model_ctx.Network.iter_hidden_parameters()))
        values = model_ctx.Session.run(tensors)

        assert(isinstance(names, list))
        assert(isinstance(values, list))
        assert(len(names) == len(values))

        for value_index, name in enumerate(names):
            self.__writer.write(data=values[value_index],
                                target=HiddenStatesWriterCallback.__target_provider(
                                    log_dir=self.__log_dir, name=name, epoch_index=epoch_index))

    def on_epoch_finished(self, pipeline, operation_cancel):
        super(HiddenStatesWriterCallback, self).on_epoch_finished(pipeline=pipeline,
                                                                  operation_cancel=operation_cancel)

        self.__epochs_passed += 1

        if len(pipeline) == 0:
            return

        self.__write(pipeline=pipeline, epoch_index=self.__epochs_passed)

    def on_predict_finished(self, pipeline):
        super(HiddenStatesWriterCallback, self).on_predict_finished(pipeline=pipeline)
        
        self.__write(pipeline=pipeline)
