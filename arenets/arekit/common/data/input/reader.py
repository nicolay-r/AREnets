class BaseReader(object):

    def read(self, target):
        raise NotImplementedError()

    def target_extension(self):
        raise NotImplementedError()
