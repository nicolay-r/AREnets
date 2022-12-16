import numpy as np


class NpzRepositoryUtils(object):

    @staticmethod
    def load(source):
        data = np.load(source)
        return data['arr_0']
