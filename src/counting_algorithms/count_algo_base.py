from abc import ABCMeta, abstractmethod


class CountAlgoBase:
    __metaclass__ = ABCMeta

    @abstractmethod
    def score(self, token, string):
        pass
