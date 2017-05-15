from abc import ABCMeta, abstractmethod


class FeatureExtractor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_headers(self, extra_data_dict={}):
        pass

    @abstractmethod
    def get_tokens(self, string_lines, extra_data_dict={}):
        pass

    @abstractmethod
    def extract_features(self, string_io, line, extra_data_dict={}):
        pass
