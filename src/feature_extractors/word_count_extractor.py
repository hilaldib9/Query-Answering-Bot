from fe_base import FeatureExtractor
import feature_extractor


class WordCountExtractor(FeatureExtractor):
    def get_headers(self, extra_data_dict={}):
        return ["Length"]

    def get_tokens(self, string_lines, extra_data_dict={}):
        # Doesn't do this
        return []

    def extract_features(self, string_io, line, extra_data_dict={}):
        count = len(line.split(" "))
        string_io.write(str(count))
        string_io.write(",")
