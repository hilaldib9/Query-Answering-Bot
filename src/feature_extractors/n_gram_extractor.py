from fe_base import FeatureExtractor
import feature_extractor


class NGramExtractor(FeatureExtractor):
    def get_headers(self, extra_data_dict={}):
        headers = []
        for word in self.output.keys():
            headers.append("Has(" + word + ")")
        return headers

    def get_tokens(self, string_lines, extra_data_dict={}):
        self.wordsets = []

        n_array = extra_data_dict["nvalues"]
        n_requirements = extra_data_dict["nrequirements"]

        for sentence in string_lines:
            sentence = sentence.split('|')[1]
            sentence = feature_extractor.process_text(sentence)  # Preprocess lines before read.

            self.wordsets.append(sentence)

        self.output = {}

        for n in n_array:
            # Go through each sentence.
            for index in range(0, len(self.wordsets) - 1):
                # Do n-grams
                sentence = self.wordsets[index]
                string = sentence.split(' ')
                for i in range(len(string) - n + 1):
                    g = ' '.join(string[i:i + n])
                    self.output.setdefault(g, 0)
                    self.output[g] += 1

        words_remove = []
        for key in self.output:
            count = int(self.output.get(key))

            num_words = len(key.split(" "))

            # This is to say, if unigram, check requirement against pos 0 in requirements.
            # Bigram, check against pos 1, etc...
            if count < n_requirements[num_words - 1]:
                words_remove.append(key)

        for key in words_remove:
            del self.output[key]

        return self.output.keys()

    def extract_features(self, string_io, line, extra_data_dict={}):
        # Doesn't do anything
        pass
