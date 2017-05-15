import feature_extractor


class Grammarity:
    # Class created to make n-gram extracton a more obvious and explicit endeavor

    def __init__(self, sentence_set, debug=False):
        self.sentences = sentence_set

        self.debug = debug

        self.wordsets = []  # Array to hold wordsets. Each entry represents on phrase in the dataset tokenized

        for sentence in self.sentences:
            sentence = sentence.split('|')[1]
            sentence = feature_extractor.process_text(sentence)  # Preprocess lines before read.

            self.wordsets.append(sentence)

    def dictionary(self, n_array, requirements_array):
        output = {}

        # Go through each desired n-gram amount
        for n in n_array:
            # Go through each sentence.
            for index in range(0, len(self.wordsets) - 1):
                # Do n-grams
                sentence = self.wordsets[index]
                string = sentence.split(' ')
                for i in range(len(string) - n + 1):
                    g = ' '.join(string[i:i + n])
                    output.setdefault(g, 0)
                    output[g] += 1

        words_remove = []
        for key in output:
            if self.debug:
                print(key)
            count = int(output.get(key))

            num_words = len(key.split(" "))

            # This is to say, if unigram, check requirement against pos 0 in requirements.
            # Bigram, check against pos 1, etc...
            if count < requirements_array[num_words - 1]:
                words_remove.append(key)
                if self.debug:
                    print("Removed: " + key)

        for key in words_remove:
            del output[key]

        return output
