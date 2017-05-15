import string
import numpy as np
from cStringIO import StringIO
import re
import spacy
import profanity
from grammarity import Grammarity
from math import tanh
import sys
import hashlib

sys.path.append("counting_algorithms/")
from standard_presence_boolean import StandardPresenceBoolean

sys.path.append("feature_extractors/")
from n_gram_extractor import NGramExtractor
from word_count_extractor import WordCountExtractor

class FeatureExtractor:
    '''
    Class responsible for extracting the feature set from a given file
    '''

    def __init__(self, file, debug=False, count_algorithm=StandardPresenceBoolean(), load_spacy=True):
        if load_spacy:
            self.load_parser()

        self.file = file
        self.debug = debug

        # Changing these values seemed to have effects on accuracy. More testing needed
        self.n_grams = [1, 2, 3, 4]  # Types of ngrams to run
        self.gram_minimums = [1, 1, 1, 1]  # unigram, bigram, trigram minimums

        self.sentences = []
        self.word_set = {}

        self.count_algorithm = count_algorithm

        self.feature_extractors = [
            NGramExtractor(),
            WordCountExtractor()
        ]

    def load(self):

        self.extract_sentences_from_file()
        self.extract_word_features_dataset()
        self.extract_header_set()

        if self.debug:
            print("List: \n" + str(self.token_dict.keys()))

    def extract_sentences_from_file(self):

        '''
        Creates an array of sentences from a text file if possible
        '''

        with open(self.file) as f:
            self.sentences = f.readlines()

    def load_parser(self):

        print("LOADING SPACY EN CORPUS")

        self.parser = spacy.en.English()

        print("LOAD COMPLETE")

    @staticmethod
    def prediction_vector(extracted_features):
        to_return = []
        for entry in string.split(extracted_features, ","):
            to_return.append(float(entry))
        return to_return

    def extract_word_features_phrase(self, line):

        '''
        Get features from an individual phrase
        '''

        line = process_text(line)  # Put line through pre-processing

        file_str = StringIO()

        # Do token extraction process. Don't need to rebuild token dictionary. It was built in the dataset method.
        for token in self.token_array:
            file_str.write(str(self.count_algorithm.score(token, line)))
            file_str.write(",")

        # Do feature extraction process
        for extractor in self.feature_extractors:
            extractor.extract_features(file_str, line)

        return file_str.getvalue()[:-1]  # Removes the trailing comma (left here because no classification on end)

    def extract_word_features_dataset(self, show_raw_values=False):

        '''
        Loads the features for the data set.
        '''

        self.extract_sentences_from_file()
        # grm = Grammarity(self.sentences)
        # self.token_dict = grm.dictionary([1, 2, 3], self.gram_minimums)
        self.features = []

        self.token_array = []

        # Build token dictionary for entire feature extractor
        for extractor in self.feature_extractors:
            extractor_dict = extractor.get_tokens(self.sentences,
                                                  {"nvalues": self.n_grams, "nrequirements": self.gram_minimums})
            self.token_array.extend(extractor_dict)

        # Parse featureset
        for line in self.sentences:
            parts = string.split(line, "|")
            classification = parts[0]
            content = process_text(parts[1])  # Put content through pre-processing

            index = 0

            file_str = StringIO()

            for token in self.token_array:
                file_str.write(str(self.count_algorithm.score(token, line)))
                file_str.write(",")

            for extractor in self.feature_extractors:
                extractor.extract_features(file_str, line)
            file_str.write(classification)

            if show_raw_values:
                print("write:\n" + file_str.getvalue())
            self.features.append(file_str.getvalue())

    # Get array of headers for dataset; Needed for model evaluator.
    def extract_header_set(self):

        '''
        Loads the array of the headers for the data set.
        '''

        self.headers = []

        for extractor in self.feature_extractors:
            self.headers.extend(extractor.get_headers())

        self.headers.append("Classification")


################ PREPROCESS UTILITIES

def get_synonym_matrix():
    # temporary dependency injection
    matrix = {
        #"greeting": ['hey', 'hi', 'hello', 'howdy', 'wagwan', 'yo', 'sup',
        #             'holla', 'hiya', "g'day", 'salutations', 'cheerio',
        #             'heil', 'greetings'],
        "item": ["sword", "boots", "armor", "cup", "mug", "shield", "spear", "hammer",
                 "shirt", "potion", "stuff", "wares", "items", "item"],
        "relation": ["friend", "enemy", "daughter", "son", 
                     "mother", "father", "hero", "nemesis", 
                     "brother", "sister", "pet", "family",
                     "step-son", "step-father", "step-mother"],
        "body_part": ["eyes", "legs", "hands", "arms", "chest", "back", "hair",
                      "nose"],
        "location": ["beach", "forest", "streets", "alleys", "hills",
                     "mountains"],
        #"profanity": profanity.get_profane_words()
    }
    return matrix;


def exchange_synonyms(document):
    '''
    Horribly inefficient way to replace all synonym matrix values with 
    their key. 
    
    input: 
    @synonym_matrix: { "flavor" : ["chocolate, vanilla"]}
    @document: "I love chocolate"
    
    output:
    "I love flavor"
    '''
    
    m = hashlib.md5()
    synonym_matrix = get_synonym_matrix()
    
    for key in synonym_matrix.keys():
        for word in synonym_matrix[key]:
            m.update(word)
            document = document.replace(" " + word + " ", " " + m.hexdigest() + " ")
            document = document.replace(" " + word, " " + m.hexdigest())
            document = document.replace(word + " ", m.hexdigest() + " ")
            
    return document


def has_word(w):
    '''
    BUGGED
    
    Checks if a word exists within a string. Word must be its own entity.

    > has_word("stone")("Mary had a walstone")
    false
    '''

    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search


def process_text(document, formatting=True):
    '''
    Method run to pre-process all lines which are then sent through feature-extractor
    Exchanges text tokens with relevant synonyms and removes any symbols
    Updated to include more consistent formatting and feel.
    Ex: Before changes below, " I want to buy an apple" yielded a classification of undo.
    "I want to buy an apple" (no space) conversely yielded 'buy'

    The accuracy values below are after all calls to process_text in this class have been changed to putting in
    raw content (no formatting down outside of this method)
    AND
    The extract_word_set method has been made to rely on this method as well

    These changes seem to have made accuracy scores much more consistent, generally.
    The only variability seems to come with the more dynamic classifiers (like PAC) which have some natural
        variability to their scoring method

    Accuracy changes:
        - The first two together had an accuracy of 61% favoring Linear Regression or Passive Agressive (rarely)
        - Adding the third brought accuracy up to 64% favoring Linear Regression or Passive Agressive (rarely)
        - Adding the fourth brought it up to 75% (woah!) accuracy favoring Linear Regression or PAC (rarely)
        - Adding start and stop tags brought it to 72% favoring Linear Regression. I still think this change is good
            for consistency in word order and feel, like we talked about. With the fix to these tags done above,
            this accuracy was increased to 75% again.
    '''

    #document = re.sub('[^a-zA-Z1-9\s]', '', document)  # Remove non-alphanumeric characters
    '''
    commented this out, are we sure we want to do this ^? it had no effect on 
    the accuracy score, and I feel like it would be useful in identifying gibberish
    and features such as "sword ?" and "mother ?" might be useful.
    '''
    
    document = document.lower()  # To lower case
    document = document.replace('\n', "")
    document = document.replace('\t', "")
    
    punctuation_list = [',', '.', '!', '-','?']
    
    for token in punctuation_list:
        document = document.replace(token, " " + token + " ")
            
    if formatting:
        document = "<s> " + document + " </s>"  # Add start and end tags to combat changes caused by lower casing
    
    document = document.strip()  # Remove leading and trailing spaces
    document = re.sub('\s{2,}', ' ', document) #replace white space sequences with one white space
    #document = exchange_synonyms(document)  # Take out synonyms
    
    return document


def count_occurrences(word, sentence):
    '''
    Returns the number of times a token appears in a sentence
    '''

    return sentence.lower().split().count(word)

################ END PREPROCESS UTILITIES
