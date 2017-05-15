from feature_extractor import FeatureExtractor
from model_evaluator import ModelEvaluator
from model_builder import ModelBuilder
from cStringIO import StringIO
from wordsToNum import parse
import sys
import spacy
import random
import sys

sys.path.append("counting_algorithms/standard_presence_boolean.py")
from standard_presence_boolean import StandardPresenceBoolean


class AlphaLayer:
    # Default constructor including path and debug toggle. Also includes count algorithm default and specification
    # This constructor does have a way to turn off the load of spacy. This is not meant for production and
    # should only be done to speed up debug times.
    def __init__(self, path, debug=False, count_algorithm=StandardPresenceBoolean(), load_spacy=True):
        self.spacy_loaded = load_spacy
        self.path = path
        self.debug = debug

        self.fe = FeatureExtractor(path, self.debug, count_algorithm=count_algorithm, load_spacy=load_spacy)
        
        self.load(True)
        
        if self.debug:
            print("Accuracy score: " + str(
                self.accuracy_score) + " with classifier " + self.chosen_model + " out of " + str(self.model_array))

    def load(self, select_new_best_model=False):
        
        '''
        Reloads data from the file and selects the best model.
        
        Useful when there are automated updates to datasets.
        '''
        
        self.fe.load()
        self.me = ModelEvaluator(self.fe.headers, self.fe.features)        
        
        if select_new_best_model:
            self.me = ModelEvaluator(self.fe.headers, self.fe.features)
            self.model_array, self.highest_rate = self.me.search_initial_best_fit_algorithm()
            self.chosen_model = self.model_array[
                random.randint(0, len(self.model_array) - 1)]
            self.mb = ModelBuilder(self.chosen_model)

        self.X_train, self.X_validation, self.Y_train, self.Y_validation = self.me.split_dataset()
        self.mb.fit_model(self.X_train, self.Y_train)
        self.accuracy_score = self.mb.accuracy_score(self.X_validation,
                                                     self.Y_validation)

    # Change dataset path
    def change_path(self, path):
        self.path = path
        self.fe.path = path

    # Append a line to the dataset. Caution: no formatting checks are done in this method.
    def add_line(self, line):
        with open(self.path, "a") as datafile:
            datafile.write('\n' + line)

    def handle_buy_item(self, sentence):
        return "Got classifier: buy\nThank you for purchasing " + self.evaluate(sentence)
        
    def handle_open_shop(self, sentence):
        return "Got classifier: shop\nHere you go, take a look at my wares.\n"
    
    def handle_conversation(self, sentence):
        return "Got classifier: convo\nI don't feel like talking to you"
    
    def handle_undo(self, sentence):
        return "Got classifier: undo\nReally mate? You sure you want to take it back?"

    def handle_sell(self, sentence):
        return "Got classifier: sell\nI'll gladly accept your " + self.evaluate(sentence)

    # Evaluate a string with spacy classifier
    def evaluate(self, line):
        doc = self.fe.parser(unicode(line))
        file_str = StringIO()
        current_string = ""
        compound_number = ""
        for word in doc:
            if word.pos_ == u'NOUN' or word.pos_ == u'PROPN':
                # Probably the thing we want to buy
                current_string += word.text + " "
            if word.pos_ == u'NUM':
                # This is an amount.
                compound_number += word.text + " "
            if word.pos_ == u'CONJ':
                # Consider this termination of the current item. Record amount and such.
                if current_string != "":
                    # Only terminate if they specified a thing to terminate with
                    current_amount = parse(compound_number.strip())
                    file_str.write(str(current_amount))
                    file_str.write(" ")
                    file_str.write(current_string.strip())
                    file_str.write("; ")
                    current_string = ""
            if word.pos_ == u'PUNCT':
                if word.text == u';':
                    current_amount = parse(compound_number.strip())
                    file_str.write(str(current_amount))
                    file_str.write(" ")
                    file_str.write(current_string.strip())
                    file_str.write("; ")
                    current_string = ""
                if word.text == u',':
                    current_amount = parse(compound_number.strip())
                    file_str.write(str(current_amount))
                    file_str.write(" ")
                    file_str.write(current_string.strip())
                    file_str.write("; ")
                    current_string = ""
        if current_string != "":
            current_amount = parse(compound_number.strip()) if compound_number.strip() != "" else 1
            file_str.write(str(current_amount))
            file_str.write(" ")
            file_str.write(current_string.strip())
        return file_str.getvalue()        