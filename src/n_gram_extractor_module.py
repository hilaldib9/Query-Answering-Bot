from feature_extractor import FeatureExtractor
from model_evaluator import ModelEvaluator
from model_builder import ModelBuilder
from cStringIO import StringIO
from wordsToNum import parse
import sys
import spacy
import random
import sys

MIN_NGRAM_LIMIT = 4

max_accuracy = 0
max_model = None

matrix = []    
        

def test_model(n_gram_mins):
    
    fe = FeatureExtractor("../dataset/slack_dialogue.txt", 
                                               n_grams=[1,2,3,4],
                                               n_gram_mins=n_gram_mins,
                                               debug = False)
    fe.load()
    me = ModelEvaluator(fe.headers, fe.features)        
    
    model_array, highest_rate = me.search_initial_best_fit_algorithm()
    chosen_model = model_array[
        random.randint(0, len(model_array) - 1)]
    mb = ModelBuilder(chosen_model)

    X_train, X_validation, Y_train, Y_validation = me.split_dataset()
    mb.fit_model(X_train, Y_train)
    accuracy_score = mb.accuracy_score(X_validation,
                                                 Y_validation) 
    
    print("Got score: " + str(accuracy_score) + " with model: " + str(model_array))
    print("Using : " +  str(n_gram_mins))
    return accuracy_score, model_array


    
    

for a in range(1, MIN_NGRAM_LIMIT):
    for b in range(1, MIN_NGRAM_LIMIT):
        for c in range(1, MIN_NGRAM_LIMIT):
            for d in range(1, MIN_NGRAM_LIMIT):
                

                mins = [a,b,c,d]
                
                accuracy_score, model = test_model(mins)
                
                if accuracy_score > max_accuracy:
                    max_accuracy = accuracy_score
                    max_model = str(model)
                
print("Max accuracy: \n" + max_accuracy)
print("Model: \n" + model)