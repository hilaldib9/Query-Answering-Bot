import sys

sys.path.append("counting_algorithms/")
from hyperbolic_tangent_count import HyperbolicTangentCount

'''
A script to test input manually on the AI
'''

from alpha_layer import AlphaLayer

alpha = AlphaLayer("../dataset/dialogue.txt", debug=False, count_algorithm=HyperbolicTangentCount(),
                   load_spacy=True)

# Don't print twice
if not alpha.debug:
    print("Accuracy " + str(alpha.accuracy_score) + " with model " + str(alpha.model_array))

while True:

    sentence = raw_input("\nSentence:")

    # Easy way to exit without error
    if "end" in sentence.lower():
        exit(0)

    features = alpha.fe.extract_word_features_phrase(sentence)
    vector = alpha.fe.prediction_vector(features)
    results = alpha.mb.get_predictions(vector)

    print(results)

    # Continue to avoid errors with spacy being unloaded
    if not alpha.spacy_loaded:
        continue

    if 'buy' in results:
        print(alpha.handle_buy_item(sentence))
    elif 'sell' in results:
        print(alpha.handle_sell(sentence))
    elif 'shop' in results:
        print(alpha.handle_open_shop(sentence))
    elif 'convo' in results:
        print(alpha.handle_conversation(sentence))
    elif 'undo' in results:
        print(alpha.handle_undo(sentence))
