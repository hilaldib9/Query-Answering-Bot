# so much cleaner in the other file.

def get_profane_words():
    with open("../dataset/profanity.txt") as f:
        return f.readlines()
