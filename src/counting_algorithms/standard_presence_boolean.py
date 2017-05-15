from count_algo_base import CountAlgoBase


class StandardPresenceBoolean(CountAlgoBase):
    # Will be either 0 or 1 depending on if the token is present in string
    def score(self, token, string):
        if token in string:
            return 1
        else:
            return 0
