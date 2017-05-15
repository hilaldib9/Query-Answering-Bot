from count_algo_base import CountAlgoBase
from math import tanh


class HyperbolicTangentCount(CountAlgoBase):
    # Will be either 0 or 1 depending on if the token is present in string
    def score(self, token, string):
        return tanh(string.count(token))
