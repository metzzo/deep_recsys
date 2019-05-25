import numpy as np


class SplitStrategy(object):
    def sample(self, array):
        raise NotImplementedError()


class RandomSampleStrategy(SplitStrategy):
    def __init__(self, split, ):
        self.split = split

    def sample(self, array):
        samples_drawn = int(len(array) * self.split)
        import random
        indices = sorted(random.sample(range(len(array)), samples_drawn))
        return array[indices]


class AllSamplesExceptStrategy(SplitStrategy):
    def __init__(self, exclude):
        self.exclude = set(exclude)

    def sample(self, array):
        return np.array([x for x in array if x not in self.exclude])


class AllSamplesIncludeStrategy(SplitStrategy):
    def __init__(self, include):
        self.include = set(include)

    def sample(self, array):

        return np.array([x for x in array if x in self.include])
