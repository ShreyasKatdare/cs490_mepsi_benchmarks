from random import sample
from .base import BaseSelector


class RandomSelector(BaseSelector):
    def __init__(self, node_count):
        super(RandomSelector, self).__init__(node_count)

    def pruning(self, size):
        return sample(range(self.node_count), size)


class AllSelector(BaseSelector):
    def __init__(self, node_count):
        super(AllSelector, self).__init__(node_count)

    def pruning(self):
        return list(range(self.node_count))
