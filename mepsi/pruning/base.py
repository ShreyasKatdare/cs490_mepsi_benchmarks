class BaseSelector:
    def __init__(self, node_count):
        self.node_count = node_count

    def pruning(self, size):
        raise NotImplementedError("pruning is not implemented!")
