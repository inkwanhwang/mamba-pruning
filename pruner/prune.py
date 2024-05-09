from .wanda import wanda
from .sparsegpt import sparsegpt
from .magnitude import mag

magnitude = None

class Pruner:
    def __init__(self):
        self.maskedParameter = list()
        self.scores = {}
        
    def loadPruner(self, method):
        prune_method = {
            'wanda' : wanda,
            'magnitude' : mag,
            'sparsegpt' : sparsegpt
        }
        return prune_method[method]