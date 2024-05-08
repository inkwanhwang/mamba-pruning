class pruner:
    def __init__(self, maskedParameter):
        self.maskedParameter = list(maskedParameter)
        self.scores = {}
        
def loadPruner(method):
    prune_method = {
        'wanda' : wanda,
        'wandaA' : wandaA,
        'magnitude' : magnitude,
        'magnitudeA' : magnitudeA,
        'sparseGPT' : spraseGPT,
        'sparseGPTA' : spraseGPTA
    }
    return prune_method[method]