from namt.methods.baselines import ASRQ75, MLSDMedian, PoCA
from namt.methods.namt import NAMT3pMLP, NAMT4pMLP

REGISTRY = {
    m.name: m
    for m in (PoCA, ASRQ75, MLSDMedian, NAMT3pMLP, NAMT4pMLP)
}


def get(name, dev="cuda:0", **kwargs):
    return REGISTRY[name](dev=dev, **kwargs)
