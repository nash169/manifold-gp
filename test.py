import numpy as np


def top_k(x, k):
    return np.argpartition(x, k)[..., -k:]


def sample_k(logits, k):
    u = np.random.uniform(size=np.shape(logits))
    z = -np.log(-np.log(u))
    return top_k(logits + z, k)
