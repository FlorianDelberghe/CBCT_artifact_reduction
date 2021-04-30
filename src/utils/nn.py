import random
from contextlib import contextmanager

import torch


@contextmanager
def evaluate(*models):
    """Context manager to evaluate models disables grad computation and sets models to eval"""

    try:
        torch.set_grad_enabled(False)
        [model.eval() for model in models]
        yield None

    finally:
        torch.set_grad_enabled(True)
        [model.train() for model in models]


class TrSampler(torch.utils.data.Sampler):
    """Samplers to limit the number of training samples"""
        
    def __init__(self, dataset_len, n_samples=10):
        self.dataset_len = dataset_len
        self.n_samples = int(dataset_len *n_samples) if 0 < n_samples <= 1 else n_samples
        
        # Returns the same samples for every iter
        self.samples = random.sample(range(self.dataset_len), self.n_samples)

    def __len__(self):
        return self.n_samples

    def __iter__(self):    
        return iter(random.sample(self.samples, len(self.samples)))


class ValSampler(torch.utils.data.Sampler):
    """Samplers to avoid going through the entire validation dataset each time"""
        
    def __init__(self, dataset_len, n_samples=100, fixed_samples=True):
        self.dataset_len = dataset_len
        self.n_samples = int(dataset_len *n_samples) if 0 < n_samples <= 1 else n_samples
        self.fixed_samples = fixed_samples
        
        # Returns the same samples in the same order for every iter
        if fixed_samples:
            self.samples = random.sample(range(self.dataset_len), self.n_samples)

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        if self.fixed_samples:
            return iter(self.samples)
            
        return iter(random.sample(range(self.dataset_len), self.n_samples))