import random
import numpy as np
import torch
from torch.utils.data import Sampler

class NLPSampler(Sampler):
    """ Sampler for variable-length data."""
    def __init__(self, data_source, tokens_per_batch, shuffle=True):
        super().__init__(data_source)
        lengths = [len(feature['input_ids']) + len(feature['decoder_input_ids']) for feature in data_source]
        assert tokens_per_batch >= max(lengths)
        self.idxs = np.arange(len(data_source))
        self.lengths = np.array(lengths)
        self.tokens_per_batch = tokens_per_batch
        self.shuffle = shuffle

        self.rebatch()
        
    def rebatch(self):
        p = np.random.permutation(len(self.lengths))
        self.idxs = self.idxs[p]
        self.lengths = self.lengths[p]
        # sort data into batches
        sort = np.argsort(-self.lengths)
        self.idxs = self.idxs[sort]
        self.lengths = self.lengths[sort]

        current_idx = 0
        self.batches = []
        while current_idx < len(sort):
            largest_sample = self.lengths[current_idx]
            batch_size = max(1, self.tokens_per_batch // largest_sample)

            if current_idx + batch_size > len(sort):
                batch = self.idxs[current_idx:]
                current_idx = len(sort)
            else:
                batch = self.idxs[current_idx:current_idx + batch_size]
                current_idx += batch_size

            self.batches.append(batch)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class NLPEvalSampler(Sampler):
    """ Sampler for variable-length data."""
    def __init__(self, data_source, tokens_per_batch):
        super().__init__(data_source)
        lengths = [len(feature['input_ids']) + len(feature['decoder_input_ids']) for feature in data_source]
        assert tokens_per_batch >= max(lengths)
        self.lengths = np.array(lengths)
        self.tokens_per_batch = tokens_per_batch

        self.rebatch()
        
    def rebatch(self):
        current_len = 0
        current_batch = []
        self.batches = []
        for i in range(len(self.lengths)):
            if current_len + self.lengths[i] > self.tokens_per_batch:
                self.batches.append(current_batch)
                current_len = 0
                current_batch = []

            current_batch.append(i)
            current_len += self.lengths[i] 

        if current_batch:
            self.batches.append(current_batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def _get_train_sampler(self) -> Sampler:
    return NLPSampler(self.train_dataset, self.args.tokens_per_batch)

def _get_eval_sampler(self, eval_dataset) -> Sampler:
    return NLPEvalSampler(eval_dataset, self.args.tokens_per_batch)

def prep_batch_for_collating(batch, collate_fn, **kwargs):
    new_batch = []
    
    firstkey = next(iter(batch[0]))
    batch_size = len(batch[0][firstkey])
    # print("batch size:", batch_size)
    for b in range(batch_size):
        bdict = {}
        for key in batch[0]:
            bdict[key] = batch[0][key][b]

        new_batch.append(bdict)

    return collate_fn(new_batch, **kwargs)