import io
import json
import math

import numpy as np
import tensorflow as tf
import zstandard as zstd


def load_npy_to_tensor(path):
    with zstd.open(path, 'rb') as fh:
        filedata = io.BytesIO(fh.readall())
        pool_npy = np.load(filedata)
        del filedata
    return pool_npy


class PickPairGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, folder, epochs_per_completion, seed=37):
        with open(folder / 'counts.json') as count_file:
            counts = json.load(count_file)
            pair_count = counts['pairs']
            context_count = counts['contexts']
        self.seed = seed
        self.pair_count = pair_count
        self.batch_size = batch_size
        self.seen = load_npy_to_tensor(folder/'seen.npy.zstd')
        self.pool = load_npy_to_tensor(folder/'picked.npy.zstd')
        self.pairs = load_npy_to_tensor(folder/'pairs.npy.zstd')
        self.context_idxs = load_npy_to_tensor(folder/'context_idxs.npy.zstd')
        self.coords = load_npy_to_tensor(folder/'coords.npy.zstd')
        self.coord_weights = load_npy_to_tensor(folder/'coord_weights.npy.zstd')
        self.y_idx = load_npy_to_tensor(folder/'y_idx.npy.zstd')
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.shuffled_indices = np.arange(self.pair_count)
        self.epoch_count = -1
        self.epochs_per_completion = epochs_per_completion
        self.on_epoch_end()
        self.original_indices = np.copy(self.shuffled_indices)

    def reset_rng(self):
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.shuffled_indices = np.copy(self.original_indices)
        self.epoch_count = 0

    def __len__(self):
        return self.pair_count // self.batch_size // self.epochs_per_completion

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.epoch_count % self.epochs_per_completion == 0:
            self.rng.shuffle(self.shuffled_indices)

    def __getitem__(self, idx):
        idx += (self.epoch_count % self.epochs_per_completion) * len(self)
        indices = self.shuffled_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        context_idxs = self.context_idxs[indices]
        result = (self.pairs[indices], self.pool[context_idxs], self.seen[context_idxs],
                  self.coords[context_idxs], self.coord_weights[context_idxs], self.y_idx[context_idxs])
        return (result, self.y_idx[context_idxs])


class PickGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, folder, epochs_per_completion, seed=29):
        with open(folder / 'counts.json') as count_file:
            counts = json.load(count_file)
            self.context_count = counts['contexts']
        self.batch_size = batch_size
        self.seed = seed
        self.seen = load_npy_to_tensor(folder/'seen.npy.zstd')
        print('seen', self.seen.shape, self.seen.dtype)
        self.pool = load_npy_to_tensor(folder/'picked.npy.zstd')
        print('pool', self.pool.shape, self.pool.dtype)
        self.coords = load_npy_to_tensor(folder/'coords.npy.zstd')
        print('coords', self.coords.shape, self.coords.dtype)
        self.coord_weights = load_npy_to_tensor(folder/'coord_weights.npy.zstd')
        print('coord_weights', self.coord_weights.shape, self.coord_weights.dtype)
        self.y_idx = load_npy_to_tensor(folder/'y_idx.npy.zstd')
        print('y_idx', self.y_idx.shape, self.y_idx.dtype)
        self.cards_in_pack = load_npy_to_tensor(folder/'cards_in_pack.npy.zstd')
        print('cards_in_pack', self.cards_in_pack.shape, self.cards_in_pack.dtype)
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.shuffled_indices = np.arange(self.context_count)
        # We call on_epoch_end immediately so this'll become 0 and be an accurate count.
        self.epoch_count = -1
        self.epochs_per_completion = epochs_per_completion
        self.on_epoch_end()

    def reset_rng(self):
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.epoch_count = -1
        self.on_epoch_end()

    def __len__(self):
        idx_base, idx_max = self.get_epoch_context_counts()
        return math.ceil((idx_max - idx_base) / self.batch_size)

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.epoch_count % self.epochs_per_completion == 0:
            self.rng.shuffle(self.shuffled_indices)

    def get_epoch_context_counts(self):
        pos_in_cycle = self.epoch_count % self.epochs_per_completion
        contexts_per_epoch_f = self.context_count / self.epochs_per_completion
        idx_base = math.ceil(pos_in_cycle * contexts_per_epoch_f)
        idx_max = min(math.ceil((pos_in_cycle + 1) * contexts_per_epoch_f), self.context_count)
        return idx_base, idx_max

    def __getitem__(self, idx):
        idx = min(idx, len(self) - 1)
        idx_base, idx_max = self.get_epoch_context_counts()
        min_idx_offset = idx * self.batch_size + idx_base
        max_idx_offset = min(min_idx_offset + self.batch_size, idx_max)
        context_idxs = self.shuffled_indices[min_idx_offset:max_idx_offset]
        result = (self.cards_in_pack[context_idxs], self.pool[context_idxs],
                  self.seen[context_idxs], self.coords[context_idxs],
                  self.coord_weights[context_idxs], self.y_idx[context_idxs])
        return (result, self.y_idx[context_idxs])
