import io
import json

import numpy as np
import tensorflow as tf
import zstandard as zstd


def load_npy_to_tensor(path):
    with zstd.open(path, 'rb') as fh:
        filedata = io.BytesIO(fh.readall())
        picked_npy = np.load(filedata)
        del filedata
    return picked_npy
    # result = tf.convert_to_tensor(picked_npy)
    # del picked_npy
    # print('loaded', path, result.dtype, result.shape, 4 * tf.size(result, out_type=tf.int64).numpy() / 1024 / 1024 / 1024)
    # return result


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
        self.picked = load_npy_to_tensor(folder/'picked.npy.zstd')
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
        result = (self.pairs[indices], self.picked[context_idxs], self.seen[context_idxs],
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
        self.picked = load_npy_to_tensor(folder/'picked.npy.zstd')
        self.coords = load_npy_to_tensor(folder/'coords.npy.zstd')
        self.coord_weights = load_npy_to_tensor(folder/'coord_weights.npy.zstd')
        self.chosen_idx = load_npy_to_tensor(folder/'chosen_idx.npy.zstd')
        self.y_idx = load_npy_to_tensor(folder/'y_idx.npy.zstd')
        self.cards_in_pack = load_npy_to_tensor(folder/'cards_in_pack.npy.zstd')
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
        return self.context_count // (self.batch_size * self.epochs_per_completion)

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.epoch_count % self.epochs_per_completion == 0:
            self.rng.shuffle(self.shuffled_indices)

    def __getitem__(self, idx):
        idx += (self.epoch_count % self.epochs_per_completion) * len(self)
        context_idxs = self.shuffled_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        result = (self.cards_in_pack[context_idxs], self.picked[context_idxs],
                  self.seen[context_idxs], self.coords[context_idxs],
                  self.coord_weights[context_idxs], self.chosen_idx[context_idxs], self.y_idx[context_idxs])
        return (result, self.y_idx[context_idxs])
