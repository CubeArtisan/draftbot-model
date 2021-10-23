import json
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd
from tqdm.rich import tqdm, trange

from mtgdraftbots.ml.generators import load_npy_to_tensor

with open('data/maps/int_to_card.json', 'rb') as fp:
    old_int_to_card = json.load(fp)


def find_used_cards(card_indices, old_int_to_new_int=None, counter=1):
    old_int_to_new_int = old_int_to_new_int or [0 for _ in old_int_to_card] + [0]
    row_len = len(card_indices[0])
    for row_idx in trange(len(card_indices), unit='row', unit_scale=True, smoothing=1e-05, dynamic_ncols=True):
        for col_idx in range(row_len):
            card_idx = card_indices[row_idx, col_idx]
            if old_int_to_new_int[card_idx] == 0:
                if card_idx == 0:
                    break
                old_int_to_new_int[card_idx] = counter
                counter += 1
            card_indices[row_idx, col_idx] = old_int_to_new_int[card_idx]
    return card_indices, old_int_to_new_int, counter


def dedupe_directory2(directory):
    card_arrays = []
    array_paths = []
    for subtype in ('training', 'validation'):
        subdirectory = directory/f'{subtype}_parsed_picks'
        with open(subdirectory / 'counts.json') as count_file:
            counts = json.load(count_file)
            pair_count = counts['pairs']
            context_count = counts['contexts']
            print(f'Starting on {subdirectory/"seen.npy.zstd"}.')
            path = subdirectory/f'seen.npy.zstd'
            card_arrays.append(load_npy_to_tensor(path)[:count].astype(np.uint16))
            array_paths.append(path.with_suffix('.zstd2'))
    original_sizes = [arr.size for arr in card_arrays]
    original_shapes = [arr.shape for arr in card_arrays]
    flat_card_array = np.uint16([0])
    while len(card_arrays) > 0:
        flat_card_array = np.concatenate([flat_card_array, card_arrays[-1].reshape((-1,))])
        del card_arrays[-1]
    del card_arrays
    print(f'{flat_card_array.size:,}')
    print('Starting to run unique')
    new_int_to_old_int, new_card_array = np.unique(flat_card_array, return_inverse=True)
    new_int_to_old_int[new_card_array[0]], new_int_to_old_int[0] = new_int_to_old_int[0], new_int_to_old_int[new_card_array[0]]
    del new_card_array[1:]
    card_arrays = []
    for size, shape in zip(original_sizes, original_shapes):
        card_arrays.append(new_card_array[:size].reshape(shape))
        del new_card_array[:size]
    del new_card_array
    old_int_to_new_int = np.zeros((len(old_int_to_card) + 1,), dtype=np.uint16)
    for i, old_int in enumerate(new_int_to_old_int):
        old_int_to_new_int[old_int] = i
    for subtype in ('training', 'validation'):
        subdirectory = directory/f'{subtype}_parsed_picks'
        with open(subdirectory / 'counts.json') as count_file:
            counts = json.load(count_file)
            pair_count = counts['pairs']
            context_count = counts['contexts']
        for filename, count in zip(('pairs', 'cards_in_pack', 'picked'), (pair_count, context_count,
                                                                          context_count)):
            print(f'Starting on {subdirectory/f"{filename}.npy.zstd"}.')
            path = subdirectory/f'{filename}.npy.zstd'
            card_arrays.append(old_int_to_new_int[load_npy_to_tensor(path)[:count]])
            array_paths.append(path.with_suffix('.zstd2'))
    cctx = zstd.ZstdCompressor(level=10, threads=-1)
    for array, path in zip(card_arrays, array_paths):
        with open(path, 'wb') as fh:
            with cctx.stream_writer(fh) as compressor:
                np.save(compressor, array, allow_pickle=False)
    int_to_card = [old_int_to_card[x - 1] for x in new_int_to_old_int[1:]]
    with open(directory/'int_to_card.json', 'w') as fp:
        json.dump(int_to_card, fp)
    card_to_int = {card['oracle_id']: i for i, card in enumerate(int_to_card)}
    with open(directory/'card_to_int.json', 'w') as fp:
        json.dump(card_to_int, fp)


def dedupe_directory3(directory):
    card_arrays = []
    array_paths = []
    for subtype in ('training', 'validation'):
        subdirectory = directory/f'{subtype}_parsed_picks'
        with open(subdirectory / 'counts.json') as count_file:
            counts = json.load(count_file)
            pair_count = counts['pairs']
            context_count = counts['contexts']
        for filename, count in zip(('pairs', 'cards_in_pack', 'picked', 'seen'), (pair_count, context_count,
                                                                                  context_count, context_count)):
            path = subdirectory/f'{filename}.npy.zstd'
            print(f'Loading {path}.')
            card_arrays.append(load_npy_to_tensor(path)[:count].astype(np.uint16))
            array_paths.append(path.with_suffix('.zstd2'))
    included = np.zeros(len(old_int_to_card) + 1, dtype=bool)
    for arr, path in zip(card_arrays, array_paths):
        print(f'Getting the values from {path}.')
        included[arr] = 1
    old_int_to_new_int = np.zeros(len(old_int_to_card) + 1, dtype=np.uint16)
    count_unique = included.sum()
    print(f'There are {count_unique} unique cards in the data.')
    old_int_to_new_int[included] = np.arange(count_unique, dtype=np.uint16)
    new_int_to_old_int = np.zeros(count_unique, dtype=np.uint16)
    for old_index, new_index in enumerate(old_int_to_new_int):
        if old_index != 0:
            new_int_to_old_int[new_index] = old_index
    cctx = zstd.ZstdCompressor(level=20, threads=-1)
    for array, path in zip(card_arrays, array_paths):
        print(f'Translating and saving {path}.')
        array = old_int_to_new_int[array]
        with open(path, 'wb') as fh:
            with cctx.stream_writer(fh) as compressor:
                np.save(compressor, array, allow_pickle=False)
    int_to_card = [old_int_to_card[x - 1] for x in new_int_to_old_int[1:]]
    with open(directory/'int_to_card.json', 'w') as fp:
        json.dump(int_to_card, fp)
    card_to_int = {card['oracle_id']: i for i, card in enumerate(int_to_card)}
    with open(directory/'card_to_int.json', 'w') as fp:
        json.dump(card_to_int, fp)


def dedupe_directory(directory):
    old_int_to_new_int = [0 for _ in old_int_to_card] + [0]
    counter = 1
    cctx = zstd.ZstdCompressor(level=10, threads=-1)
    for subtype in ('training', 'validation'):
        subdirectory = directory/f'{subtype}_parsed_picks'
        with open(subdirectory / 'counts.json') as count_file:
            counts = json.load(count_file)
            pair_count = counts['pairs']
            context_count = counts['contexts']
        for filename, count in zip(('pairs', 'cards_in_pack', 'picked', 'seen'), (pair_count, context_count,
                                                                                  context_count, context_count)):
            print(f'Starting on {subdirectory/f"{filename}.npy.zstd"}.')
            arr, old_int_to_new_int, counter = find_used_cards(
                load_npy_to_tensor(subdirectory/f'{filename}.npy.zstd')[:count], old_int_to_new_int,
                counter)
            with open(subdirectory/f'{filename}.npy.zstd.2', 'wb') as fh:
                with cctx.stream_writer(fh) as compressor:
                    np.save(compressor, arr.astype(np.uint16), allow_pickle=False)
    max_old_index
    old_indices_kept = [i for i, x in enumerate(old_int_to_new_int) if x]
    int_to_card = [None for _ in range(counter)]
    for new_index, old_index in enumerate(old_int_to_new_int):
        int_to_card[new_index + 1] = old_int_to_card[old_index - 1]
    with open(directory/'int_to_card.json', 'w') as fp:
        json.dump(int_to_card, fp)
    card_to_int = {card['oracle_id']: i for i, card in enumerate(int_to_card)}
    with open(directory/'card_to_int.json', 'w') as fp:
        json.dump(card_to_int, fp)


if __name__ == '__main__':
    dedupe_directory3(Path(sys.argv[1]))
