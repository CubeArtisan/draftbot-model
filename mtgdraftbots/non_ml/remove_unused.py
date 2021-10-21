import json
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd
from tqdm.auto import tqdm, trange

from mtgdraftbots.ml.generators import load_npy_to_tensor

with open('data/maps/int_to_card.json', 'rb') as fp:
    old_int_to_card = json.load(fp)

def find_used_cards(card_indices, seen_card_indices=None, old_int_to_new_int=None):
    seen_card_indices = seen_card_indices or set()
    old_int_to_new_int = old_int_to_new_int or {}
    for row_idx in trange(len(card_indices), unit='row', unit_scale=True, smoothing=1e-05, dynamic_ncols=True):
        for col_idx in range(len(card_indices[row_idx])):
            card_idx = card_indices[row_idx][col_idx]
            if card_idx != 0:
                if card_idx not in seen_card_indices:
                    seen_card_indices.add(card_idx)
                    old_int_to_new_int[card_idx] = len(seen_card_indices)
                card_indices[row_idx][col_idx] = old_int_to_new_int[card_idx]
    return card_indices, seen_card_indices, old_int_to_new_int

def dedupe_directory(directory):
    seen_card_indices = set((0,))
    old_int_to_new_int = {0: 0}
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
            arr = find_used_cards(load_npy_to_tensor(subdirectory/f'{filename}.npy.zstd')[:count],
                                  seen_card_indices, old_int_to_new_int)
            with open(subdirectory/f'{filename}.npy.zstd', 'wb') as fh:
                with cctx.stream_writer(fh) as compressor:
                    np.save(compressor, arr)
    int_to_card = [None for _ in range(1, len(old_int_to_new_int))]
    for old_index, new_index in old_int_to_new_int.items():
        if old_index > 0:
            int_to_card[new_index - 1] = old_int_to_card[old_index - 1]
    with open(directory/'int_to_card.json', 'wb') as fp:
        json.dump(int_to_card, fp)
    card_to_int = {card['oracle_id']: i for i, card in enumerate(int_to_card)}
    with open(directory/'card_to_int.json', 'wb') as fp:
        json.dump(card_to_int, fp)


if __name__ == '__main__':
    dedupe_directory(Path(sys.argv[1]))

