import glob
import itertools
import json
import mmap
import random
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

with open('data/maps/old_int_to_card.json') as fp:
    old_int_to_card = json.load(fp)
with open('data/maps/card_to_int.json') as fp:
    card_to_int = json.load(fp)
old_int_to_new_int = [card_to_int[c["oracle_id"]] for c in old_int_to_card]
default_basic_ids = [
    "56719f6a-1a6c-4c0a-8d21-18f7d7350b68",
    "b2c6aa39-2d2a-459c-a555-fb48ba993373",
    "bc71ebf6-2056-41f7-be35-b2e5c34afa99",
    "b34bb2dc-c1af-4d77-b0b3-a0fb342a5fc6",
    "a3fb7228-e76b-4e96-a40e-20b5fed75685",
]
default_basics = [card_to_int[c] for c in default_basic_ids]

MAX_PICKED = 120
MAX_SEEN = 480


def pad(arr, desired_length):
    return arr + [0 for _ in range(desired_length - len(arr))]


def interpolate(pickNum, numPicks, packNum, numPacks):
    fpackIdx = 3 * packNum / numPacks
    fpickIdx = 15 * pickNum / numPicks
    floorpackIdx = min(2, int(fpackIdx))
    ceilpackIdx = min(2, floorpackIdx + 1)
    floorpickIdx = min(14, int(fpickIdx))
    ceilpickIdx = min(14, floorpickIdx + 1)
    modpackIdx = fpackIdx - floorpackIdx
    modpickIdx = fpickIdx - floorpickIdx
    coords = ((floorpackIdx, floorpickIdx), (floorpackIdx, ceilpickIdx), (ceilpackIdx, floorpickIdx), (ceilpackIdx, ceilpickIdx))
    weights = ((1 - modpackIdx) * (1 - modpickIdx), (1 - modpackIdx) * modpickIdx, modpackIdx * (1 - modpickIdx), modpackIdx * modpickIdx)
    return coords, weights


def picks_from_draft(draft):
    if 'picks' in draft:
        picks = (([x + 1 for x in pick['cardsInPack']], [x + 1 for x in pick['picked']],
                  [x + 1 for x in pick['seen']],
                  *interpolate(pick['pickNum'], pick['numPicks'], pick['packNum'], pick['numPacks']),
                  pick.get('pickedIdx', pick.get('trashedIdx', None)), 0 if 'pickedIdx' in pick else 1,
                  [x + 1 for x in draft['basics']] if draft['basics'] else default_basics)
                 for pick in draft['picks'] if all(isinstance(x, int) for x in pick['picked']))
        return ((pick[0], pick[1] + 8 * pick[-1], pick[2] + 8 * pick[-1], pick[3], pick[4], pick[5], pick[6])
                for pick in picks if len(pick[1]) + 8 * len(pick[-1]) <= MAX_PICKED and len(pick[2]) + 8 * len(pick[-1]) <= MAX_SEEN)
    else:
        return ()


def load_picks(drafts_file):
    with open(drafts_file) as fp:
        drafts = json.load(fp)
    return itertools.chain.from_iterable(picks_from_draft(draft)
                                         for draft in tqdm(drafts, leave=False, dynamic_ncols=True,
                                                           unit='draft', unit_scale=1))

def picks_from_draft2(draft):
    if 'picks' in draft:
        picks = (([old_int_to_new_int[x] + 1 for x in pick['cardsInPack']], [old_int_to_new_int[x] + 1 for x in pick['picked']],
                  [old_int_to_new_int[x] + 1 for x in pick['seen']],
                  *interpolate(pick['pick'], pick['packSize'], pick['pack'], pick['packs']),
                  pick['cardsInPack'].index(pick['chosenCard']), 0, default_basics)
                 for pick in draft['picks']
                 if all(isinstance(x, int) for x in pick['picked'])
                    and all(isinstance(x, int) for x in pick['cardsInPack'])
                    and all(isinstance(x, int) for x in pick['seen']) and pick['chosenCard'] in pick['cardsInPack'])
        return ((pick[0], pick[1] + 8 * pick[-1], pick[2] + 8 * pick[-1], pick[3], pick[4], pick[5], pick[6])
                for pick in picks if len(pick[1]) + 8 * len(pick[-1]) <= MAX_PICKED and len(pick[2]) + 8 * len(pick[-1]) <= MAX_SEEN)
    else:
        return ()


def load_picks2(drafts_file):
    with open(drafts_file) as fp:
        drafts = json.load(fp)
    return itertools.chain.from_iterable(picks_from_draft2(draft)
                                         for draft in tqdm(drafts, leave=False, dynamic_ncols=True,
                                                           unit='draft', unit_scale=1))


def load_drafts(drafts_dir):
    if ';' in drafts_dir:
        return itertools.chain.from_iterable(load_drafts(draft_dir) for draft_dir in drafts_dir.split(';'))
    else:
        return itertools.chain.from_iterable(load_picks(draft_file) for draft_file
                                             in tqdm(glob.glob(f'{drafts_dir}/*.json'), leave=False,
                                                     dynamic_ncols=True, unit='file', unit_scale=1))


def load_drafts2(drafts_dir):
    if ';' in drafts_dir:
        return itertools.chain.from_iterable(load_drafts2(draft_dir) for draft_dir in drafts_dir.split(';'))
    else:
        return itertools.chain.from_iterable(load_picks2(draft_file) for draft_file
                                             in tqdm(glob.glob(f'{drafts_dir}/*.json'), leave=False,
                                                     dynamic_ncols=True, unit='file', unit_scale=1))


def picks_to_pairs(picks, mm, dest_folder):
    context_count = len(picks)
    pairs = np.memmap(dest_folder/'pairs.npy', dtype=np.int32, mode='w+', shape=(context_count * 24, 2))
    context_idxs = np.memmap(dest_folder/'context_idxs.npy', dtype=np.int32, mode='w+', shape=(context_count * 24,))
    picked = np.memmap(dest_folder/'picked.npy', dtype=np.int32, mode='w+', shape=(context_count, MAX_PICKED))
    seen = np.memmap(dest_folder/'seen.npy', dtype=np.int32, mode='w+', shape=(context_count, MAX_SEEN))
    coords = np.memmap(dest_folder/'coords.npy', dtype=np.int32, mode='w+', shape=(context_count, 4, 2))
    coord_weights = np.memmap(dest_folder/'coord_weights.npy', dtype=np.float32, mode='w+', shape=(context_count, 4))
    y_idx = np.memmap(dest_folder/'y_idx.npy', dtype=np.int32, mode='w+', shape=(context_count,))
    pair_idx = 0
    for context_idx, offset in enumerate(tqdm(picks, leave=False, dynamic_ncols=True, unit='picks', unit_scale=1)):
        mm.seek(offset)
        line = mm.readline()
        try:
            pick = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Invalid JSON `{line.strip()}`')
        selected = pick[5]
        if selected is not None and 0 <= selected < len(pick[0]):
            picked[context_idx] = np.int32(pad(pick[1], MAX_PICKED))
            seen[context_idx] = np.int32(pad(pick[2], MAX_SEEN))
            coords[context_idx] = np.int32(pick[3])
            coord_weights[context_idx] = np.float32(pick[4])
            y_idx[context_idx] = np.int32(pick[6])
            for i, idx in enumerate(pick[0]):
                if i != selected and idx > 0:
                    pairs[pair_idx][0] = pick[0][selected]
                    pairs[pair_idx][1] = idx
                    context_idxs[pair_idx] = context_idx
                    pair_idx += 1
    with open(dest_folder / 'counts.json', 'w') as count_file:
        json.dump(count_file, {"pairs": pair_idx, "contexts": context_count})
    print(dest_folder, pair_idx, context_count)
    cctx = zstd.ZstdCompressor(level=10, threads=-1)
    for name, arr in (('pairs', pairs), ('context_idxs', context_idxs), ('picked', picked), ('seen', seen),
                      ('coords', coords), ('coord_weights', coord_weights), ('y_idx', y_idx)):
        with open(dest_folder / f'{name}.npy.zstd', 'wb') as fh:
            with cctx.stream_writer(fh) as compressor:
                np.save(compressor, arr)
        print(f'Saved {name} with zstd.')


if __name__ == '__main__':
    offsets = []
    with open('data/parsed_picks.json', 'rb+') as of1:
        of1.write(b'\n')
        of1.flush()
        with mmap.mmap(of1.fileno(), 0) as mm:
            mm.resize(1024 * 1024 * 1024)
            with open('data/parsed_pick_offsets.json', 'w') as off1:
                for pick in itertools.chain(load_drafts(sys.argv[1]), load_drafts2(sys.argv[2])):
                    offsets.append(mm.tell())
                    off1.write(f'{offsets[-1]}\n')
                    dumped = json.dumps(pick)
                    while True:
                        try:
                            mm.write(f'{dumped}\n'.encode('utf-8'))
                            break
                        except ValueError:
                            mm.resize(int(mm.size() * 1.2))
            random.shuffle(offsets)
            split_point = len(offsets) * 4 // 5
            training_pick_offsets = offsets[:split_point]
            validation_pick_offsets = offsets[split_point:]
            del offsets
            picks_to_pairs(training_pick_offsets, mm, Path('data/training_parsed_picks'))
            del training_pick_offsets
            picks_to_pairs(validation_pick_offsets, mm, Path('data/validation_parsed_picks'))
            del validation_pick_offsets
