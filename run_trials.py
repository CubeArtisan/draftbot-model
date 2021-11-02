import itertools
import random
import subprocess
import sys

FIXED = [
    "--dir", sys.argv[1],
    "--deterministic",
    "--xla",
    "-32",
    "--epochs", '16',
    "--epochs_per_cycle", "4",
    "--seed", '127',
    "--pool_context_ratings",
]


def value_arg(name, choices):
    for choice in choices:
        yield (f'--{name}', str(choice))


def bool_arg(name):
    return ((f'--{name}',), ())


def launch_command(args, counter):
    command = ' '.join(itertools.chain(('python', '-m', 'mtgdraftbots.ml.train_draftbots', '--name', f'run-{counter:03d}'), args))
    print('Running', command)
    print('Result', subprocess.run(command, shell=True))


VALUE_HYPER_PARAMS = {
    'batch_size': (128, 1024, 8192),
    'learning_rate': ('1e-04', '1e-03', '1e-02'),
    'embed_dims': (2, 64),
    'seen_dims': (2, 64),
    'picked_dims': (2, 64),
    'dropout_picked': (0.0, 0.5),
    'dropout_seen': (0.0, 0.5),
    'dropout_dense': (0.0, 0.5),
    'contrastive_loss_weight': (1.0,),
    'log_loss_weight': (0.0,),
    'rating_uniformity_weight': (0,),
    'picked_synergy_uniformity_weight': (0,),
    'seen_synergy_uniformity_weight': (0,),
    'margin': (0, 1),
    'picked_variance_weight': (0.0,),
    'seen_variance_weight': (0.0,),
    'picked_distance_l2_weight': (0.0,),
    'seen_distance_l2_weight': (0.0,),
    # 'picked_variance_weight': (0.0, '1e-01'),
    # 'seen_variance_weight': (0.0, '1e-01'),
    # 'picked_distance_l2_weight': (0.0, '1e-04'),
    # 'seen_distance_l2_weight': (0.0, '1e-04'),
    'activation': ('selu', 'tanh', 'linear'),
    'final_activation': ('tanh', 'linear'),
    'optimizer': ('adam', 'adamax'),
}
BOOL_HYPER_PARAMS = (
    'seen_context_ratings',
    'item_ratings',
    'hyperbolic',
    'bounded_distance',
    'normalize_sum',
)
HYPER_PARAMS = [value_arg(k, v) for k, v in VALUE_HYPER_PARAMS.items()] + [bool_arg(k) for k in BOOL_HYPER_PARAMS]

def run_trials():
    counter = 0
    args_arr = [arg_tuples for arg_tuples in itertools.product(*HYPER_PARAMS)]
    print(f'There are {len(args_arr):,} configurations')
    random.shuffle(args_arr)
    for args in args_arr:
        launch_command(itertools.chain(FIXED, *args), counter)
        counter += 1


if __name__ == '__main__':
    run_trials()
