import itertools
import random
import subprocess
import sys

FIXED = [
    "--dir", sys.argv[1],
    "--deterministic",
    "--xla",
    "-32",
    "--epochs", '5',
    "--epochs_per_cycle", "5",
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
    print(command)
    print('Result', subprocess.run(command, shell=True))


VALUE_HYPER_PARAMS = {
    'batch_size': (4096,),
    'learning_rate': ('1e-04',),
    'embed_dims': (256,),
    'dropout_picked': (0.0,),
    'dropout_seen': (0.0,),
    'picked_dropout_dense': (0.5,),
    'picked_hidden_units': (128,),
    'picked_dims': (512,),
    'seen_dropout_dense': (0.5,),
    'seen_hidden_units': (128,),
    'seen_dims': (512,),
    'triplet_loss_weight': (1.0,),
    'margin': (1,),
    'activation': ('tanh',),
    'final_activation': ('linear',),
    'optimizer': ('adam',),
    'log_loss_weight': (0.0, 1.0),
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
    args_arr = [arg_tuples for arg_tuples in itertools.product(*HYPER_PARAMS)]
    print(f'There are {len(args_arr):,} configurations')
    random.shuffle(args_arr)
    for counter, args in enumerate(args_arr):
        launch_command(itertools.chain(FIXED, *args), counter + 1)


if __name__ == '__main__':
    random.seed(int(sys.argv[2]))
    run_trials()
