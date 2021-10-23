import subprocess

fixed_args = [
    "--dir", "data/m19_drafts",
    "--deterministic",
    "--xla",
    "-32",
    "--epochs", '1',
    "--seed", '127',
    "--batch_size", '4096',
    "--learning_rate", '1e-03',
    "--dropout_picked", "0.25",
    "--dropout_seen", "0.25",
    "--dropout_dense", "0.0",
    "--contrastive_loss_weight", "1.0",
    "--rating_uniformity_weight", "0.0",
    "--picked_synergy_uniformity_weight", "0.0",
    "--seen_synergy_uniformity_weight", "0.0",
    "--picked_variance_weight", "0.0",
    "--seen_variance_weight", "0.0",
    "--picked_distance_l2_weight", "0.0",
    "--seen_distance_l2_weight", "0.0",
    "--optimizer", "adamax",
    "--epochs_per_cycle", "1",
]

def launch_command(args, counter):
    command = ['python', '-m' 'mtgdraftbots.ml.train_draftbots', '--name', f'run-{counter:03d}'] + args
    print(command)
    print(subprocess.run(' '.join(command), shell=True))

counter = 0
for hyperbolic in (True, False):
    args_0 = fixed_args + (['--hyperbolic'] if hyperbolic else [])
    for log_loss_weight in (0.0,):
        args_1 = args_0 + ['--log_loss_weight', str(log_loss_weight)]
        for margin in (1,):
            args_2 = args_1 + ['--margin', str(margin)]
            # for activation in ('relu', 'elu', 'tanh', 'linear'):
            for activation in ('elu',):
                args_3 = args_2 + ['--activation', activation]
                for use_pool in (True, False):
                    args_4 = args_3 + (['--pool_context_ratings'] if use_pool else [])
                    for use_seen in (True, False):
                        args_5 = args_4 + (['--seen_context_ratings'] if use_seen else [])
                        for use_ratings in (True, False):
                            if not use_ratings and not use_pool and not use_seen:
                                continue
                            args_6 = args_5 + (['--item_ratings'] if use_ratings else [])
                            for dims in (2, 64):
                                dims = dims + (1 if hyperbolic else 0)
                                args_7 = args_6 + ['--embed_dims', str(dims), '--seen_dims', str(dims), '--picked_dims', str(dims)]
                                launch_command(args_7, counter)
                                counter += 1
