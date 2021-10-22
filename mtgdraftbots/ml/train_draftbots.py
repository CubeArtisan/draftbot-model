import argparse
import datetime
import io
import json
import locale
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import zstandard as zstd
from tensorboard.plugins.hparams import api as hp

from mtgdraftbots.ml.draftbots import DraftBot
from mtgdraftbots.ml.generators import PickGenerator, PickPairGenerator
from mtgdraftbots.ml.tqdm_callback import TQDMProgressBar
from mtgdraftbots.ml.utils import Range, TensorBoardFix

if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, '')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    BATCH_CHOICES = tuple(2 ** i for i in range(4, 18))
    EMBED_DIMS_CHOICES = tuple(2 ** i for i in range(1, 10))
    ACTIVATION_CHOICES = ('relu', 'selu', 'swish', 'tanh', 'sigmoid', 'linear', 'gelu', 'elu')
    HYPER_PARAMS = (
        {"name": "batch_size", "type": int, "choices": BATCH_CHOICES, "range": hp.Discrete(BATCH_CHOICES),
         "default": 8192, "help": "The batch size for one step."},
        {"name": "learning_rate", "type": float, "choices": [Range(1e-06, 1e+01)],
         "range": hp.RealInterval(1e-06, 1e+05), "default": 1e-03, "help": "The initial learning rate to train with."},
        {"name": "embed_dims", "type": int, "default": 16, "choices": EMBED_DIMS_CHOICES,
         "range": hp.Discrete(EMBED_DIMS_CHOICES), "help": "The number of dimensions to use for card embeddings."},
        {"name": "seen_dims", "type": int, "default": 16, "choices": EMBED_DIMS_CHOICES,
         "range": hp.Discrete(EMBED_DIMS_CHOICES), "help": 'The number of dimensions to use for seen card embeddings.'},
        {"name": 'picked_dims', "type": int, "default": 16, "choices": EMBED_DIMS_CHOICES,
         "range": hp.Discrete(EMBED_DIMS_CHOICES), "help": 'The number of dimensions to use for picked card embeddings.'},
        {"name": 'dropout_picked', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The percent of cards to drop from picked when calculating the pool embedding.'},
        {"name": 'dropout_seen', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The percent of cards to drop from picked when calculating the seen embedding.'},
        {"name": 'dropout_dense', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The percent of values to drop from the dense layers when calculating pool/seen embeddings.'},
        {"name": 'contrastive_loss_weight', "type": float, "default": 1.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The relative weight of the loss based on difference of the scores.'},
        {"name": 'log_loss_weight', "type": float, "default": 1.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The relative weight of the loss based on the log of the probability we guess correctly for each pair.'},
        {"name": 'rating_uniformity_weight', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The weight of the loss to make card ratings uniformly distributed.'},
        {"name": 'picked_synergy_uniformity_weight', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The weight of the loss to make picked synergies uniformly distributed.'},
        {"name": 'seen_synergy_uniformity_weight', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The weight of the loss to make seen synergies uniformly distributed.'},
        {"name": 'margin', "type": float, "default": 1.0, "choices": [Range(0.0, 1e+02)],
         "range": hp.RealInterval(0.0, 1e+02), "help": 'The minimum amount the score of the correct option should win by.'},
        {"name": 'picked_variance_weight', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The weight given to making the variance of picked contextual rating close to that of a uniform distribution.'},
        {"name": 'seen_variance_weight', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The weight given to making the variance of seen contextual rating close to that of a uniform distribution.'},
        {"name": 'picked_distance_l2_weight', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The weight given to the L2 loss on the picked contextual rating distances.'},
        {"name": 'seen_distance_l2_weight', "type": float, "default": 0.0, "choices": [Range(0.0, 1.0)],
         "range": hp.RealInterval(0.0, 1.0), "help": 'The weight given to the L2 loss on the seen contextual rating distances.'},
        {"name": 'activation', "type": str, "default": 'elu', "choices": ACTIVATION_CHOICES,
         "range": hp.Discrete(ACTIVATION_CHOICES), "help": "The activation function for the hidden layers."},
    )

    parser.add_argument('--epochs', '-e', type=int, required=True, help="The maximum number of epochs to train for")
    parser.add_argument('--name', '-o', '-n', type=str, required=True, help="The name to save this model under.")
    parser.add_argument('--seed', type=int, default=37, help='The random seed to initialize things with to improve reproducibility.')
    for param in HYPER_PARAMS:
        parser.add_argument(f'--{param["name"]}', type=param["type"], default=param["default"],
                            choices=param["choices"], help=param["help"])
    parser.add_argument('--hyperbolic', action='store_true', help='Use the hyperbolic geometry model.')
    float_type_group = parser.add_mutually_exclusive_group()
    float_type_group.add_argument('-16', dest='float_type', const=tf.float16, action='store_const', help='Use 16 bit numbers throughout the model.')
    float_type_group.add_argument('--auto16', '-16rw', action='store_true', help='Automatically rewrite some operations to use 16 bit numbers.')
    float_type_group.add_argument('--keras16', '-16k', action='store_true', help='Have Keras automatically convert the synergy calculations to 16 bit.')
    float_type_group.add_argument('-32', dest='float_type', const=tf.float32, action='store_const', help='Use 32 bit numbers (the default) throughout the model.')
    float_type_group.add_argument('-64', dest='float_type', const=tf.float64, action='store_const', help='Use 64 bit numbers throughout the model.')
    xla_group = parser.add_mutually_exclusive_group()
    xla_group.add_argument('--xla', action='store_true', dest='use_xla', help='Enable using xla to optimize the model (the default).')
    xla_group.add_argument('--no-xla', action='store_false', dest='use_xla', help='Disable using xla to optimize the model.')
    parser.add_argument('--debug', action='store_true', help='Enable debug dumping of tensor stats.')
    parser.add_argument('--mlir', action='store_true', help='Enable MLIR passes on the data (EXPERIMENTAL).')
    parser.add_argument('--profile', action='store_true', help='Enable profiling a range of batches from the first epoch.')
    parser.add_argument('--deterministic', action='store_true', help='Try to keep the run deterministic so results can be reproduced.')
    parser.add_argument('--dir', type=str, required=True, help='The soure directory where the training and validation data are stored')
    parser.set_defaults(float_type=tf.float32, use_xla=True)
    args = parser.parse_args()
    hparams = {hp.HParam(param["name"], param["range"]): getattr(args, param["name"]) for param in HYPER_PARAMS}
    hparams[hp.HParam('hyperbolic', hp.Discrete((True, False)))] =  args.hyperbolic
    tf.keras.utils.set_random_seed(args.seed)
    directory = Path(args.dir)

    logging.info('Loading card data for seeding weights.')
    with open(directory/'int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
        card_ratings = [-1] + [(c.get('elo', 1200) / 1200) - 1  for c in cards_json]
        blank_embedding = [1 for _ in range(64)]
        card_names = [''] + [c['name'] for c in cards_json]

    logging.info('Creating the pick Datasets.')
    train_epochs_per_cycle = 1
    pick_generator_train = PickPairGenerator(args.batch_size, Path(directory/'training_parsed_picks'),
                                             train_epochs_per_cycle, args.seed)
    logging.info(f"There are {len(pick_generator_train):,} training batches.")
    pick_generator_test = PickGenerator(args.batch_size // 4, Path(directory/'validation_parsed_picks'),
                                        1, args.seed)
    logging.info(f"There are {len(pick_generator_test):n} validation batches.")
    logging.info(f"There are {len(cards_json):n} cards being trained on.")
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.debug:
        log_dir = "logs/debug/"
        logging.info('Enabling Debugging')
        tf.debugging.experimental.enable_dump_debug_info(
            log_dir,
            tensor_debug_mode='FULL_HEALTH',
            circular_buffer_size=-1,
            tensor_dtypes=[args.float_type],
            # op_regex="(?!^(Placeholder|Constant)$)"
        )
    if args.mlir:
        tf.config.experimental.enable_mlir_graph_optimization()
        tf.config.experimental.enable_mlir_bridge()
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    Path(log_dir).mkdir(exist_ok=True, parents=True)

    if args.keras16 or args.float_type == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    tf.config.optimizer.set_jit(args.use_xla)
    if args.debug:
        tf.config.optimizer.set_experimental_options=({
            'layout_optimizer': True,
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': False,
            'disable_model_pruning': True,
            'scoped_allocator_optimization': True,
            'pin_to_host_optimization': True,
            'implementation_selector': True,
            'disable_meta_optimizer': True,
            'min_graph_nodes': 1,
        })
    else:
        tf.config.optimizer.set_experimental_options=({
            'layout_optimizer': True,
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
            'disable_model_pruning': False,
            'scoped_allocator_optimization': True,
            'pin_to_host_optimization': True,
            'implementation_selector': True,
            'disable_meta_optimizer': False,
            'min_graph_nodes': 1,
        })
    tf.config.threading.set_intra_op_parallelism_threads(32)
    tf.config.threading.set_inter_op_parallelism_threads(32)

    metadata = os.path.join(log_dir, 'metadata.tsv')
    with open(metadata, "w") as f:
        f.write('Index\tName\tColors\tMana Value\tType\n')
        f.write('0\t"PlaceholderForTraining"\t1278\t1287\t1827\n')
        for i, card in enumerate(cards_json):
            f.write(f'{i+1}\t"{card["name"]}"\t{"".join(sorted(card.get("color_identity")))}\t{card["cmc"]}\t{card.get("type")}\n')

    logging.info('Loading DraftBot model.')
    output_dir = f'././ml_files/{args.name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_batches = len(pick_generator_train)
    tensorboard_period = num_batches // 20
    draftbots_kwargs = {param["name"]: getattr(args, param["name"]) for param in HYPER_PARAMS}
    del draftbots_kwargs["batch_size"]
    del draftbots_kwargs["learning_rate"]
    draftbots_kwargs["hyperbolic"] = args.hyperbolic
    draftbots = DraftBot(num_items=len(cards_json) + 1, summary_period=tensorboard_period * 4,
                         name='DraftBot', **draftbots_kwargs)
    latest = tf.train.latest_checkpoint(output_dir)
    learning_rate = args.learning_rate or 0.001
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    # opt = tfa.optimizers.LazyAdam(learning_rate=learning_rate)
    # opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    # opt = tfa.optimizers.NovoGrad(learning_rate=learning_rate)
    # opt = tfa.optimizers.LAMB(learning_rate=learning_rate)
    # opt = tfa.optimizers.Lookahead(opt, sync_period=16, slow_step_size=0.5)
    if args.float_type == tf.float16:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic_growth_steps=num_batches // 128)
    if args.auto16:
        logging.warn("WARNING 16 bit rewrite mode can cause numerical instabilities.")
        tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(opt)
    if latest is not None:
        logging.info('Loading Checkpoint.')
        draftbots.load_weights(latest)
    draftbots.compile(optimizer=opt, loss=lambda y_true, y_pred: 0.0)

    logging.info('Starting training')
    callbacks = []
    if not args.debug:
        mcp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir + 'model',
            monitor='val_accuracy_top_1',
            verbose=False,
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            save_freq='epoch')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + '/model-{epoch:04d}.ckpt',
            monitor='val_accuracy_top_1',
            verbose=False,
            save_best_only=False,
            save_weights_only=True,
            mode='max',
            save_freq='epoch')
        callbacks.append(mcp_callback)
        callbacks.append(cp_callback)
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy_top_1', patience=8, min_delta=2**-8,
                                                   mode='max', restore_best_weights=True, verbose=True)
    tb_callback = TensorBoardFix(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                 update_freq=tensorboard_period, embeddings_freq=None,
                                 profile_batch=0 if args.debug or not args.profile else (num_batches // 2 - 16, num_batches // 2 + 15))
    hp_callback = hp.KerasCallback(log_dir, hparams)
    BAR_FORMAT = "{n_fmt}/{total_fmt}{bar} {elapsed}/{remaining}s - {rate_fmt} - {desc}"
    tqdm_callback = TQDMProgressBar(smoothing=0.01, epoch_bar_format=BAR_FORMAT, ascii=True)
    callbacks.append(nan_callback)
    # callbacks.append(es_callback)
    callbacks.append(tb_callback)
    callbacks.append(hp_callback)
    callbacks.append(tqdm_callback)
    draftbots.fit(
        pick_generator_train,
        validation_data=pick_generator_test,
        validation_freq=train_epochs_per_cycle,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
        max_queue_size=2**8,
    )
    if not args.debug:
        logging.info('Saving final model.')
        Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
        draftbots.save(f'{output_dir}/final', save_format='tf')
