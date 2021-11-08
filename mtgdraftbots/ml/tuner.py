import tensorflow as tf

import argparse
import datetime
import io
import json
import locale
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import zstandard as zstd
from keras_tuner import Hyperband, Objective, BayesianOptimization
from tensorboard.plugins.hparams import api as hp

from mtgdraftbots.ml.draftbots import DraftBot
from mtgdraftbots.ml.generators import PickGenerator, PickPairGenerator
from mtgdraftbots.ml.tqdm_callback import TQDMProgressBar
from mtgdraftbots.ml.utils import Range, TensorBoardFix

directory = Path(sys.argv[1])
seed = int(sys.argv[3])
tf.keras.utils.set_random_seed(seed)
print('Loading generators.')
pick_generator_train = PickGenerator(1, directory/'training_parsed_picks', 8, seed)
print(f"There are {len(pick_generator_train):,} training picks.")
pick_generator_test = PickGenerator(8192, directory/'validation_parsed_picks', 1, seed)
print(f"There are {len(pick_generator_test):,} validation batches.")
with open(directory/'int_to_card.json', 'r') as cards_file:
    cards_json = json.load(cards_file)

# tf.config.experimental.enable_op_determinism()
tf.config.optimizer.set_jit(True)
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

def model_builder(hp):
    batch_size = 2048
    learning_rate = hp.Float('learning_rate', 1e-04, 1e-02, sampling='log', default=1e-03)
    # optimizer = hp.Choice('optimizer', ('adam', 'adamax', 'lazyadam', 'rectadam', 'novograd', 'adadelta', 'nadam',))
    optimizer = 'adam'
    log_loss_weight = hp.Float('log_loss_weight', 0, 1, step=0.01, default=0.5)
    hparams = {
        'dropout_dense': hp.Float('dropout_dense', 0, 0.95, step=0.01, default=0.5),
        'dropout_picked': hp.Float('dropout_picked', 0, 0.9, step=0.01, default=0.5),
        'dropout_seen': hp.Float('dropout_seen', 0, 0.95, step=0.01, default=0.5),
        'activation': 'tanh',
        # 'activation': hp.Choice('activation', ('elu', 'selu', 'relu', 'tanh', 'sigmoid', 'linear', 'swish'), default='tanh'),
        'normalize_sum': hp.Boolean('normalize_sum', default=True),
        'contrastive_loss_weight': 1.0 - log_loss_weight,
        'log_loss_weight': log_loss_weight,
        'embed_dims': 128,
        'picked_dims': 128,
        'seen_dims': 256,
        'margin': 1,
        'pool_context_ratings': True,
        'seen_context_ratings': True,
        'rating_uniformity_weight': 0.0,
        'picked_synergy_uniformity_weight': 0.0,
        'seen_synergy_uniformity_weight': 0.0,
        'picked_variance_weight': 0.0,
        'seen_variance_weight': 0.0,
        'picked_distance_l2_weight': 0.0,
        'seen_distance_l2_weight': 0.0,
        'item_ratings': True,
        'hyperbolic': False,
        'bounded_distance': True,
        'final_activation': 'linear',
    }
    draftbots = DraftBot(num_items=len(cards_json) + 1, **hparams, name='DraftBot')
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    if optimizer == 'nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    if optimizer == 'adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    if optimizer == 'lazyadam':
        opt = tfa.optimizers.LazyAdam(learning_rate=learning_rate)
    if optimizer == 'rectadam':
        opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    if optimizer == 'novograd':
        opt = tfa.optimizers.NovoGrad(learning_rate=learning_rate)
    pick_generator_train.epochs_per_completion = int(math.ceil(len(pick_generator_train.picked) / 2048 / batch_size))
    draftbots.compile(optimizer=opt, loss=lambda y_true, y_pred: 0.0)
    pick_generator_test.reset_rng()
    pick_generator_train.reset_rng()
    pick_generator_train.batch_size = batch_size
    return draftbots


if __name__ == '__main__':
    # tuner = Hyperband(
    #     model_builder,
    #     objective=Objective("val_accuracy_top_1", direction="max"),
    #     max_epochs=16,
    #     factor=4,
    #     hyperband_iterations=20,
    #     seed=seed,
    #     project_name='MtgDraftBots',
    #     directory=sys.argv[2],
    #     overwrite=True,
    # )
    tuner = BayesianOptimization(
        model_builder,
        objective=Objective("val_accuracy_top_1", direction="max"),
        max_trials=18 * 20,
        seed=seed,
        num_initial_points=18,
        project_name='MtgDraftBots',
        directory=sys.argv[2],
        overwrite=True,
        alpha=1e-02,
        beta=5.0,
        # max_model_size=2**26,
    )
    callbacks = []
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy_top_1', patience=1, min_delta=1e-04,
                                                   mode='max', restore_best_weights=False, verbose=True)
    tb_callback = TensorBoardFix(log_dir=sys.argv[2], histogram_freq=1, write_graph=True,
                                 update_freq=1024, embeddings_freq=1,
                                 profile_batch=0)
    time_stopping = tfa.callbacks.TimeStopping(seconds=200, verbose=1)
    callbacks.append(nan_callback)
    callbacks.append(es_callback)
    callbacks.append(tb_callback)
    # Comment out if using Hyperband
    callbacks.append(time_stopping)
    tuner.search(pick_generator_train, validation_data=pick_generator_test, callbacks=callbacks,
                 epochs=32, validation_freq=1, max_queue_size=2**10)
    tuner.results_summary()
