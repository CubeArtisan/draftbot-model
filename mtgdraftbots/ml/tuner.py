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
import optuna
import tensorflow as tf
import tensorflow_addons as tfa
import zstandard as zstd
from optuna.integration import TFKerasPruningCallback
from tensorboard.plugins.hparams import api as hp

from mtgdraftbots.ml.draftbots import DraftBot
from mtgdraftbots.ml.generators import PickGenerator, PickPairGenerator
from mtgdraftbots.ml.tqdm_callback import TQDMProgressBar
from mtgdraftbots.ml.utils import Range, TensorBoardFix

directory = Path(sys.argv[1])
base_log_dir = Path(sys.argv[2])
seed = int(sys.argv[3])
CMA = len(sys.argv) > 4
print('Loading generators.')
epochs_per_cycle = 1
min_epochs = epochs_per_cycle
reduction_factor = 2
brackets = 4
max_epochs = min_epochs * reduction_factor ** (brackets - 1)
fixed_params = {
    'hyperbolic': 0,
    'batch_size': 512,
    # 'pool_dims': 512,
    # 'embed_dims': 128,
    # 'pool_hidden_units': 128,
    # 'margin': 1,
    # 'activation': 'elu',
    # 'final_activation': 'tanh',
    # 'adam_learning_rate': 2e-03,
    # 'optimizer': 'adam',
    # 'seen_dims': 512,
    # 'seen_hidden_units': 128,
    # 'pool_dropout_dense': 0.5,
    # 'seen_dropout_dense': 0.5,
    # 'seen_context_ratings': 0,
    # 'normalize_sum': 0,
    # 'item_ratings': 0,
    # 'dropout_pool': 0,
    # 'log_loss_weight': 0,
    # 'bounded_distance': 0,
}

pick_generator_train = PickGenerator(1, directory/'training_parsed_picks', epochs_per_cycle, seed)
print(f"There are {len(pick_generator_train):,} training picks.")
pick_generator_test = PickGenerator(8192, directory/'validation_parsed_picks', 1, seed)
print(f"There are {len(pick_generator_test):,} validation batches.")
with open(directory/'int_to_card.json', 'r') as cards_file:
    cards_json = json.load(cards_file)

tf.config.experimental.enable_op_determinism()
tf.config.optimizer.set_jit(False)
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


def get_hparams(trial):
    real_distributions = (
        optuna.distributions.UniformDistribution,
        optuna.distributions.LogUniformDistribution,
        optuna.distributions.DiscreteUniformDistribution,
    )
    int_distributions = (
        optuna.distributions.IntUniformDistribution,
        optuna.distributions.IntLogUniformDistribution,
    )
    categorical_distributions = (optuna.distributions.CategoricalDistribution,)
    supported_distributions = (
        real_distributions + int_distributions + categorical_distributions
    )

    hparams = {}

    for param_name, param_distribution in trial.distributions.items():
        if isinstance(param_distribution, real_distributions):
            hparams[param_name] = hp.HParam(
                param_name,
                hp.RealInterval(float(param_distribution.low), float(param_distribution.high)),
            )
        elif isinstance(param_distribution, int_distributions):
            hparams[param_name] = hp.HParam(
                param_name,
                hp.IntInterval(param_distribution.low, param_distribution.high),
            )
        elif isinstance(param_distribution, categorical_distributions):
            hparams[param_name] = hp.HParam(
                param_name,
                hp.Discrete(param_distribution.choices),
            )
        else:
            distribution_list = [
                distribution.__name__ for distribution in supported_distributions
            ]
            raise NotImplementedError(
                "The distribution {} is not implemented. "
                "The parameter distribution should be one of the {}".format(
                    param_distribution, distribution_list
                )
            )
    results = {}
    for param_name, param_value in trial.params.items():
        if param_name not in fixed_params:
            results[hparams[param_name]] = param_value
    return results


def create_model(trial):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()
    batch_size = trial.suggest_int('batch_size', 32, 2048, step=8)
    optimizer = 'adam' if CMA else trial.suggest_categorical('optimizer', ('adam', 'adamax', 'adadelta', 'nadam', 'sgd', 'lazyadam', 'rectadam', 'novograd'))
    log_loss_weight = trial.suggest_discrete_uniform('log_loss_weight', 0, 1, 0.01)
    seen_context_ratings = trial.suggest_int('seen_context_ratings', 0, 1)
    hparams = {
        'activation': 'tanh' if CMA else trial.suggest_categorical('activation', ('elu', 'selu', 'relu', 'tanh', 'sigmoid', 'linear', 'swish')),
        'normalize_sum': trial.suggest_int('normalize_sum', 0, 1),
        'triplet_loss_weight': 1.0 - log_loss_weight,
        'log_loss_weight': log_loss_weight,
        'embed_dims': trial.suggest_int('embed_dims', 8, 256, step=8),
        'pool_context_ratings': True,
        'pool_dims': trial.suggest_int('pool_dims', 8, 512, step=8),
        'pool_hidden_units': trial.suggest_int('pool_hidden_units', 8, 512, step=8),
        'seen_context_ratings': seen_context_ratings,
        'dropout_pool': trial.suggest_discrete_uniform('dropout_pool', 0, 0.99, 0.01),
        'pool_dropout_dense': trial.suggest_discrete_uniform('pool_dropout_dense', 0, 0.99, 0.01),
        'seen_dims': trial.suggest_int('seen_dims', 8, 256, step=8) if CMA or seen_context_ratings else 1,
        'seen_hidden_units': trial.suggest_int('seen_hidden_units', 8, 512, step=8) if CMA or seen_context_ratings else 1,
        'dropout_seen': trial.suggest_discrete_uniform('dropout_seen', 0, 0.99, 0.01) if CMA or seen_context_ratings else 1,
        'seen_dropout_dense': trial.suggest_discrete_uniform('seen_dropout_dense', 0, 0.99, 0.01) if CMA or seen_context_ratings else 0,
        'margin': trial.suggest_discrete_uniform('margin', 0, 10, 0.1),
        'item_ratings': trial.suggest_int('item_ratings', 0, 1),
        'hyperbolic': trial.suggest_int('hyperbolic', 0, 1),
        'bounded_distance': trial.suggest_int('bounded_distance', 0, 1),
        'final_activation': 'linear' if CMA else trial.suggest_categorical('final_activation', ('tanh', 'linear')),
    }
    draftbots = DraftBot(num_items=len(cards_json) + 1, **hparams, name='DraftBot')
    learning_rate = trial.suggest_loguniform('learning_rate' if CMA else f"{optimizer}_learning_rate", 1e-04, 1e-02)
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer == 'adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    if optimizer == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    if optimizer == 'nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    if optimizer == 'sgd':
        momentum = 0 if CMA else trial.suggest_loguniform('sgd_momentum', 1e-05, 1e-01)
        nesterov = 0 if CMA else trial.suggest_int('sgd_nesterov', 0, 1)
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    if optimizer == 'lazyadam':
        opt = tfa.optimizers.LazyAdam(learning_rate=learning_rate)
    if optimizer == 'rectadam':
        weight_decay = 0 if CMA else trial.suggest_loguniform('rectadam_weight_decay', 1e-08, 1e-01)
        opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, weight_decay=weight_decay)
    if optimizer == 'novograd':
        weight_decay = 0 if CMA else trial.suggest_loguniform('novograd_weight_decay', 1e-08, 1e-01)
        opt = tfa.optimizers.NovoGrad(learning_rate=learning_rate, weight_decay=weight_decay)
    tf.keras.utils.set_random_seed(seed)
    pick_generator_test.reset_rng()
    pick_generator_train.reset_rng()
    pick_generator_train.batch_size = batch_size
    draftbots.compile(optimizer=opt, loss=lambda y_true, y_pred: 0.0)
    return draftbots


def objective(trial):
    global trial_num
    model = create_model(trial)
    log_dir = base_log_dir/f'run-{trial.number:04d}'
    callbacks = []
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs_per_cycle,
                                                   min_delta=2e-04, mode='min',
                                                   restore_best_weights=False, verbose=True)
    tb_callback = TensorBoardFix(log_dir=str(log_dir), histogram_freq=0, write_graph=True,
                                 update_freq=len(pick_generator_train) // 4, embeddings_freq=0,
                                 profile_batch=0)
    hp_callback = hp.KerasCallback(str(log_dir), get_hparams(trial))
    pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy_top_1')
    time_stopping = tfa.callbacks.TimeStopping(seconds=3600, verbose=1)
    mcp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= str(log_dir/'model'),
        monitor='val_accuracy_top_1',
        verbose=False,
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        save_freq='epoch')
    callbacks.append(nan_callback)
    callbacks.append(es_callback)
    callbacks.append(tb_callback)
    callbacks.append(hp_callback)
    callbacks.append(pruning_callback)
    callbacks.append(time_stopping)
    callbacks.append(mcp_callback)
    history = model.fit(
        pick_generator_train,
        validation_data=pick_generator_test,
        epochs=max_epochs,
        callbacks=callbacks,
        max_queue_size=2**10,
    )
    return max(*history.history['val_accuracy_top_1'])


def show_result(study):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    n_startup_trials = 20 - len(fixed_params)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed, consider_endpoints=True,
                                         consider_magic_clip=True, n_startup_trials=20)
    if CMA:
        sampler = optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=16,
                                               independent_sampler=sampler, restart_strategy='ipop',
                                               consider_pruned_trials=True)
    sampler = optuna.samplers.PartialFixedSampler(fixed_params, sampler)
    pruner = optuna.pruners.HyperbandPruner(min_resource=min_epochs, max_resource=max_epochs,
                                            reduction_factor=reduction_factor)
    study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler,
                                study_name='MtgDraftBots')
    study.optimize(objective, n_trials=320)
