import logging

import numpy as np
import tensorflow as tf


class DynamicAdjustmentCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience_degrading=16, cooldown_degrading=0, patience_plateau=32,
                 cooldown_plateau=0, verbose=0, mode='min', min_delta=1e-04):
        super(DynamicAdjustmentCallback, self).__init__()
        self.monitor = monitor
        self.patience_degrading = patience_degrading
        self.cooldown_degrading = cooldown_degrading
        self.patience_plateau = patience_plateau
        self.cooldown_plateau = cooldown_plateau
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown_counter = 0
        self.wait = 0
        self.monitor_improving = None
        self.monitor_degrading = None
        self._reset()

    def _reset(self):
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        # Default to improving when monitor goes up by more than min_delta.
        self.monitor_improving = lambda a, b: np.greater(a, b + self.min_delta)
        # Degrading when monitor goes down by more than min_delta.
        self.monitor_degrading = lambda a, b: np.less(a, b - self.min_delta)
        self.best = -np.Inf
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_improving, self.monitor_degrading = self.monitor_degrading, self.monitor_improving
            self.best = np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        if current is None:
            logging.warning(f'Learning rate reduction is conditioned on metric `{self.monitor}` '
                            f'which is not available. Available metrics are: {",".join(list(logs.keys()))}')
            return
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        if self.monitor_improving(current, self.best):
            if self.verbose > 1:
                logging.info(f'{self.monitor} improved from {self.best} to {current}.')
            self.best = current
            self.wait = 0
            self._on_improving(epoch, logs, current)
        else:
            self.wait += 1
            if self.monitor_degrading(current, self.best):
                if self.verbose > 1:
                    logging.info(f'{self.monitor} degraded from {self.best} to {current}.')
                if self.wait >= self.patience_degrading:
                    self._on_degrading(epoch, logs, current)
                    self.cooldown_counter = self.cooldown_degrading
                    self.wait = 0
            else:
                if self.verbose > 1:
                    logging.info(f'{self.monitor} plateaued, going from {self.best} to {current}.')
                if self.wait >= self.patience_plateau:
                    self._on_plateau(epoch, logs, current)
                    self.cooldown_counter = self.cooldown_plateau
                    self.wait = 0

    def _on_improving(self, epoch, logs, monitor):
        pass

    def _on_degrading(self, epoch, logs, monitor):
        pass

    def _on_plateau(self, epoch, logs, monitor):
        pass


class DynamicLearningRateCallback(DynamicAdjustmentCallback):
    def __init__(self, shrink_factor=0.5, grow_factor=None, min_lr=1e-07, max_lr=1e-01, **kwargs):
        super(DynamicLearningRateCallback, self).__init__(**kwargs)
        self.shrink_factor = shrink_factor
        self.grow_factor = grow_factor if grow_factor is not None else (1 / shrink_factor)
        self.min_lr = np.float32(min_lr)
        self.max_lr = np.float32(max_lr)
        self.last_action = None
        self.action_count = 0

    def grow_learning_rate(self, epoch, exp=1):
        old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        if old_lr < self.max_lr:
            new_lr = min(old_lr * self.grow_factor ** exp, self.max_lr)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                logging.info(f'\nEpoch {epoch + 1:05d}: increasing learning rate to {new_lr}.')
            if self.last_action == 'grow':
                self.action_count += exp
            else:
                self.last_action = 'grow'
                self.action_count = exp

    def shrink_learning_rate(self, epoch, exp=1):
        old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        if old_lr > self.min_lr:
            new_lr = max(old_lr * self.shrink_factor ** exp, self.min_lr)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                logging.info(f'\nEpoch {epoch + 1:05d}: decreasing learning rate to {new_lr}.')
            if self.last_action == 'shrink':
                self.action_count += exp
            else:
                self.last_action = 'shrink'
                self.action_count = exp

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        super(DynamicLearningRateCallback, self).on_epoch_end(epoch, logs)

    def _on_improving(self, epoch, logs, monitor):
        self.last_action = None
        self.action_count = 0
        self.grow_learning_rate(epoch)

    def _on_degrading(self, epoch, logs, monitor):
        if self.last_action == 'shrink':
            self.grow_learning_rate(epoch, exp=self.action_count + 2)
        elif self.last_action == 'grow':
            self.shrink_learning_rate(epoch, exp=self.action_count + 2)
        else:
            self.shrink_learning_rate(epoch, exp=2)
        self.best = monitor

    def _on_plateau(self, epoch, logs, monitor):
        self.shrink_learning_rate(epoch)