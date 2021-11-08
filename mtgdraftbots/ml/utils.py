import tensorflow as tf

# @tf.function
def dropout(vecs, rate, scale=None, training=False, name=None):
    if 0 >= rate or not training:
        return vecs
    elif rate >= 1:
        return tf.zeros_like(vecs, name=name or f'Dropout{rate}')
    else:
        with tf.name_scope(name or f'Dropout{rate}') as scope:
            noise = tf.random.uniform(tf.shape(vecs), minval=0, maxval=1, dtype=tf.float32, seed=67, name='noise')
            noise_mult = tf.where(noise >= rate, tf.ones_like(vecs), tf.zeros_like(vecs), name='noise_mult')
            if scale:
                dropped_vecs = tf.multiply(vecs, noise_mult, name='dropped_vecs')
                if scale == 'sum':
                    return tf.math.multiply(tf.math.divide_no_nan(tf.reduce_sum(vecs, name='sum_pre_drop'),
                                                                  tf.reduce_sum(dropped_vecs, name='sum_post_drop') + 1e-04,
                                                                  name='scaling_factor'),
                                            dropped_vecs, name=scope)
                else:
                    return tf.math.multiply(tf.math.divide_no_nan(tf.norm(vecs, axis=-1, keepdims=True, ord=scale, name='norm_pre_drop'),
                                                                  tf.norm(dropped_vecs, axis=-1, ord=scale, keepdims=True, name='norm_post_drop') + 1e-04,
                                                                  name='scaling_factor'),
                                            dropped_vecs, name=scope)
            else:
                return tf.math.multiply(vecs, noise_mult, name=scope)


class TensorBoardFix(tf.keras.callbacks.TensorBoard):
    """
    This fixes incorrect step values when using the TensorBoard callback with custom summary ops
    """
    def on_train_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)

    def on_test_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_test_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._val_step)

TensorBoardFix.__name__ = 'TensorBoard'


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __str__(self):
        return f'values in the inclusive range [{self.start}, {self.end}]'

    def __repr__(self):
        return str(self)
