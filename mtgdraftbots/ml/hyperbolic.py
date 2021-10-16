import tensorflow as tf

from mtgdraftbots.ml.utils import dropout


# @tf.function
def clipped_sinh(x, clipped_mag=8, name=None):
    with tf.name_scope(name or 'ClippedSinh') as scope:
        clip_op = tf.grad_pass_through(lambda y: tf.clip_by_value(y, tf.constant(-clipped_mag, dtype=y.dtype),
                                                                  tf.constant(clipped_mag, dtype=y.dtype),
                                                                  name='clipped_x'))
        return tf.math.sinh(clip_op(x), name=scope)


# @tf.function
def clipped_cosh(x, clipped_mag=8, name=None):
    with tf.name_scope(name or 'ClippedCosh') as scope:
        clip_op = tf.grad_pass_through(lambda y: tf.clip_by_value(y, tf.constant(-clipped_mag, dtype=y.dtype),
                                                                  tf.constant(clipped_mag, dtype=y.dtype),
                                                                  name='clipped_x'))
        return tf.math.cosh(clip_op(x), name=scope)


def safe_acosh(x, name=None):
    with tf.name_scope(name or 'SafeACosh') as scope:
        clip_op = tf.grad_pass_through(lambda y: tf.maximum(y, tf.constant(1.0001, dtype=y.dtype), name='clipped_x'))
        return tf.math.acosh(clip_op(x), name=scope)


# @tf.function
def inner_product_hyper(x, y, keepdims=False, name='HyperbolicInnerProduct'):
    with tf.name_scope(name) as scope:
        prod_xy = tf.math.multiply(x, y, name='prod_xy')
        summed = tf.math.reduce_sum(prod_xy[..., 1:], axis=-1, name='summed')
        result = tf.math.subtract(summed, prod_xy[..., 0], name=scope)
        if keepdims:
            return tf.expand_dims(result, -1)
        else:
            return result


# @tf.function
def distance_hyper(x, y, curvature, keepdims=False, name='HyperbolicDistance'):
    with tf.name_scope(name) as scope:
        inner = inner_product_hyper(x, y, keepdims=keepdims, name='inner_product')
        return tf.math.multiply(tf.math.sqrt(curvature, name='sqrt_curvature'),
                                safe_acosh(tf.math.divide(tf.math.negative(inner, name='negative_inner'),
                                                          curvature, name='scaled_neg_inner'),
                                           name='acosh_scaled_neg_inner'),
                                name=scope)


# @tf.function
def norm_hyper(vec, keepdims=False, name='HyperbolicNorm'):
    with tf.name_scope(name) as scope:
        return tf.math.sqrt(inner_product_hyper(vec, vec, keepdims=keepdims, name='inner_product'),
                            name=scope)


# @tf.function
def to_hyper(x, curvature, name='ToHyperbolic'):
    with tf.name_scope(name) as scope:
        sqrt_curvature = tf.math.sqrt(curvature, name='sqrt_curvature')
        scaled_norm_x = tf.add(tf.math.divide(tf.norm(x, axis=-1, keepdims=True, name='norm_x'), sqrt_curvature, name='scaled_norm_x'),
                               tf.constant(1e-04, dtype=x.dtype), name='shifted_norm_x')
        first = tf.math.multiply(sqrt_curvature, clipped_cosh(scaled_norm_x, name='cosh_norm_x'), name='first_coord')
        rest_multiplier = tf.where(scaled_norm_x > 0,
                                   tf.math.divide_no_nan(clipped_sinh(scaled_norm_x, name='sinh_norm_x'),
                                                         scaled_norm_x, name='x_multiplier_non_degenerate'),
                                   tf.ones_like(scaled_norm_x), name='rest_multiplier')
        rest = tf.math.multiply(rest_multiplier, x, 'scaled_x')
        return tf.concat([first, rest], axis=-1, name=scope)


# @tf.function
def logmap(vec, origin, curvature, name='HyperbolicLogMap'):
    with tf.name_scope(name) as scope:
        inner_scaled = tf.math.divide(inner_product_hyper(vec, origin, keepdims=True, name='inner_product'),
                                      curvature, name='inner_product_scaled')
        distance = tf.math.multiply(tf.math.sqrt(curvature, name='sqrt_curvature'),
                                    safe_acosh(tf.math.negative(inner_scaled, name='negative_inner_scaled'), name='acosh_inner'),
                                    name='distance')
        unnormalized = tf.math.add(vec, tf.math.multiply(origin, inner_scaled, name='scaled_origin'), name='unnormalized')
        multiplier = tf.math.divide_no_nan(distance, norm_hyper(unnormalized, keepdims=True, name='norm_unnormalized'),
                                           name='multiplier')
        result = tf.math.multiply(multiplier, unnormalized, name='result')
        return tf.where(tf.math.reduce_all(vec == origin, axis=-1, keepdims=True, name='match_origin'),
                        tf.zeros_like(result), result, name=scope)


# @tf.function
def expmap(vec, origin, curvature, name='HyperbolicExpMap'):
    with tf.name_scope(name) as scope:
        scaled_norm_vec = tf.add(tf.math.divide(tf.norm(vec, axis=-1, keepdims=True, name='norm_vec'),
                                                tf.math.sqrt(curvature, name='sqrt_curvature'), name='scaled_norm_x'),
                                 tf.constant(1e-04, dtype=vec.dtype), name='shifted_norm_vec')
        origin_coeff = clipped_cosh(scaled_norm_vec, name='origin_coeff')
        scaled_origin = tf.math.multiply(origin_coeff, origin, name='scaled_origin')
        vec_coeff = tf.where(scaled_norm_vec > 0,
                             tf.math.divide_no_nan(clipped_sinh(scaled_norm_vec, name='sinh_norm_vec'),
                                                   scaled_norm_vec, name='vec_coeff'),
                             tf.ones_like(scaled_norm_vec), name='vec_multiplier')
        scaled_vec = tf.math.multiply(vec_coeff, vec, name='scaled_vec')
        return tf.math.add(scaled_origin, scaled_vec, name=scope)


# @tf.function
def origin_hyper(curvature, dims, name='HyperbolicOrigin'):
    with tf.name_scope(name) as scope:
        return tf.concat([tf.reshape(tf.math.sqrt(curvature, name='sqrt_curvature'), (1,)),
                          tf.constant(0, dtype=curvature.dtype, shape=(dims - 1,))],
                         0, name=scope)


class DenseHyperbolic(tf.keras.layers.Layer):
    def __init__(self, output_dims, dropout_rate=0.0, use_bias=True, activation='linear', **kwargs):
        super(DenseHyperbolic, self).__init__(**kwargs)
        self.output_dims = output_dims
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation = activation

    def get_config(self):
        config = super(DenseHyperbolic, self).__init__(**kwargs)
        config.update({
            "output_dims": self.output_dims,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "activation": self.activation
        })
        return config

    def build(self, input_shape):
        self.input_dims = input_shape[0][-1]
        self.euclidean_dense = self.add_weight('euclidean_dense', shape=(self.input_dims, self.output_dims),
                                               initializer='random_normal', trainable=True)
        if self.use_bias:
            self.euclidean_bias = self.add_weight('euclidean_bias', shape=(self.output_dims - 1,),
                                                  initializer=tf.random_normal_initializer(0, 0.01), trainable=True)
        self.activation_layer = tf.keras.layers.Activation(self.activation)

    def call(self, inputs, training=True):
        vectors, in_curvature, out_curvature = inputs
        in_origin = origin_hyper(in_curvature, self.input_dims, name='in_origin')
        out_origin = origin_hyper(in_curvature, self.output_dims, name='out_origin')
        out_origin2 = origin_hyper(out_curvature, self.output_dims, name='out_origin2')
        tangent_vecs = logmap(vectors, origin=in_origin, curvature=in_curvature, name='tangent_vecs')
        weighted_vecs = tf.linalg.matmul(tangent_vecs, self.euclidean_dense, name='weighted_vecs')
        if self.use_bias:
            hyper_weighted_vecs = expmap(weighted_vecs, origin=out_origin, curvature=in_curvature, name='hyper_weighted_vecs')
            # bias is in the tangent space of out_origin
            bias = tf.concat([tf.constant(0, dtype=self.compute_dtype, shape=(1,)), self.euclidean_bias], 0, name='bias')
            # weighted vecs is the logmap of hyper_weighted_vecs with origin out_origin
            inner_num = inner_product_hyper(weighted_vecs, tf.expand_dims(bias, 0), keepdims=True, name='inner_weighted_vecs_bias')
            distance = distance_hyper(out_origin, hyper_weighted_vecs, in_curvature, keepdims=True, name='distance_hyper_weighted_vecs')
            summed_logs = tf.math.add(logmap(out_origin, origin=hyper_weighted_vecs, curvature=in_curvature, name='log_hyper_weighted_vecs'),
                                      weighted_vecs) # weighted vecs is the logmap of hyper_weighted_vecs with origin as out_origin
            summed_logs_mult = tf.math.divide(inner_num, tf.math.multiply(distance, distance, name='distance_hyper_weighted_vecs_squared'),
                                              name='summed_logs_mult')
            bias_tangent = tf.subtract(bias, tf.math.multiply(summed_logs, summed_logs_mult, name='summed_logs_scaled'), name='bias_tangent')
            # bias_tangent is in the tangent space of hyper_weighted_vecs so we need to bring it into hyperbolic space.
            biased = expmap(bias_tangent, origin=hyper_weighted_vecs, curvature=in_curvature, name='biased')
            # now we can lower it into the tangent space of out_origin.
            biased_tangent = logmap(biased, origin=out_origin, curvature=in_curvature, name='biased_tangent')
            removing_nans = tf.where(tf.math.reduce_all(hyper_weighted_vecs == out_origin, axis=-1, keepdims=True,
                                                        name='hyper_weighted_vecs_match_origin'),
                                     tf.expand_dims(bias, 0), biased_tangent, name='biased_tangent_no_nan')
            return expmap(dropout(self.activation_layer(removing_nans), self.dropout_rate, scale=2, training=training, name='dropped_out_result'),
                          origin=out_origin2, curvature=out_curvature, name=self.name)
        else:
            return expmap(dropout(self.activation_layer(weighted_vecs), self.dropout_rate, scale=2, training=training, name='dropped_out_result'),
                          origin=out_origin2, curvature=out_curvature, name=self.name)


class SetHyperEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_items, embed_dims, final_dims=None, item_dropout_rate=0.0, dense_dropout_rate=0.0, **kwargs):
        super(SetHyperEmbedding, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.item_dropout_rate = item_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.final_dims = final_dims or embed_dims

    def get_config(self):
        config = super(SetHyperEmbedding, self).get_config()
        config.update({
            "num_items": self.num_items,
            "embed_dims": self.embed_dims,
            "item_dropout_rate": self.item_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "final_dims": self.final_dims,
        })
        return config

    def build(self, input_shape):
        self.embeddings = self.add_weight('item_embeddings', shape=(self.num_items - 1, self.embed_dims - 1),
                                          initializer=tf.random_normal_initializer(0, 1 / self.embed_dims / self.embed_dims),
                                          trainable=True)
        self.upcast_2x = DenseHyperbolic(2 * self.embed_dims, activation='relu', use_bias=False, dropout_rate=self.dense_dropout_rate, name='upcast_2x')
        self.upcast_4x = DenseHyperbolic(4 * self.embed_dims, activation='relu', use_bias=False, dropout_rate=self.dense_dropout_rate, name='upcast_4x')
        self.downcast_final = DenseHyperbolic(self.final_dims, activation='linear', use_bias=False, dropout_rate=self.dense_dropout_rate, name='downcast_final')
        self.initial_bias = self.add_weight('initial_bias', shape=(self.embed_dims - 1,),
                                            initializer=tf.random_normal_initializer(0, 1 / self.embed_dims / self.embed_dims),
                                            trainable=True)

    def call(self, inputs, training=False, mask=None):
        indices, curvature = inputs
        embeddings = tf.concat([tf.zeros((1, self.embed_dims - 1), dtype=self.compute_dtype), self.embeddings], 0, name='embeddings')
        dropped_indices = dropout(indices, self.item_dropout_rate, training=training, name='dropped_indices')
        item_embeds = tf.gather(embeddings, dropped_indices, name='item_embeds')
        summed_embeds = tf.math.reduce_sum(item_embeds, 1, name='summed_embeds')
        normalized_embeds = tf.math.l2_normalize(tf.math.add(summed_embeds, self.initial_bias, name='biased_embeds'),
                                                 axis=-1, epsilon=1e-04, name='normalized_embeds')
        hyper_embeds = to_hyper(normalized_embeds, curvature, name='hyper_embeds')
        upcast_2x = self.upcast_2x((hyper_embeds, curvature, curvature), training=training)
        upcast_4x = self.upcast_4x((upcast_2x, curvature, curvature), training=training)
        return self.downcast_final((upcast_4x, curvature, curvature), training=training)
