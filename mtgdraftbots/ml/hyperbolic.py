import tensorflow as tf


@tf.function
def inner_product_hyper(x, y, dims, keepdims=False, name='HyperbolicInnerProduct'):
    with tf.name_scope(name) as scope:
        g_x = tf.constant([-1 if i == 0 else 1 for i in range(dims)], dtype=x.dtype, name=f'g_x_{dims}')
        return tf.math.reduce_sum(tf.math.multiply(g_x, tf.math.multiply(x, y, name='prod_xy'),
                                                   name='prod_xy_g_x'), axis=-1, keepdims=keepdims, name=scope)


@tf.function
def distance_hyper(x, y, curvature, dims, keepdims=False, name='HyperbolicDistance'):
    with tf.name_scope(name) as scope:
        inner = inner_product_hyper(x, y, dims, keepdims=keepdims, name='inner_product')
        return tf.math.multiply(tf.math.sqrt(curvature, name='sqrt_curvature'),
                                tf.math.acosh(tf.math.divide(tf.math.negative(inner, name='negative_inner'),
                                                             curvature, name='scaled_neg_inner'),
                                              name='acosh_scaled_neg_inner'), name=scope)


@tf.function
def norm_hyper(vec, dims, keepdims=False, name='HyperbolicNorm'):
    with tf.name_scope(name) as scope:
        return tf.math.sqrt(inner_product_hyper(vec, vec, dims, keepdims=keepdims, name='inner_product'),
                            name=scope)


@tf.function
def to_hyper(x, curvature, name='ToHyperbolic'):
    with tf.name_scope(name) as scope:
        sqrt_curvature = tf.math.sqrt(curvature, name='sqrt_curvature')
        norm_x = tf.math.divide(tf.norm(x, axis=-1, keepdims=True, name='unnormalized_norm_x'),
                                sqrt_curvature, name='norm_x')
        first = tf.math.multiply(sqrt_curvature, tf.math.cosh(norm_x, name='norm_x/cosh'), name='first_coord')
        rest = tf.math.multiply(tf.math.divide(tf.math.sinh(norm_x, name='norm_x/sinh'), norm_x, name='x_multiplier'),
                                x, 'scaled_x')
        return tf.concat([first, rest], axis=-1, name=scope)


@tf.function
def logmap(vec, origin, curvature, dims, name='HyperbolicLogMap'):
    with tf.name_scope(name) as scope:
        inner = tf.math.divide(inner_product_hyper(vec, origin, dims, keepdims=True, name='unnormalized_inner_product'),
                               curvature, name='inner_product')
        distance = tf.math.multiply(tf.math.sqrt(curvature, name='sqrt_curvature'),
                                    tf.math.acosh(tf.math.negative(inner, name='negative_inner')),
                                    name='distance')
        unnormalized = tf.math.add(vec, tf.math.multiply(origin, inner, name='scaled_origin'), name='unnormalized')
        multiplier = tf.math.divide(distance, norm_hyper(vec, dims, keepdims=True, name='norm_unnormalized'), name='normalized_distance')
        return tf.math.multiply(multiplier, unnormalized, name=scope)


@tf.function
def expmap(vec, origin, curvature, dims, name='HyperbolicExpMap'):
    with tf.name_scope(name) as scope:
        norm_vec = tf.math.divide(norm_hyper(vec, dims, keepdims=True, name='norm_vec'),
                                  tf.math.sqrt(curvature, name='sqrt_curvature'))
        origin_coeff = tf.math.cosh(norm_vec, name='origin_coeff')
        scaled_origin = tf.math.multiply(origin_coeff, origin, name='scaled_origin')
        vec_coeff = tf.math.divide(tf.math.sinh(norm_vec, name='sinh_norm_vec'), norm_vec, name='vec_coeff')
        scaled_vec = tf.math.multiply(vec_coeff, vec, name='scaled_vec')
        return tf.math.add(scaled_origin, scaled_vec, name=scope)


@tf.function
def origin_hyper(curvature, dims, name='HyperbolicOrigin'):
    with tf.name_scope(name) as scope:
        return tf.scatter_nd(((0,),), (tf.math.sqrt(curvature, name='sqrt_curvature'),), (dims,), name=scope)


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
        print(input_shape)
        self.input_dims = input_shape[0][-1]
        self.euclidean_dense = self.add_weight('euclidean_dense', shape=(self.input_dims, self.output_dims),
                                               initializer='random_normal', trainable=True)
        self.euclidean_bias = self.add_weight('euclidean_bias', shape=(self.output_dims - 1,),
                                              initializer='zeros', trainable=True)
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.activation_layer = tf.keras.layers.Activation(self.activation)

    def call(self, inputs, training=True):
        vectors, in_curvature, out_curvature = inputs
        in_origin = origin_hyper(in_curvature, self.input_dims, name='in_origin')
        out_origin = origin_hyper(in_curvature, self.output_dims, name='out_origin')
        tangent_vecs = logmap(vectors, in_origin, in_curvature, self.input_dims, name='tangent_vecs')
        weighted_vecs = tf.linalg.matmul(tangent_vecs, self.euclidean_dense, name='weighted_vecs')
        if self.use_bias:
            hyper_weighted_vecs = expmap(weighted_vecs, out_origin, in_curvature, self.output_dims, name='hyper_weighted_vecs')
            bias = tf.concat((tf.constant(0, shape=(1,), dtype=self.compute_dtype), self.euclidean_bias), axis=0, name='bias')
            inner_num = inner_product_hyper(weighted_vecs, bias, self.output_dims, keepdims=True, name='inner_weighted_vecs_bias')
            distance = distance_hyper(out_origin, hyper_weighted_vecs, in_curvature, self.output_dims, keepdims=True, name='distance_hyper_weighted_vecs')
            summed_logs = weighted_vecs + logmap(out_origin, hyper_weighted_vecs, in_curvature, self.output_dims, name='log_hyper_weighted_vecs')
            bias_tangent = bias - inner_num * summed_logs / distance / distance
            biased = expmap(bias_tangent, hyper_weighted_vecs, in_curvature, self.output_dims, name='biased')
            biased_tangent = logmap(biased, out_origin, in_curvature, self.output_dims, name='biased_tangent')
            return expmap(self.dropout(self.activation_layer(biased_tangent)), out_origin, out_curvature, self.output_dims, name=self.name)
        else:
            return expmap(self.dropout(self.activation_layer(weighted_vecs)), out_origin, out_curvature, self.output_dims, name=self.name)


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
        self.embeddings = tf.keras.layers.Embedding(self.num_items, self.embed_dims - 1, mask_zero=True, input_length=input_shape[0][1], name='item_embeddings')
        self.upcast_2x = DenseHyperbolic(2 * self.embed_dims, activation='tanh', use_bias=True, dropout_rate=self.dense_dropout_rate, name='upcast_2x')
        self.upcast_4x = DenseHyperbolic(4 * self.embed_dims, activation='tanh', use_bias=True, dropout_rate=self.dense_dropout_rate, name='upcast_4x')
        self.downcast_final = DenseHyperbolic(self.final_dims, activation='linear', use_bias=True, dropout_rate=self.dense_dropout_rate, name='downcast_final')
        self.item_dropout = tf.keras.layers.Dropout(rate=self.item_dropout_rate, noise_shape=(input_shape[0][0], input_shape[0][1], 1))
        self.dense_dropout = tf.keras.layers.Dropout(rate=self.dense_dropout_rate)
        self.curvature_1 = self.add_weight('curvature_1', shape=(), initializer=tf.constant_initializer(1), trainable=True)
        self.curvature_2 = self.add_weight('curvature_1', shape=(), initializer=tf.constant_initializer(1), trainable=True)
        self.curvature_3 = self.add_weight('curvature_1', shape=(), initializer=tf.constant_initializer(1), trainable=True)

    def call(self, inputs, training=False, mask=None):
        indices, output_curvature = inputs
        item_embeds = self.item_dropout(self.embeddings(indices), training=training)
        summed_embeds = tf.math.reduce_sum(item_embeds * tf.expand_dims(tf.cast(indices > 0, dtype=self.compute_dtype), 2), 1, name='summed_embeds')
        hyper_embeds = to_hyper(summed_embeds, self.curvature_1, name='hyper_embeds')
        upcast_2x = self.upcast_2x((hyper_embeds, self.curvature_1, self.curvature_2), training=training)
        upcast_4x = self.upcast_4x((upcast_2x, self.curvature_2, self.curvature_3), training=training)
        return self.downcast_final((upcast_4x, self.curvature_3, output_curvature), training=training)
