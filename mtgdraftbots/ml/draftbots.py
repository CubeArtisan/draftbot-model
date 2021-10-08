import tensorflow as tf

from mtgdraftbots.ml.timeseries import log_timeseries


class SetEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_items, embed_dims, final_dims=None, item_dropout_rate=0.0, dense_dropout_rate=0.0, **kwargs):
        super(SetEmbedding, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.item_dropout_rate = item_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.final_dims = final_dims or embed_dims

    def get_config(self):
        config = super(SetEmbedding, self).get_config()
        config.update({
            "num_items": self.num_items,
            "embed_dims": self.embed_dims,
            "item_dropout_rate": self.item_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "final_dims": self.final_dims,
        })
        return config

    def build(self, input_shape):
        print('INPUT_SHAPE', input_shape)
        self.embeddings = tf.keras.layers.Embedding(self.num_items, self.embed_dims, mask_zero=True, input_length=input_shape[1], name='item_embeddings')
        self.upcast_2x = tf.keras.layers.Dense(2 * self.embed_dims, activation='tanh', use_bias=True, name='upcast_2x')
        # self.upcast_4x = tf.keras.layers.Dense(4 * self.embed_dims, activation='tanh', use_bias=True, name='upcast_4x')
        self.downcast_final = tf.keras.layers.Dense(self.final_dims, activation='linear', use_bias=True, name='downcast_final')
        self.item_dropout = tf.keras.layers.Dropout(rate=self.item_dropout_rate, noise_shape=(input_shape[0], input_shape[1], 1))
        self.dense_dropout = tf.keras.layers.Dropout(rate=self.dense_dropout_rate)

    def call(self, inputs, training=False, mask=None):
        item_embeds = self.item_dropout(self.embeddings(inputs), training=training)
        summed_embeds = tf.math.reduce_sum(item_embeds * tf.expand_dims(tf.cast(inputs > 0, dtype=self.compute_dtype), 2), 1, name='summed_embeds')
        upcast_2x = self.dense_dropout(self.upcast_2x(summed_embeds), training=training)
        # upcast_4x = self.dense_dropout(self.upcast_4x(upcast_2x), training=training)
        return self.downcast_final(upcast_2x)


class ContextualRating(tf.keras.layers.Layer):
    def __init__(self, num_items, embed_dims, context_dims, hyperbolic=False, item_dropout_rate=0.0,
                 dense_dropout_rate=0.0, summary_period=1024, **kwargs):
        super(ContextualRating, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.context_dims = context_dims
        self.hyperbolic = hyperbolic
        self.item_dropout_rate = item_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.summary_period = summary_period

    def get_config(self):
        config = super(ContextualRating, self).get_config()
        config.update({
            "num_items": self.num_items,
            "embed_dims": self.embed_dims,
            "context_dims": self.context_dims,
            "hyperbolic": self.hyperbolic,
            "item_dropout_rate": self.item_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "summary_period": self.summary_period,
        })
        return config

    def build(self, input_shape):
        if self.hyperbolic:
            self.item_curvature = self.add_weight('item_curvature', shape=(), initializer=tf.constant_initializer(1), trainable=True)
            self.item_embeddings = tf.keras.layers.Embedding(self.num_items, self.embed_dims - 1, mask_zero=True, input_length=input_shape[0][1])
            self.pool_embedding = SetHyperEmbedding(self.num_items, self.context_dims, final_dims=self.embed_dims, item_dropout_rate=self.item_dropout_rate, dense_dropout_rate=self.dense_dropout_rate, name='pool_set_embedding')
            self.distance = lambda x, y: distance_hyper(x, y, self.item_curvature, self.embed_dims, name='distances')
        else:
            self.item_embeddings = tf.keras.layers.Embedding(self.num_items, self.embed_dims, mask_zero=True, input_length=input_shape[0][1])
            self.pool_embedding = SetEmbedding(self.num_items, self.context_dims, final_dims=self.embed_dims, item_dropout_rate=self.item_dropout_rate, dense_dropout_rate=self.dense_dropout_rate, name='pool_set_embedding')
            self.distance = lambda x, y: tf.norm(x - y, axis=-1, name='distances')

    def call(self, inputs, training=False):
        item_indices, context_indices = inputs
        if self.hyperbolic:
            item_embeds = to_hyper(self.item_embeddings(item_indices), self.item_curvature, name='item_embeds')
            context_embeds = self.pool_embedding((context_indices, self.item_curvature), training=training)
        else:
            item_embeds = self.item_embeddings(item_indices)
            context_embeds = self.pool_embedding(context_indices, training=training)
        distances = self.distance(item_embeds, tf.expand_dims(context_embeds, 1))
        nonlinear_distances = 1 - tf.math.tanh(distances, name='nonlinear_distances')

        # Logging for tensorboard
        if training and tf.summary.experimental.get_step() % self.summary_period == 0:
            if self.hyperbolic:
                tf.summary.scalar('weights/item_curvature', self.item_curvature)
            tf.summary.histogram('outputs/distances', distances)
            tf.summary.histogram('outputs/nonlinear_distances', nonlinear_distances)
        return nonlinear_distances
        # return -distances


class ItemRating(tf.keras.layers.Layer):
    def __init__(self, num_items, summary_period=1024, **kwargs):
        super(ItemRating, self).__init__(**kwargs)
        self.num_items = num_items
        self.summary_period = summary_period

    def get_config(self):
        config = super(ItemRating, self).get_config()
        config.update({
            "num_items": self.num_items,
            "summary_period": self.summary_period,
        })
        return config

    def build(self, input_shape):
        self.item_rating_logits = self.add_weight('item_rating_logits', shape=(self.num_items,), initializer='random_normal', trainable=True)

    def call(self, inputs, training=False):
        item_ratings = tf.nn.sigmoid(4 * self.item_rating_logits, name='item_ratings')
        ratings = tf.gather(item_ratings, inputs[0], name='ratings')

        # Logging for Tensorboard
        if training and tf.summary.experimental.get_step() % self.summary_period == 0:
            tf.summary.histogram('weights/item_rating_logits', self.item_rating_logits)
            tf.summary.histogram('weights/item_ratings', item_ratings)
        return ratings


class TimeVaryingLinear(tf.keras.layers.Layer):
    def __init__(self, sub_layers, sub_layer_arg_maps, time_shape, summary_period=1024, **kwargs):
        super(TimeVaryingLinear, self).__init__(**kwargs)
        self.sub_layers = sub_layers
        self.sub_layer_arg_maps = sub_layer_arg_maps
        self.time_shape = time_shape
        self.summary_period = summary_period

    def build(self, input_shape):
        self.sub_layer_weights = self.add_weight('sub_layer_weights', shape=(*self.time_shape, len(self.sub_layers)),
                                                 initializer='random_normal', trainable=True)

    def call(self, inputs, training=False):
        sub_layer_args = [tuple(inputs[i] for i in arg_map) for arg_map in self.sub_layer_arg_maps]
        coords, coord_weights = inputs[-2:]
        sub_layer_weights_orig = tf.math.softplus(self.sub_layer_weights, name='sub_layer_weights')
        sub_layer_weight_values = tf.gather_nd(sub_layer_weights_orig, coords, name='sub_layer_weight_values')
        sub_layer_weights = tf.einsum('...xo,...x->...o', sub_layer_weight_values, coord_weights, name='sub_layer_weights')
        with tf.experimental.async_scope():
            sub_layer_values = tf.stack(tuple(layer(args, training=training) for layer, args in zip(self.sub_layers, sub_layer_args)),
                                        axis=-1, name='sub_layer_values')
            scores = tf.einsum('...io,...o->...i', sub_layer_values, sub_layer_weights)

        # Logging for Tensorboard
        if training and tf.summary.experimental.get_step() % self.summary_period == 0:
            temperatures = tf.reduce_sum(sub_layer_weights_orig, axis=-1, keepdims=True)
            relative_sub_layer_weights = sub_layer_weights_orig / temperatures
            log_timeseries(f'weights/temperatures', temperatures, start_index=1)
            for i, sub_layer in enumerate(self.sub_layers):
                log_timeseries(f'weights/sub_layers/multiplier/{sub_layer.name}', tf.gather(sub_layer_weights_orig, i, axis=-1), start_index=-1)
                log_timeseries(f'weights/sub_layers/relative/{sub_layer.name}', tf.gather(relative_sub_layer_weights, i, axis=-1), start_index=-1)
            tf.summary.histogram('outputs/scores', scores)
        return scores


class DraftBot(tf.keras.models.Model):
    def __init__(self, num_items, embed_dims=64, seen_dims=16, picked_dims=32,
                 contrastive_loss_weight=0.0, dropout_picked_rate=0.0, dropout_seen_rate=0.0,
                 dropout_dense_rate=0.0, hyperbolic=False, margin=1, summary_period=1024, **kwargs):
        kwargs.update({'dynamic': False})
        super(DraftBot, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.seen_dims = seen_dims
        self.picked_dims = picked_dims
        self.contrastive_loss_weight = contrastive_loss_weight
        self.dropout_picked_rate = dropout_picked_rate
        self.dropout_seen_rate = dropout_seen_rate
        self.dropout_dense_rate = dropout_dense_rate
        self.hyperbolic = hyperbolic
        self.margin = margin
        self.summary_period = summary_period
        self.loss_weights = {k: tf.constant(v, dtype=self.compute_dtype) for k, v in
            (('log_loss', 1), ('contrastive_loss', contrastive_loss_weight))}
        self.mean_metrics = {k: tf.keras.metrics.Mean() for k in
            ('loss', 'log_loss', 'contrastive_loss')}
        self.accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(1)

    def get_config(self):
        config = super(DraftBot, self).get_config()
        config.update({
            "num_items": self.num_items,
            "embed_dims": self.embed_dims,
            "seen_dims": self.seen_dims,
            "picked_dims": self.picked_dims,
            "contrastive_loss_weight": self.contrastive_loss_weight,
            "dropout_picked_rate": self.dropout_picked_rate,
            "dropout_seen_rate": self.dropout_seen_rate,
            "dropout_dense_rate": self.dropout_dense_rate,
            "hyperbolic": self.hyperbolic,
            "margin": self.margin,
            "summary_period": self.summary_period,
        })
        return config

    def build(self, input_shapes):
        print('TOP_LEVEL', input_shapes)
        picked_contextual_rating = ContextualRating(self.num_items, self.embed_dims, self.picked_dims,
                                                    hyperbolic=self.hyperbolic, item_dropout_rate=self.dropout_picked_rate,
                                                    dense_dropout_rate=self.dropout_dense_rate,
                                                    summary_period=self.summary_period, name='PickedSynergy')
        seen_contextual_rating = ContextualRating(self.num_items, self.embed_dims, self.seen_dims,
                                                  hyperbolic=self.hyperbolic, item_dropout_rate=self.dropout_seen_rate,
                                                  dense_dropout_rate=self.dropout_dense_rate,
                                                  summary_period=self.summary_period, name='SeenSynergy')
        card_rating = ItemRating(self.num_items, summary_period=self.summary_period, name='CardRating')
        self.time_varying = TimeVaryingLinear((card_rating, picked_contextual_rating, seen_contextual_rating),
                                              ((0,), (0, 1), (0, 2)), (3, 15), summary_period=self.summary_period,
                                              name='OracleCombination')

    def call(self, inputs, training=False, mask=None):
        inputs = inputs[:5]
        scores = self.time_varying(inputs, training=training)
        return scores

    def _update_metrics(self, mean_metrics, probs, y_idx):
        for key, value in mean_metrics.items():
            self.mean_metrics[key].update_state(value)
        self.accuracy_metric.update_state(y_idx, probs)
        result = {
            'accuracy': self.accuracy_metric.result(),
        }
        result.update({k: v.result() for k, v in self.mean_metrics.items() if k in mean_metrics})
        return result

    def calculate_loss(self, data, training=False):
        x_vals = data[:5]
        y_idx = data[5]
        scores = self(x_vals, training=training)
        losses = {}
        losses['log_loss'] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_idx, logits=scores, name='log_loss')
        fy_idx = tf.cast(y_idx, dtype=scores.dtype)
        weight_0 = 1 - 2 * fy_idx
        weights = tf.stack((-weight_0, weight_0), axis=-1, name='option_weights')
        losses['contrastive_loss'] = tf.maximum(tf.constant(0, dtype=self.compute_dtype),
                                                tf.constant(self.margin, dtype=scores.dtype)
                                                + tf.math.reduce_sum(weights * scores, axis=-1,
                                                                     name='contrastive_pre_clip'),
                                                name='contrastive_loss')
        losses['loss'] = sum(weight * losses[k] for k, weight in self.loss_weights.items() if k in losses)
        return losses, scores

    def train_step(self, data):
        if len(data) == 1:
            data = data[0]
        with tf.GradientTape() as tape:
            losses, scores = self.calculate_loss(data, training=True)
        gradients = tape.gradient(losses['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self._update_metrics(losses, scores, data[5])

    def test_step(self, data):
        if len(data) == 1:
            data = data[0]
        losses, scores = self.calculate_loss(data, training=False)
        return self._update_metrics(losses, scores, data[5])

    @property
    def metrics(self):
        return list(self.mean_metrics.values()) + [self.accuracy_metric]
