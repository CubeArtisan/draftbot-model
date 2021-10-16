import tensorflow as tf

from mtgdraftbots.ml.hyperbolic import SetHyperEmbedding, distance_hyper, to_hyper
from mtgdraftbots.ml.losses import uniformity_loss
from mtgdraftbots.ml.timeseries import log_timeseries
from mtgdraftbots.ml.utils import dropout


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
        self.embeddings = self.add_weight('item_embeddings', shape=(self.num_items - 1, self.embed_dims),
                                          initializer=tf.random_normal_initializer(0, 1 / self.embed_dims / self.embed_dims),
                                          trainable=True)
        self.upcast_2x = tf.keras.layers.Dense(2 * self.embed_dims, activation='relu', use_bias=True, name='upcast_2x')
        self.upcast_4x = tf.keras.layers.Dense(4 * self.embed_dims, activation='relu', use_bias=True, name='upcast_4x')
        self.downcast_final = tf.keras.layers.Dense(self.final_dims, activation='linear', use_bias=True, name='downcast_final')

    def call(self, inputs, training=False, mask=None):
        embeddings = tf.concat([tf.zeros((1, self.embed_dims), dtype=self.compute_dtype), self.embeddings], 0, name='embeddings')
        dropped_inputs = dropout(inputs, self.item_dropout_rate, training=training, name='inputs_dropped')
        item_embeds = tf.gather(embeddings, dropped_inputs, name='item_embeds')
        summed_embeds = tf.math.reduce_sum(item_embeds, 1, name='summed_embeds')
        normalized_embeds = tf.math.l2_normalize(summed_embeds, axis=-1, epsilon=1e-04, name='normalized_embeds')
        upcast_2x = dropout(self.upcast_2x(normalized_embeds), self.dense_dropout_rate, training=training, scale=2, name='dropout_upcast_2x')
        upcast_4x = dropout(self.upcast_4x(upcast_2x), self.dense_dropout_rate, training=training, scale=2, name='dropout_upcast_4x')
        return self.downcast_final(upcast_4x)


class ContextualRating(tf.keras.layers.Layer):
    def __init__(self, num_items, embed_dims, context_dims, hyperbolic=False, item_dropout_rate=0.0,
                 dense_dropout_rate=0.0, uniformity_weight=0.0, distance_l2_weight=1/256,
                 variance_weight=0.01, summary_period=1024, **kwargs):
        super(ContextualRating, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.context_dims = context_dims
        self.hyperbolic = hyperbolic
        self.item_dropout_rate = item_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.uniformity_weight = uniformity_weight
        self.distance_l2_weight = distance_l2_weight
        self.variance_weight = variance_weight
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
            "uniformity_weight": self.uniformity_weight,
            "distance_l2_weight": self.distance_l2_weight,
            "variance_weight": self.variance_weight,
            "summary_period": self.summary_period,
        })
        return config

    def build(self, input_shape):
        if self.hyperbolic:
            # self.item_curvature = tf.constant(1, shape=(), dtype=self.compute_dtype)
            self.item_curvature = self.add_weight('item_curvature', shape=(), initializer=tf.constant_initializer(1), trainable=True)
            self.item_embeddings = tf.keras.layers.Embedding(self.num_items, self.embed_dims - 1, mask_zero=True, input_length=input_shape[0][1], name='item_embeddings')
            self.pool_embedding = SetHyperEmbedding(self.num_items, self.context_dims, final_dims=self.embed_dims, item_dropout_rate=self.item_dropout_rate, dense_dropout_rate=self.dense_dropout_rate, name='pool_set_embedding')
            self.distance = lambda x, y: distance_hyper(x, y, tf.math.softplus(self.item_curvature), name='distances')
        else:
            self.item_embeddings = tf.keras.layers.Embedding(self.num_items, self.embed_dims, mask_zero=True,
                                                             input_length=input_shape[0][1], name='item_embeddings')
            self.pool_embedding = SetEmbedding(self.num_items, self.context_dims, final_dims=self.embed_dims,
                                               item_dropout_rate=self.item_dropout_rate,
                                               dense_dropout_rate=self.dense_dropout_rate, name='pool_set_embedding')
            self.distance = lambda x, y: tf.norm(x - y, axis=-1, name='distances')

    def call(self, inputs, training=False):
        item_indices, context_indices = inputs
        if self.hyperbolic:
            item_curvature = tf.math.softplus(self.item_curvature, name='item_curvature')
            item_embeds = to_hyper(self.item_embeddings(item_indices), curvature=item_curvature, name='item_embeds')
            context_embeds = self.pool_embedding((context_indices, item_curvature), training=training)
        else:
            item_embeds = self.item_embeddings(item_indices)
            context_embeds = self.pool_embedding(context_indices, training=training)
        distances = self.distance(item_embeds, tf.expand_dims(context_embeds, 1))
        nonlinear_distances = tf.math.subtract(tf.constant(1, dtype=self.compute_dtype), tf.math.tanh(distances, name='nonlinear_distances_neg'), name='nonlinear_distances')
        if self.variance_weight > 0:
            variance_loss = tf.abs(tf.math.subtract(tf.constant(1, dtype=self.compute_dtype),
                                                  tf.math.multiply(tf.constant(12, dtype=self.compute_dtype, name='inverse_unit_variance'),
                                                                   tf.math.reduce_variance(nonlinear_distances, name='variance_nonlinear_distances'),
                                                                   name='scaled_variance_nonlinear_distances'),
                                                  name='diff_variance_nonlinear_distances'),
                                 name='variance_loss')
            self.add_loss(tf.math.multiply(variance_loss, tf.constant(self.variance_weight, dtype=self.compute_dtype, name='variance_loss_weight'),
                                           name='variance_loss_weighted'))
            self.add_metric(variance_loss, name=f'{self.name}_variance_loss')
        if self.distance_l2_weight > 0:
            distance_loss = tf.reduce_mean(tf.math.multiply(distances, distances, name='distances_squared'), name='distance_loss')
            self.add_loss(tf.math.multiply(distance_loss, tf.constant(self.distance_l2_weight, dtype=self.compute_dtype), name='weighted_distance_l2_loss'))
            self.add_metric(distance_loss, name=f'{self.name}_distance_l2_loss')
        if self.uniformity_weight > 0:
            loss, variance = uniformity_loss(nonlinear_distances, 32, 8, name='nonlinear_distances_uniformity')
            self.add_loss(tf.math.multiply(loss, tf.constant(self.uniformity_weight, dtype=self.compute_dtype, name='uniformity_weight'), name='uniformity_loss_weighted'))
            self.add_metric(loss, name=f'{self.name}_uniformity_loss')
            self.add_metric(variance, name=f'{self.name}_variance')

        # Logging for tensorboard
        if training and tf.summary.experimental.get_step() % self.summary_period == 0:
            if self.hyperbolic:
                tf.summary.scalar('weights/item_curvature', tf.nn.softplus(self.item_curvature))
            tf.summary.histogram('outputs/distances', distances)
            tf.summary.histogram('outputs/nonlinear_distances', nonlinear_distances)
        return nonlinear_distances


class ItemRating(tf.keras.layers.Layer):
    def __init__(self, num_items, uniformity_weight=1.0, summary_period=1024, **kwargs):
        super(ItemRating, self).__init__(**kwargs)
        self.num_items = num_items
        self.uniformity_weight = uniformity_weight
        self.summary_period = summary_period

    def get_config(self):
        config = super(ItemRating, self).get_config()
        config.update({
            "num_items": self.num_items,
            "uniformity_weight": self.uniformity_weight,
            "summary_period": self.summary_period,
        })
        return config

    def build(self, input_shape):
        self.item_rating_logits = self.add_weight('item_rating_logits', shape=(self.num_items - 1,),
                                                  initializer=tf.random_normal_initializer(0, 0.25),
                                                  trainable=True)

    def call(self, inputs, training=False):
        item_ratings = tf.concat([tf.zeros((1,), dtype=self.compute_dtype),
                                  tf.nn.sigmoid(self.item_rating_logits, name='positive_item_ratings')],
                                 0, name='item_ratings')
        ratings = tf.gather(item_ratings, inputs[0], name='ratings')
        if self.uniformity_weight > 0:
            loss, variance = uniformity_loss(item_ratings, 32, 8, name='ratings_uniformity')
            self.add_loss(tf.math.multiply(loss, tf.constant(self.uniformity_weight, dtype=self.compute_dtype, name='uniformity_weight'), 'uniformity_loss_weighted'))
            self.add_metric(loss, name=f'{self.name}_uniformity_loss')
            self.add_metric(variance, name=f'{self.name}_variance')

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
                                                 initializer=tf.constant_initializer(16),
                                                 trainable=True)

    def get_config(self):
        config = super(TimeVaryingLinear, self).get_config()
        config.update({
            "sub_layers": self.sub_layers,
            "sub_layer_arg_maps": self.sub_layer_arg_maps,
            "time_shape": self.time_shape,
            "summary_period": self.summary_period,
        })
        return config

    def call(self, inputs, training=False):
        sub_layer_args = [tuple(inputs[i] for i in arg_map) for arg_map in self.sub_layer_arg_maps]
        coords, coord_weights = inputs[-2:]
        sub_layer_weights_orig = tf.math.softplus(self.sub_layer_weights, name='sub_layer_weights')
        sub_layer_weight_values = tf.gather_nd(sub_layer_weights_orig, coords, name='sub_layer_weight_values')
        sub_layer_weights = tf.einsum('...xo,...x->...o', sub_layer_weight_values, coord_weights, name='sub_layer_weights')
        sub_layer_values_pre_stack = tuple(layer(args, training=training) for layer, args in zip(self.sub_layers, sub_layer_args))
        sub_layer_values = tf.stack(sub_layer_values_pre_stack, axis=-1, name='sub_layer_values')
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
                 contrastive_loss_weight=0.0, dropout_picked=0.0, dropout_seen=0.0,
                 dropout_dense=0.0, hyperbolic=False, margin=1, rating_uniformity_weight=1.0,
                 picked_synergy_uniformity_weight=0.1, seen_synergy_uniformity_weight=0.1,
                 log_loss_weight=1.0, picked_variance_weight=0.0, picked_distance_l2_weight=0.0,
                 seen_variance_weight=0.0, seen_distance_l2_weight=0.0, summary_period=1024, **kwargs):
        kwargs.update({'dynamic': False})
        super(DraftBot, self).__init__(**kwargs)
        self.num_items = num_items
        self.embed_dims = embed_dims
        self.seen_dims = seen_dims
        self.picked_dims = picked_dims
        self.contrastive_loss_weight = contrastive_loss_weight
        self.dropout_picked = dropout_picked
        self.dropout_seen = dropout_seen
        self.dropout_dense = dropout_dense
        self.hyperbolic = hyperbolic
        self.margin = margin
        self.rating_uniformity_weight = rating_uniformity_weight
        self.picked_synergy_uniformity_weight = picked_synergy_uniformity_weight
        self.seen_synergy_uniformity_weight = seen_synergy_uniformity_weight
        self.log_loss_weight = log_loss_weight
        self.picked_variance_weight = picked_variance_weight
        self.picked_distance_l2_weight = picked_distance_l2_weight
        self.seen_variance_weight = seen_variance_weight
        self.seen_distance_l2_weight = seen_distance_l2_weight
        self.summary_period = summary_period

    def get_config(self):
        config = super(DraftBot, self).get_config()
        config.update({
            "num_items": self.num_items,
            "embed_dims": self.embed_dims,
            "seen_dims": self.seen_dims,
            "picked_dims": self.picked_dims,
            "contrastive_loss_weight": self.contrastive_loss_weight,
            "dropout_picked": self.dropout_picked,
            "dropout_seen": self.dropout_seen,
            "dropout_dense": self.dropout_dense,
            "hyperbolic": self.hyperbolic,
            "margin": self.margin,
            "rating_uniformity_weight": self.rating_uniformity_weight,
            "picked_synergy_uniformity_weight": self.picked_synergy_uniformity_weight,
            "seen_synergy_uniformity_weight": self.seen_synergy_uniformity_weight,
            "log_loss_weight": self.log_loss_weight,
            "picked_variance_weight": self.picked_variance_weight,
            "picked_distance_l2_weight": self.picked_distance_l2_weight,
            "seen_variance_weight": self.seen_variance_weight,
            "seen_distance_l2_weight": self.seen_distance_l2_weight,
            "summary_period": self.summary_period,
        })
        return config

    def build(self, input_shapes):
        picked_contextual_rating = ContextualRating(self.num_items, self.embed_dims, self.picked_dims,
                                                    hyperbolic=self.hyperbolic, item_dropout_rate=self.dropout_picked,
                                                    dense_dropout_rate=self.dropout_dense,
                                                    uniformity_weight=self.picked_synergy_uniformity_weight,
                                                    variance_weight=self.picked_variance_weight,
                                                    distance_l2_weight=self.picked_distance_l2_weight,
                                                    summary_period=self.summary_period, name='PickedSynergy')
        seen_contextual_rating = ContextualRating(self.num_items, self.embed_dims, self.seen_dims,
                                                  hyperbolic=self.hyperbolic, item_dropout_rate=self.dropout_seen,
                                                  dense_dropout_rate=self.dropout_dense,
                                                  uniformity_weight=self.seen_synergy_uniformity_weight,
                                                  variance_weight=self.seen_variance_weight,
                                                  distance_l2_weight=self.seen_distance_l2_weight,
                                                  summary_period=self.summary_period, name='SeenSynergy')
        card_rating = ItemRating(self.num_items, uniformity_weight=self.rating_uniformity_weight,
                                 summary_period=self.summary_period, name='CardRating')
        self.time_varying = TimeVaryingLinear((card_rating, picked_contextual_rating, seen_contextual_rating),
                                              ((0,), (0, 1), (0, 2)), (3, 15), summary_period=self.summary_period,
                                              name='OracleCombination')

    def call(self, inputs, training=False, mask=None):
        loss_dtype = tf.float32
        scores = tf.cast(self.time_varying(inputs[:5], training=training), dtype=loss_dtype)
        if len(inputs) > 5:
            y_idx = inputs[5]
            log_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_idx, logits=scores, name='log_losses')
            self.add_loss(tf.math.multiply(tf.math.reduce_mean(log_losses, name='log_loss'),
                                           tf.constant(self.log_loss_weight, dtype=loss_dtype, name='log_loss_weight'),
                                           name='log_loss_weighted'))
            self.add_metric(log_losses, name='log_loss')
            fy_idx = tf.cast(y_idx, dtype=scores.dtype)
            weight_0 = tf.math.subtract(tf.constant(1, dtype=loss_dtype),
                                        tf.math.multiply(tf.constant(2, dtype=loss_dtype), fy_idx),
                                        name='weight_0')
            scaled = tf.math.multiply(tf.reshape(weight_0, (-1, 1)), scores, 'scaled_scores')
            weights = tf.stack((-weight_0, weight_0), axis=-1, name='option_weights')
            contrastive_pre_clip = tf.math.add(tf.constant(self.margin, dtype=scores.dtype),
                                               tf.math.subtract(scaled[:, 1], scaled[:, 0], name='difference_of_scores'),
                                               name='contrastive_pre_clip'),
            contrastive_losses = tf.maximum(tf.constant(0, dtype=loss_dtype), contrastive_pre_clip,
                                            name='contrastive_losses')
            # if training and tf.summary.experimental.get_step() % self.summary_period == 0:
            #     tf.summary.histogram(f'outputs/contrast_margins', contrastive_pre_clip)
            #     tf.summary.histogram(f'outputs/contrastive_losses', contrastive_losses)
            #     tf.summary.histogram(f'outputs/log_losses', log_losses)
            self.add_loss(tf.math.multiply(tf.math.reduce_mean(contrastive_losses, name='contrastive_loss'),
                                           tf.constant(self.contrastive_loss_weight, dtype=loss_dtype, name='contrastive_loss_weight'),
                                           name='contrastive_loss_weighted'))
            self.add_metric(contrastive_losses, name='contrastive_loss')
            accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_idx, scores)
            self.add_metric(accuracy, name='accuracy')
        return scores
