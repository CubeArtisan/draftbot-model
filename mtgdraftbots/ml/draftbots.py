import tensorflow as tf

from mtgdraftbots.ml.timeseries import log_timeseries


class DraftBot(tf.keras.models.Model):
    def __init__(self, num_cards, batch_size, embed_dims=64, num_heads=16, summary_period=1024,
                 l2_loss_weight=0.0, l1_loss_weight=0.0, **kwargs):
        kwargs.update({'dynamic': False})
        super(DraftBot, self).__init__(**kwargs)
        self.summary_period = summary_period
        self.batch_size = batch_size
        self.l2_loss_weight = l2_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.loss_metric = tf.keras.metrics.Mean()
        self.log_loss_metric = tf.keras.metrics.Mean()
        self.l2_loss_metric = tf.keras.metrics.Mean()
        self.l1_loss_metric = tf.keras.metrics.Mean()
        self.accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(1)
        self.top_2_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(2)
        self.top_3_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(3)
        self.average_prob_metric = tf.keras.metrics.Mean()
        # Our preprocessing guarantees that the human choice is always at index 0. Our calculations are permutation
        # invariant so this does not introduce any bias.
        self.rating_mult = tf.constant(embed_dims, dtype=self.compute_dtype)
        self.default_target = tf.zeros((batch_size,), dtype=tf.int32)
        self.oracle_weights = self.add_weight('oracle_weights', shape=(3, 15, 6), initializer=tf.constant_initializer(0.5), trainable=True)
        self.card_rating_logits = self.add_weight('card_rating_logits', shape=(num_cards,), initializer='random_uniform',
                                                  trainable=True)
        self.card_embeddings = self.add_weight('card_embeddings', shape=(num_cards, embed_dims),
                                               initializer='random_uniform', trainable=True)
        # self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads, 2 * embed_dims // num_heads, name='self_attention')
        self.summed_weights = self.add_weight('summed_weights', shape=(embed_dims, 2 * embed_dims), initializer='random_normal',
                                              trainable=True)
        # self.nonlinear_weights = self.add_weight('nonlinear_weights', shape=(embed_dims, 2 * embed_dims),
        #                                          initializer='random_normal', trainable=True)
        self.higher_embed_bias = self.add_weight('higher_embed_bias', shape=(2 * embed_dims,), initializer='zeros', trainable=True)
        self.project_back = self.add_weight('project_back', shape=(2 * embed_dims, embed_dims), initializer='random_normal', trainable=True)

    def get_config(self):
        config = super(DraftBot, self).get_config()
        config.update({
            "num_cards": self.num_cards,
            "batch_size": self.batch_size,
            "embed_dims": self.embed_dims,
            "num_heads": self.num_heads,
            "summary_period": self.summary_period,
            "l2_loss_weight": self.l2_loss_weight,
            "l1_loss_weight": self.l1_loss_weight,
            "temp_weight": self.temp_weight,
        })
        return config
        
    def _get_linear_higher_pool_embedding(self, picked_card_embeds, picked_probs, training=False):
        higher_embeddings = tf.nn.bias_add(tf.einsum('bpe,ef->bpf', picked_card_embeds, self.summed_weights, name='higher_embeddings'),
                                           self.higher_embed_bias, name='higher_embeddings_biased')
        regular_embeddings = tf.einsum('bpf,blp,fe->ble', higher_embeddings, picked_probs, self.project_back, name='regular_embeddings')
        return tf.math.l2_normalize(tf.nn.bias_add(regular_embeddings, self.card_embeddings[0], name='biased_regular_embeddings'),
                                    axis=2, epsilon=1e-04, name='pool_embed')

    def _get_linear_pool_embedding(self, picked_card_embeds, picked_probs, training=False):
        summed_embeddings = tf.einsum('bpe,blp->ble', picked_card_embeds, picked_probs, name='summed_embeddings')
        return tf.math.l2_normalize(tf.nn.bias_add(summed_embeddings, self.card_embeddings[0]), axis=2, epsilon=1e-04, name='pool_embed')

    def _get_tanh_pool_embedding(self, picked_card_embeds, picked_probs, reciprocal_total_picked_prob, training=False):
        pre_nonlinear_embeddings = tf.einsum('bpe,blp,bl->ble', tf.math.tanh(picked_card_embeds), picked_probs,
                                             reciprocal_total_picked_prob, 'pre_nonlinear_embeddings')
        return tf.math.atanh(pre_nonlinear_embeddings)

    def _get_attention_pool_embedding(self, picked_cards, picked_probs, training=False):
        picked_card_embeds = tf.gather(self.card_embeddings, picked_cards, name='picked_card_embeds')
        # We don't care about the positions without cards so we mask them out here.
        picked_card_embeds._keras_mask = picked_cards > 0
        attention_mask = tf.logical_and(tf.expand_dims(picked_cards > 0, 1), tf.expand_dims(picked_cards > 0, 2))
        # We use self-attention to model higher-order interactions in the pool of picked cards
        pool_attentions = self.self_attention(picked_card_embeds, picked_card_embeds, attention_mask=attention_mask)
        # We sum weighted by the casting probabilities to collapse down to a single embedding then normalize for cosine similarity.
        return tf.einsum('bpe,blp->ble', pool_attentions, picked_probs, name='unnormalized_pool_embed')

    def _get_pool_embedding(self, picked_cards, picked_probs, training=False):
        picked_card_embeds = tf.gather(self.card_embeddings, picked_cards, name='picked_card_embeds')
        return self._get_linear_pool_embedding(picked_card_embeds, picked_probs, training=training)
        
        # reciprocal_total_picked_prob = 1 / (tf.math.reduce_sum(picked_probs, axis=2) + 1e-04)
        # linear_pool_embeddings = tf.einsum('ble,ef->blf', self._get_linear_pool_embedding(picked_card_embeds, picked_probs, reciprocal_total_picked_prob), self.summed_weights)
        # nonlinear_pool_embeddings = tf.einsum('ble,ef->blf', self._get_tanh_pool_embedding(picked_card_embeds, picked_probs, reciprocal_total_picked_prob), self.nonlinear_weights)
        # higher_embedding = tf.nn.bias_add(linear_pool_embeddings + nonlinear_pool_embeddings, self.embed_bias)
        # combined_embedding = tf.einsum('blf,fe->ble', higher_embedding, self.project_back)
        # return tf.math.l2_normalize(combined_embedding, axis=2, epsilon=1e-04, name='pool_embedding')
    
    def _get_ratings(self, training=False):
        return tf.nn.sigmoid(tf.constant(32, dtype=self.compute_dtype) * self.card_rating_logits, name='card_ratings')
        
    def _get_weights(self, training=False):
        mult = tf.constant(32, dtype=self.compute_dtype)
        if training:
            return mult * self.oracle_weights
        else:
            return tf.math.maximum(mult * self.oracle_weights, tf.constant(0, dtype=self.compute_dtype))
        
    def call(self, inputs, training=False, mask=None):
        in_pack_cards, seen_cards, seen_counts, picked_cards, picked_counts, coords, coord_weights, prob_seens,\
            prob_pickeds, prob_in_packs = inputs
        # We precalculate 8 land combinations that are heuristically verified to be likely candidates for highest scores
        # and guarantee these combinations are diverse. To speed up computation we store the probabilities as unsigned
        # 8-bit fixed-point integers this converts back to float
        prob_seens = tf.cast(prob_seens, dtype=self.compute_dtype) / tf.constant(255, dtype=self.compute_dtype)
        prob_pickeds = tf.cast(prob_pickeds, dtype=self.compute_dtype) / tf.constant(255, dtype=self.compute_dtype)
        prob_in_packs = tf.cast(prob_in_packs, dtype=self.compute_dtype) / tf.constant(255, dtype=self.compute_dtype)
        # Ratings are in the range (0, 1) so we use a sigmoid activation.
        # We gather before the sigmoid to make the gradient sparse over the ratings which helps with LazyAdam.
        card_ratings = self._get_ratings(training=training)
        picked_ratings = tf.gather(card_ratings, picked_cards, name='picked_ratings')
        in_pack_ratings = tf.gather(card_ratings, in_pack_cards, name='in_pack_ratings')
        seen_ratings = tf.gather(card_ratings, seen_cards, name='seen_ratings')
        # We normalize here to allow computing the cosine similarity.
        normalized_embeddings = tf.math.l2_normalize(self.card_embeddings, axis=1, epsilon=1e-04, name='normalized_card_embeds')
        normalized_in_pack_embeds = tf.gather(normalized_embeddings, in_pack_cards, name='normalized_in_pack_embeds')
        normalized_picked_embeds = tf.gather(normalized_embeddings, picked_cards, name='normalized_picked_embeds')
        normalized_seen_embeds = tf.gather(normalized_embeddings, seen_cards,prob_seens, name='normalized_seen_embeds')
        pool_embed = self._get_pool_embedding(picked_cards, prob_pickeds, training=training)
        # We calculate the weight for each oracle as the linear interpolation of the weights on a 2d (3 x 15) grid.
        # There are 6 oracles so we can group them into one variable here for simplicity. The input coords are 4 points
        # on the 2d grid that we'll interpolate between. coord_weights is the weight for each of the four points.
        # Negative oracle weights don't make sense so we apply softplus here to ensure they are positive.
        oracle_weights_orig = self._get_weights(training=training)
        oracle_weight_values = tf.gather_nd(oracle_weights_orig, coords, name='oracle_weight_values')  # (-1, 4, 6)
        oracle_weights = tf.einsum('bxo,bx->bo', oracle_weight_values, coord_weights, name='oracle_weights')
        # These are the per-card oracles to choose between cards in the pack given a lands configuration.
        # The pick synergy oracle for each card is the cosine similarity between its embedding and the pools embedding
        # times the cards casting probability.
        pick_synergy_scores_pre = (tf.einsum('ble,bce->blc', pool_embed, normalized_in_pack_embeds, name='pick_synergies_scores_pre')
                                   + tf.constant(1, dtype=self.compute_dtype)) / tf.constant(2, dtype=self.compute_dtype)
        pick_synergy_scores = (tf.einsum('blc,blc->bcl', pick_synergy_scores_pre, prob_in_packs, name='pick_synergy_scores'))
        # The rating oracle for each card is its rating times its casting probability.
        rating_scores = tf.einsum('bc,blc->bcl', in_pack_ratings, prob_in_packs, name='rating_scores')
        # The internal synergy oracle for a land configuration is the mean cosine similarity times casting probability
        # of the cards that have been picked this draft.
        reciprocal_picked_count = tf.constant(1, dtype=self.compute_dtype) / tf.math.maximum(tf.constant(1, dtype=self.compute_dtype), picked_counts)
        internal_synergy_scores_pre = (tf.einsum('ble,bpe->blp', pool_embed, normalized_picked_embeds, name='internal_synergies_scores_pre')
                                       + tf.constant(1, dtype=self.compute_dtype)) / tf.constant(2, dtype=self.compute_dtype)
        internal_synergy_scores = (tf.einsum('blp,blp,b->bl', internal_synergy_scores_pre, prob_pickeds, reciprocal_picked_count, name='internal_synergy_scores'))
        # These are per-land-configuration oracles that help choose between different land configurations.
        # The colors oracle for a land configuration is the mean rating times casting probability of the cards in the pool.
        colors_scores = tf.einsum('bp,blp,b->bl', picked_ratings, prob_pickeds, reciprocal_picked_count, name='colors_score')
        # The seen synergy oracle for a land configuration is the mean cosine similarity times casting probability
        # of the cards that have been seen this draft with cards that have been seen multiple times included multiple times.
        reciprocal_seen_count = tf.constant(1, dtype=self.compute_dtype) / tf.math.maximum(tf.constant(1, dtype=self.compute_dtype), seen_counts)
        seen_synergy_scores_pre = (tf.einsum('ble,bse->bls', pool_embed, normalized_seen_embeds, name='seen_synergies_scores_pre')
                                   + tf.constant(1, dtype=self.compute_dtype)) / tf.constant(2, dtype=self.compute_dtype)
        seen_synergy_scores = (tf.einsum('bls,bls,b->bl', seen_synergy_scores_pre, prob_seens, reciprocal_seen_count, name='seen_synergy_scores'))
        # The openness oracle for a land configuration is the mean rating times casting probability of the cards that
        # have been seen this draft with cards that have been seen multiple times included multiple times.
        openness_scores = tf.einsum('bs,bls,b->bl', seen_ratings, prob_seens, reciprocal_seen_count, name='openness_scores')
        # Combine the oracle scores linearly according to the oracle weights to get a score for every card/land-configuration pair.
        in_pack_scores = tf.einsum('bclo,bo->bcl', tf.stack([rating_scores, pick_synergy_scores], axis=3),
                                   oracle_weights[:, 0:2], name='in_pack_scores')
        picked_scores = tf.einsum('blo,bo->bl', tf.stack([colors_scores, internal_synergy_scores], axis=2),
                                  oracle_weights[:, 2:4], name='picked_scores')
        seen_scores = tf.einsum('blo,bo->bl', tf.stack([openness_scores, seen_synergy_scores], axis=2),
                                oracle_weights[:, 4:6], name='seen_scores')
        scores = in_pack_scores + tf.expand_dims(picked_scores + seen_scores, 1)
        # scores = in_pack_scores + tf.expand_dims(seen_scores, 1)
        # scores = in_pack_scores
        # max_card_scores = tf.reduce_logsumexp(scores, 2)
        # Here we compute softmax(max(scores, axis=2)) with the operations broken apart to allow optimizing the calculation.
        # Since logsumexp and softmax are translation invariant we shrink the scores so the max score is 0 to reduce numerical instability.
        # choice_probs = tf.nn.softmax(max_card_scores, 1)
        # This is needed to allow masking out the positions without cards so they don't participate in the softmax or logsumexp computations.
        # in_pack_mask = tf.cast(in_pack_cards > 0, dtype=tf.float32)
        max_scores = tf.stop_gradient(tf.reduce_max(scores, [1, 2], keepdims=True, name='max_scores'))
        exp_scores = tf.reduce_sum(tf.math.exp(scores - max_scores, name='exp_scores_pre_sum'), 2, name='exp_scores')
        # # Since the first operation of softmax is exp and the last of logsumexp is log we can combine them into a no-op.
        choice_probs = exp_scores / tf.math.reduce_sum(exp_scores, 1, keepdims=True, name='total_exp_scores')

        # This is all logging for tensorboard. It can't easily be factored into a separate function since it uses so many
        # local variables.
        if training and tf.summary.experimental.get_step() % self.summary_period == 0:
            # num_cards_in_pack = tf.reduce_sum(in_pack_mask, 1)
            max_probs = tf.math.reduce_max(choice_probs, 1)
            max_score = tf.math.reduce_max(scores, [1, 2])
            min_score = tf.math.reduce_min(scores, [1, 2])
            max_diff = max_score - min_score
            min_correct_prob = tf.math.reduce_min(choice_probs[:, 0])
            max_correct_prob = tf.math.reduce_max(choice_probs[:, 0])
            temperatures = tf.math.reduce_sum(oracle_weights_orig[:, :, 0:2], axis=2)
            relative_oracle_weights = oracle_weights_orig / tf.expand_dims(temperatures, 2)
            in_top_1 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 1), tf.float32)
            in_top_2 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 2), tf.float32)
            in_top_3 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 3), tf.float32)

            def to_timeline(key, values, **kwargs):
                tiled_values = tf.expand_dims(values, 1) * coord_weights
                total_values = tf.tensor_scatter_nd_add(tf.zeros((3, 15), dtype=tf.float32), coords, tiled_values)
                count_values = tf.tensor_scatter_nd_add(tf.zeros((3, 15), dtype=tf.float32), coords, coord_weights)
                log_timeseries(key, total_values / count_values, **kwargs)

            with tf.xla.experimental.jit_scope(compile_ops=False):
                tf.summary.histogram('weights/card_ratings', card_ratings)
                log_timeseries(f'weights/oracles/temperature', temperatures, start_index=1)
                for name, idx, values, expand in (('rating', 0, rating_scores, False),
                                                  ('pick_synergy', 1, pick_synergy_scores, False),
                                                  ('colors', 2, colors_scores, True),
                                                  ('internal_synergy', 3, internal_synergy_scores, True),
                                                  ('openness', 4, openness_scores, True),
                                                  ('seen_synergy', 5, seen_synergy_scores, True)
                                                  ):
                    log_timeseries(f'weights/oracles/multiplier/{name}', oracle_weights_orig[:, :, idx], start_index=1)
                    log_timeseries(f'weights/oracles/relative/{name}', relative_oracle_weights[:, :, idx],
                                   start_index=1)
                    if expand:
                        values = tf.expand_dims(values, 1)
                    diffs = tf.reduce_max(values, [1, 2]) - tf.math.reduce_min(values, [1, 2])
                    diffs_with_temp = diffs * oracle_weights[:, idx]
                    relative_diffs = diffs_with_temp / max_diff
                    to_timeline(f'outputs/oracles/weighted_diffs/timeline/{name}', diffs_with_temp)
                    tf.summary.histogram(f'outputs/oracles/diffs/values/{name}', diffs)
                    to_timeline(f'outputs/oracles/relative/diffs/timeline/{name}', relative_diffs)
                    tf.summary.histogram(f'outputs/oracles/values/{name}/correct', values[:, 0])
                tf.summary.histogram(f'outputs/scores/diffs/correct', tf.reduce_max(scores[:, 0], 1) - min_score)
                tf.summary.histogram('outputs/scores/diffs', max_diff)
                to_timeline('outputs/scores/diffs/timeline', max_diff)
                tf.summary.histogram('outputs/probs/correct', choice_probs[:, 0])
                tf.summary.scalar('outputs/probs/correct/min', min_correct_prob)
                tf.summary.scalar('outputs/probs/correct/max_from_one', 1 - max_correct_prob)
                tf.summary.histogram('outputs/probs/max', max_probs)
                to_timeline(f'outputs/probs/correct/timeline', choice_probs[:, 0], start_index=1)
                to_timeline(f'outputs/accuracy/timeline', in_top_1, start_index=1)
                to_timeline(f'outputs/accuracy_top_2/timeline', in_top_2, start_index=1)
                to_timeline(f'outputs/accuracy_top_3/timeline', in_top_3, start_index=1)
        return choice_probs

    def _update_metrics(self, loss, log_loss, l2_loss, l1_loss, probs):
        self.loss_metric.update_state(loss)
        self.log_loss_metric.update_state(log_loss)
        self.l2_loss_metric.update_state(l2_loss)
        self.l1_loss_metric.update_state(l1_loss)
        self.accuracy_metric.update_state(self.default_target, probs)
        self.top_2_accuracy_metric.update_state(self.default_target, probs)
        self.top_3_accuracy_metric.update_state(self.default_target, probs)
        self.average_prob_metric.update_state(probs[:, 0])
        return {
            'loss': self.loss_metric.result(),
            'log_loss': self.log_loss_metric.result(),
            'l2_loss': self.l2_loss_metric.result(),
            'l1_loss': self.l1_loss_metric.result(),
            'accuracy': self.accuracy_metric.result(),
            'accuracy_top_2': self.top_2_accuracy_metric.result(),
            'accuracy_top_3': self.top_3_accuracy_metric.result(),
            'average_prob_correct': self.average_prob_metric.result(),
        }

    def calculate_loss(self, data, training=False):
        probs = self(data, training=training)
        num_cards_in_pack = tf.reduce_sum(tf.cast(data[0] > 0, dtype=self.compute_dtype), 1)
        log_loss = tf.reduce_mean(-tf.math.log(probs[:, 0] + 1e-16) * num_cards_in_pack)
        oracle_weights = self._get_weights(training=training)
        # Dividing by 45 makes it the mean over all of the 2d grid points and the sum over the 6 different oracles.
        l2_loss = tf.reduce_sum(tf.cast(oracle_weights * oracle_weights, dtype=tf.float32)) / 45
        l1_loss = tf.math.reduce_sum(tf.cast(oracle_weights, dtype=tf.float32)) / 45
        loss = log_loss + self.l2_loss_weight * l2_loss + self.l1_loss_weight * l1_loss
        return loss, log_loss, l2_loss, l1_loss, probs

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, log_loss, l2_loss, l1_loss, probs = self.calculate_loss(data[0], training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self._update_metrics(loss, log_loss, l2_loss, l1_loss, probs)

    def test_step(self, data):
        loss, log_loss, l2_loss, l1_loss, probs = self.calculate_loss(data[0], training=False)
        return self._update_metrics(loss, log_loss, l2_loss, l1_loss, probs)

    @property
    def metrics(self):
        return [self.loss_metric, self.log_loss_metric, self.l2_loss_metric, self.l1_loss_metric,
                self.accuracy_metric, self.top_2_accuracy_metric, self.top_3_accuracy_metric, self.average_prob_metric]
