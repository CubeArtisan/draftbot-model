import tensorflow as tf

@tf.function
def get_triangular_arr(matrix, name=None):
    with tf.name_scope(name or 'GetTriangularVector') as scope:
        matrix = tf.linalg.band_part(matrix, -1, 0, name='lower_triangular')
        # assuming square matrix
        n = tf.shape(matrix)[0]
        m = tf.cast(n * (n + 1) / 2, dtype=tf.int32)
        # This gets the full row at the bottom
        initial_elements = tf.reverse(matrix[-1, :-1], axis=[0])
        triangular_portion = matrix[..., :-1, :]
        # This makes an upper triangular version where the last row of the lower is on the right of the first row
        rotated_triangular_portion = tf.reverse(tf.reverse(triangular_portion, axis=[1]), axis=[0])
        consolidated = tf.reshape(triangular_portion + rotated_triangular_portion, (n * (n - 1),), name='consolidated')
        result = tf.concat([initial_elements, consolidated[:m - n]], axis=-1, name=scope)
        return result


@tf.function
def uniformity_loss2(arr, level, name='UniformityLoss'):
    with tf.name_scope(name) as scope:
        flat_arr = tf.reshape(arr, (-1,), 'flat_arr')
        diffs = tf.abs(tf.expand_dims(flat_arr, 1) - tf.expand_dims(flat_arr, 0), name='diffs')
        tri_diffs = get_triangular_arr(diffs, name='tri_diffs')
        summed = tf.math.reduce_sum(tri_diffs, name='summed')
        uniformity = tf.cond(summed > 0, lambda: tf.linalg.norm(tri_diffs / summed), lambda: 0.0)
        dimensionality = tf.cast(tf.size(flat_arr, out_type=tf.int64), dtype=tf.float32, name='dimensionality')
        return tf.math.multiply(uniformity, dimensionality, name=scope)


@tf.function
def uniformity_loss(arr, bins, window_size):
    results = []
    count = tf.cast(tf.size(arr, out_type=tf.int64), dtype=arr.dtype)
    with tf.experimental.async_scope():
        inv_bins = tf.constant(1 / bins, dtype=arr.dtype)
        expected_count =count * inv_bins
        for i in range(1 - window_size, bins):
            lower_bound = tf.math.maximum(tf.constant(i, dtype=arr.dtype) * inv_bins, 0.0)
            upper_bound = tf.math.minimum(tf.constant(i + window_size, dtype=arr.dtype) * inv_bins, 1.0)
            filter_cond = tf.math.logical_and(lower_bound <= arr, arr <= upper_bound)
            trimmed = tf.where(filter_cond, arr, 0)
            num_filtered = tf.reduce_sum(tf.cast(filter_cond, dtype=arr.dtype))
            result = tf.cond(num_filtered > 0, lambda: tf.math.abs(tf.math.reduce_sum(trimmed) / num_filtered - (lower_bound + upper_bound) / 2), lambda: tf.constant(0, dtype=arr.dtype))
            results.append(result)
    stddev_diff = tf.math.abs(tf.math.reduce_std(arr) - tf.math.sqrt(1 / 12))
    return sum(results) + stddev_diff, stddev_diff
