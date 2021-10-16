import tensorflow as tf

# @tf.function
def uniformity_loss(arr, bins, window_size, range_lower=0, range_upper=1, name='UniformityLoss'):
    with tf.name_scope(name) as scope:
        results = []
        range_size = tf.constant(range_upper - range_lower, dtype=arr.dtype, name='range_size')
        inv_bins = tf.constant((range_upper - range_lower) / bins, dtype=arr.dtype, name='inv_bins')
        window_size_const = tf.constant(window_size, dtype=arr.dtype, name='window_size')
        const_lower = tf.constant(range_lower, dtype=arr.dtype, name='range_lower')
        const_upper = tf.constant(range_lower, dtype=arr.dtype, name='range_upper')
        const_zero = tf.constant(0, dtype=arr.dtype, name='const_zero')
        const_one = tf.constant(1, dtype=arr.dtype, name='const_one')
        const_two = tf.constant(2, dtype=arr.dtype, name='const_two')
        with tf.experimental.async_scope():
            for i in range(1 - window_size, bins):
                with tf.name_scope(f'bin_{i}') as scope_inner:
                    const_i = tf.constant(i, dtype=arr.dtype, name=f'i')
                    lower_bound = tf.math.add(tf.math.maximum(tf.math.multiply(const_i, inv_bins, name='lower_bound_unbounded'),
                                                              const_zero, name='lower_bound_unshifted'),
                                              const_lower, name='lower_bound')
                    upper_bound = tf.math.add(tf.math.minimum(tf.math.multiply(tf.math.add(const_i, window_size_const, name='high_bin_number'),
                                                                               inv_bins, name='upper_bound_unbounded'),
                                                              range_size, name='upper_bound_unshifted'),
                                              const_lower, name='upper_bound')
                    filter_cond = tf.math.logical_and(lower_bound <= arr, arr <= upper_bound, name='filter_cond')
                    trimmed = tf.where(filter_cond, arr, tf.zeros_like(arr), name='trimmed')
                    num_filtered = tf.reduce_sum(tf.cast(filter_cond, dtype=arr.dtype, name='float_filter_cond'), name='num_filtered')
                    sum_trimmed = tf.math.reduce_sum(trimmed, name='sum_trimmed')
                    mean_trimmed = tf.math.divide_no_nan(sum_trimmed, num_filtered, name='mean_trimmed')
                    mean_bound = tf.math.reduce_mean([lower_bound, upper_bound], name='mean_bound')
                    abs_error_mean = tf.math.abs(tf.math.subtract(mean_trimmed, mean_bound, 'error_mean'), name=scope_inner)
                    results.append(abs_error_mean)
        stddev = tf.math.reduce_std(arr, name='stddev')
        stddev_diff = tf.math.abs(tf.math.subtract(tf.math.multiply(tf.math.sqrt(tf.constant(12, dtype=arr.dtype, name='twelve')),
                                                                    stddev, name='scaled_stddev'),
                                                   range_size, name='stddev_diff'), name='stddev_diff_abs')
        return tf.math.add(tf.math.reduce_sum(results, name='total_abs_error_mean'), stddev_diff, name=scope), stddev
