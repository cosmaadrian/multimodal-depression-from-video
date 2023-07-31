import torch

class FractionalPositionalEncoding(torch.nn.Module):
    """Implementation based on the official implementation of the paper entitled
      'Synchronized Audio-Visual Frames with Fractional Positional Encoding for Transformers in Video-to-Text Translation'
    """
    def __init__(self, positions, d_model):
        # TODO finding out what these positions are
        self.positions = positions
        self.d_model = d_model

        self.fpe = get_fractional_position_encoding(self.positions, self.d_model)

    def get_angles_tf(self, pos, i , d_model):
        """
            angle_rates = 1 / tf.math.pow(tf.cast(10000, tf.float32), (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            return pos * angle_rates
        """
        return pos

    def get_fractional_position_encoding(self, positions, d_model):
        """
	    d_model_range = tf.expand_dims(tf.range(d_model, dtype=tf.float32), axis=0)
	    # get angle base vals
	    angle_rads_tf = get_angles_tf(positions, d_model_range, d_model)

	    # create indcs where to use sin or cos
	    sin_idcs = tf.expand_dims(tf.range(0, d_model, 2), axis=0)
	    cos_idcs = tf.expand_dims(tf.range(1, d_model, 2), axis=0)

	    # Choose every element with x % 2 == 0 and apply sin and choose every element with (x+1) % 2 == 0 and apply cos
	    tmp = tf.reduce_sum(tf.one_hot(sin_idcs, d_model), axis=1) * tf.cast(tf.sin(angle_rads_tf),
	                                                                         dtype=my_dtype) + tf.reduce_sum(
	        tf.one_hot(cos_idcs, d_model), axis=1) * tf.cast(tf.cos(angle_rads_tf), dtype=my_dtype)

	    pos_encoding = tf.expand_dims(tmp, axis=0)
	    return tf.cast(pos_encoding, dtype=my_dtype)
        """

        return self.get_angles_tf(positions, d_model_range, d_model)

    def forward(self, x):
        return x + self.fpe
