import tensorflow as tf


class band_cnn_block(tf.keras.layers.Layer):
    def __init__(self, name='band_cnn_block', K=64, P=0.5, **kwargs):
        self.K = K
        self.P = P
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.cnn = tf.keras.layers.Convolution2D(filters=self.K,
                                                 kernel_size=[8, 20],
                                                 strides=(1, 1),
                                                 activation='relu')
        self.dropped_cnn = tf.keras.layers.Dropout(self.P)
        self.pooled_cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                       strides=(2, 2))
        super().build(input_shape)

    def call(self, inputs):
        x = self.cnn(inputs)
        x = self.dropped_cnn(x)
        x = self.pooled_cnn(x)

        return x


class n_band_cnn_layer(tf.keras.layers.Layer):
    def __init__(self, n, name='', K=64, P=0.5, **kwargs):
        self.n = n
        self.K = K
        self.P = P
        self.table = {
            2: [slice(0, 26), slice(14, 40)],
            3: [slice(0, 16), slice(12, 28),
                slice(24, 40)],
            4: [slice(0, 14),
                slice(8, 22),
                slice(16, 30),
                slice(26, 40)]
        }
        _name = name if name else '{}_band_cnn_layer'.format(self.n)
        super().__init__(name=_name, **kwargs)

    def build(self, input_shape):
        # input_shape should be a 40dims mfcc features
        self.split_cnns = [
            band_cnn_block(K=self.K, P=self.P) for _ in range(self.n)
        ]

        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        # yapf:disable
        concatenated_output = tf.keras.layers.concatenate([
            split_cnn(inputs[:, slice_feature, :, :]) for split_cnn, slice_feature
            in zip(self.split_cnns, self.table[self.n])
        ], axis=-1)
        #yapf:enable

        return concatenated_output
