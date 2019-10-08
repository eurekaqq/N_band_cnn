import tensorflow as tf

from band_cnn import n_band_cnn_layer


class n_band_cnn_model(tf.keras.Model):
    def __init__(self, n, K=64, P=0.5, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.K = K
        self.P = P
        self.n_band_cnn_layer = n_band_cnn_layer(n=self.n, K=self.K, P=self.P)
        self.cnn = tf.keras.layers.Convolution2D(filters=self.K,
                                                 kernel_size=[4, 10],
                                                 strides=(1, 1),
                                                 activation='relu')
        self.dropped_cnn = tf.keras.layers.Dropout(self.P)
        self.bn_cnn = tf.keras.layers.BatchNormalization()
        self.feature = tf.keras.layers.Flatten()
        self.fc_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.n_band_cnn_layer(inputs)
        x = self.cnn(x)
        x = self.dropped_cnn(x)
        x = tf.cast(x, dtype=tf.float32)
        x = self.bn_cnn(x)
        x = self.feature(x)
        x = self.fc_layer(x)

        return x
