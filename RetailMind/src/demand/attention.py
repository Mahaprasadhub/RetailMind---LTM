"""Keras additive attention for encoder-decoder demand forecasting."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="RetailMind")
class AdditiveAttention(tf.keras.layers.Layer):
    """Attend from every decoder step to all historical encoder outputs."""

    def __init__(self, attention_units: int, **kwargs):
        super().__init__(**kwargs)
        self.attention_units = attention_units
        self.encoder_projection = tf.keras.layers.Dense(attention_units, use_bias=False)
        self.decoder_projection = tf.keras.layers.Dense(attention_units, use_bias=False)
        self.score_projection = tf.keras.layers.Dense(1, use_bias=False)

    def build(self, input_shape):
        encoder_shape, decoder_shape = input_shape
        self.encoder_projection.build(encoder_shape)
        self.decoder_projection.build(decoder_shape)
        self.score_projection.build((None, None, None, self.attention_units))
        super().build(input_shape)

    def call(self, inputs):
        encoder_outputs, decoder_outputs = inputs
        projected_encoder = self.encoder_projection(encoder_outputs)
        projected_decoder = self.decoder_projection(decoder_outputs)
        combined = tf.tanh(
            tf.expand_dims(projected_encoder, axis=1)
            + tf.expand_dims(projected_decoder, axis=2)
        )
        scores = tf.squeeze(self.score_projection(combined), axis=-1)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, encoder_outputs)
        return context, weights

    def get_config(self):
        config = super().get_config()
        config.update({"attention_units": self.attention_units})
        return config
