"""Build and inspect the Day 10 encoder-decoder attention model."""

from __future__ import annotations

import tensorflow as tf

from src.demand.attention import AdditiveAttention


def build_seq2seq_model(
    look_back: int,
    horizon: int,
    encoder_features: int,
    decoder_features: int,
    units: int = 64,
    attention_units: int = 32,
) -> tf.keras.Model:
    """Build a many-to-many LSTM with future-calendar decoder inputs and additive attention."""
    encoder_input = tf.keras.layers.Input(
        shape=(look_back, encoder_features), name="historical_sales_and_calendar"
    )
    decoder_input = tf.keras.layers.Input(
        shape=(horizon, decoder_features), name="future_calendar"
    )

    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
        units, return_sequences=True, return_state=True, name="encoder_lstm"
    )(encoder_input)
    decoder_outputs = tf.keras.layers.LSTM(
        units, return_sequences=True, name="decoder_lstm"
    )(decoder_input, initial_state=[state_h, state_c])
    context, _ = AdditiveAttention(attention_units, name="additive_attention")(
        [encoder_outputs, decoder_outputs]
    )
    combined = tf.keras.layers.Concatenate(name="decoder_context")(
        [decoder_outputs, context]
    )
    predictions = tf.keras.layers.Dense(1, name="sales_forecast")(combined)
    model = tf.keras.Model(
        inputs=[encoder_input, decoder_input],
        outputs=predictions,
        name="seq2seq_attention_forecaster",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_attention_model(model: tf.keras.Model) -> tf.keras.Model:
    """Expose the trained model's decoder-by-history attention weight tensor."""
    attention_layer = model.get_layer("additive_attention")
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=attention_layer.output[1],
        name="attention_weight_extractor",
    )


def make_model_inputs(windows, scaler, calendar_feature_count: int):
    """Scale sales with training-only parameters and convert window records to TensorFlow tensors."""
    encoder_values = []
    decoder_values = []
    targets = []
    minimum = scaler["min_value"]
    maximum = scaler["max_value"]
    scale = maximum - minimum
    for window in windows:
        if any(len(row) != calendar_feature_count for row in window["history_calendar"]):
            raise ValueError("Historical calendar feature count does not match model configuration.")
        if any(len(row) != calendar_feature_count for row in window["future_calendar"]):
            raise ValueError("Future calendar feature count does not match model configuration.")
        scaled_sales = [(value - minimum) / scale for value in window["history_sales"]]
        encoder_values.append(
            [
                [scaled_sales[index], *window["history_calendar"][index]]
                for index in range(len(scaled_sales))
            ]
        )
        decoder_values.append(window["future_calendar"])
        targets.append(
            [[(value - minimum) / scale] for value in window["target_sales"]]
        )
    return (
        tf.convert_to_tensor(encoder_values, dtype=tf.float32),
        tf.convert_to_tensor(decoder_values, dtype=tf.float32),
        tf.convert_to_tensor(targets, dtype=tf.float32),
    )
