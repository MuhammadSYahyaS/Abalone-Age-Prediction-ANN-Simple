import tensorflow as tf


def get_model_mlp(n_classes: int, p_drop=0.1):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(rate=p_drop),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(rate=p_drop),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(rate=p_drop),
        tf.keras.layers.Dense(n_classes)
    ])
