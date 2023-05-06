import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, n_classes: int, p_drop=0.1):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(512, activation="relu")
        self.fc2 = tf.keras.layers.Dense(512, activation="relu")
        self.fc3 = tf.keras.layers.Dense(256, activation="relu")
        self.fc4 = tf.keras.layers.Dense(n_classes)
        self.dropout = tf.keras.layers.Dropout(rate=p_drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
