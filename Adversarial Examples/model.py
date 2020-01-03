import tensorflow as tf



class FC(tf.keras.models.Model):

    def __init__(self, **kwargs):
        super(FC, self).__init__(**kwargs)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

        self.model.add(tf.keras.layers.Dense(10, tf.nn.softmax))

    def call(self, inputs):
        return self.model(inputs)




