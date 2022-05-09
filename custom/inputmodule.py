import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations

class InputModuleLayer(Layer):
    def __init__(self, units):
        self.units = units

        super(DMNInputModuleLayer, self).__init__()

    def build(self, input_shape):
        # self.w = self.add_weight(
        #     shape=(input_shape[-1], self.units),
        #     initializer="random_normal",
        #     trainable=True,
        # )
        # self.b = self.add_weight(
        #     shape=(self.units,), initializer="random_normal", trainable=True
        # )
        self.datlen = input_shape[1]


    def call(self, inputs):
        def get_encoded_fact(i):
            a_sentout = i[0]
            a_nl_input = i[1]
            factout = tf.gather_nd(a_sentout, tf.where(tf.not_equal(a_nl_input, 0)))
            paddings = tf.gather_nd(a_sentout, tf.where(tf.equal(a_nl_input, 0)))

            # factnum = factout.get_shape().as_list()
            # paddings = [[0, self.datlen - factnum], [0, 0]]
            # padded_factout = tf.pad(factout, paddings=paddings, mode='CONSTANT', constant_values=0)
            # return padded_factout

            return tf.concat([factout, paddings], 0)

        sentout = inputs[0]
        nl_input = inputs[1]
        facts_stacked = tf.map_fn(get_encoded_fact, (sentout,nl_input), dtype=tf.float32)

        return facts_stacked

    def get_config(self):
        config = {
            'units': self.units,
        }

        base_config = super(DMNInputModuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
