import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow import keras
import numpy as np

# This layer implements the episodic memory module for
# dynamic memory networks

class EpisodicMemoryModuleLayer(Layer):
    def __init__(self, units, **kwargs):
        super(EpisodicMemoryModuleLayer, self).__init__(**kwargs)

        self.units = units
        self.memory_hob = 2
        self.gruCell_for_memory = keras.layers.GRUCell(units=self.units)
        self.gruCell_for_episode = keras.layers.GRUCell(units=self.units)

    def build(self, input_shape):
        self.embedding_size = input_shape[2] # input_shape[0][2]
        self.layer1 = keras.layers.Dense(units=self.embedding_size, activation='tanh')
        self.layer2 = keras.layers.Dense(units=1, activation='sigmoid')
        print(input_shape)
        print(self.embedding_size)

    def update(self, facts, questions, memory):
        # modified compute_attention_gate from
        # https://github.com/vchudinov-zz/dynamic_memory_networks_with_keras/blob/ea28a8ba9100c113025fb782d6b4f086edd6c6fb/episodic_memory_module.py
        def compute_attention_gate(i):
            """Computes an attention score over a single fact vector,
            question and memory
            """
            a_facts = i[0]
            a_questions = i[1]
            a_memory = i[2]

            f_i = [a_facts * a_questions,
                   a_facts * a_memory,
                   keras.backend.abs(
                       a_facts - a_questions),
                   keras.backend.abs(
                       a_facts - a_memory),
                   ]

            g_i = self.layer1(keras.backend.concatenate(f_i, axis=1))
            g_i = self.layer2(g_i)

            return g_i

        attentions = tf.map_fn(compute_attention_gate, (facts, questions, memory), dtype=tf.float32)
        h = tf.zeros_like(memory)

        '''
        modified episode.py and __init__.py from
        https://github.com/DongjunLee/dmn-tensorflow/blob/09796bda5f068d8e6d53cfe71da4a234e67c6a7d/dynamic_memory/episode.py
        c : encoded raw text and stacked by each sentence
            shape: fact_count x [batch_size, num_units]
        '''
        c = tf.unstack(tf.transpose(facts, [1, 0, 2]))
        # change attentions shape to match to c's shape
        g = tf.unstack(tf.transpose(attentions, [1, 0, 2]))

        for t, fact in enumerate(c):
            g_t = g[t]  # attention for t-th fact
            h = g_t * self.gruCell_for_episode(fact, h)[0] + (1 - g_t) * h

        return h

    def call(self, inputs, **kwargs):

        # inputs: question, facts
        facts = inputs
        # "End-To-End Memory Networks" Sec. 5, question is fixed with 0.1 vector
        # here we have one question for one fact, question has the same embedding size as a fact
        questions = keras.backend.ones_like(facts)
        questions = questions * 0.1

        # memory is initialized as a question
        memories = list()
        memory_i = questions[:,0,:]
        for _ in range(self.memory_hob):
            episode_i = self.update(facts, questions, memory_i)
            memory_i, _ = self.gruCell_for_memory(episode_i, memory_i)
            memories.append(memory_i)

        return memories

    def get_config(self):
        config = super(EpisodicMemoryModuleLayer, self).get_config()
        config.update({'units': self.units})
        return config

