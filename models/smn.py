import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, Flatten, TimeDistributed, Activation, dot, concatenate
from tensorflow.keras.models import Model
import numpy as np

from custom.inputmodule import InputModuleLayer
from custom.episodicmemory import EpisodicMemoryModuleLayer


# This dynamic memory networks can use either the input module in "Ask me anything" or the input module in "End-to-End"
# The rest of the architecture follows the architecture in "Ask me anything"

class StatementMemoryNetworks:
    def __init__(self, config):

        config['tdatlen'] = 200

        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        self.batch_size = config['batch_size']

        self.embdims = 100
        self.recdims = 100

        self.memorynetwork_input = config['memorynetwork_input']
        self.config['batch_config'] = [['tdat_sent', 'com'], ['comout']]
        self.sentence_cnt = config['max_sentence_cnt']
        self.sentence_size = config['max_sentence_len']
            # calculating the positional encoding
            
        if self.memorynetwork_input != 'eos-embedding':
            
            self.positional_encoding = np.ones((self.embdims, self.sentence_size), dtype=np.float32)
            for k in range(0, self.embdims):
                for j in range(0, self.sentence_size):
                    self.positional_encoding[k, j] = (1 - j / self.sentence_size) - \
                                                     (k / self.embdims) * (1 - 2 * j / self.sentence_size)

            self.positional_encoding = np.transpose(self.positional_encoding)
            self.positional_encoding = self.positional_encoding.reshape((1, self.sentence_size, self.embdims), )

    def create_model(self):
        print("sentence cnt: {}, sentence len: {}".format(self.sentence_cnt, self.sentence_size))
        dat_input = Input(shape=(self.sentence_cnt, self.sentence_size,))
        print("comlen: {}".format(self.comlen))
        com_input = Input(shape=(self.comlen,))

        dat_input_reshaped = Reshape((self.sentence_cnt * self.sentence_size,), )(dat_input)
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input_reshaped)
        ee_reshaped = Reshape((self.sentence_cnt, self.sentence_size, self.embdims), )(ee)
            
        if self.memorynetwork_input == 'eos-embedding':
            # move batch to the second dimension
            ee_reshaped = tf.transpose(ee_reshaped, [1, 0, 2, 3])

            enc = GRU(self.recdims, return_state=True, return_sequences=False)
            assert ee_reshaped.shape[0] == self.sentence_cnt
            gru_sentence_output_list = []
            for i in range(ee_reshaped.shape[0]):
                encout, state_h = enc(ee_reshaped[i])
                # the return_sequences is false, so the encout is the last output
                last_output = tf.expand_dims(encout, 1)
                gru_sentence_output_list.append(last_output)

            facts = tf.concat(gru_sentence_output_list, 1)
        
        else:
            weighted_word_embedding = keras.layers.Multiply()([ee_reshaped, self.positional_encoding])
            facts = tf.reduce_sum(weighted_word_embedding, 2)  # shape should be: (self.sentence_cnt, self.embdims)


        episodic_memory_module = EpisodicMemoryModuleLayer(self.recdims)
        
        memories = episodic_memory_module(facts)
        last_memory = memories[-1]
        
        enct = GRU(self.recdims, return_state=True, return_sequences=True)
        encoutt, state_ht = enct(ee)

        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = GRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=last_memory)

        memories = tf.convert_to_tensor(memories, dtype=np.float32)
        memories = tf.transpose(memories, perm=[1, 0, 2])

        # smn attention
        attn = dot([decout, memories], axes=[2, 2])
        attn = Activation('softmax')(attn)
        context = dot([attn, memories], axes=[2, 1])
        #context = concatenate([context, decout])

        # traditional tdats encoder attention
        attnt = dot([decout, encoutt], axes=[2, 2])
        attnt = Activation('softmax')(attnt)
        contextt = dot([attnt, encoutt], axes=[2, 1])
        
        context = concatenate([context, contextt, decout])

        # squash into 1/2 of the original size of context
        out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)

        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)

        model = Model(inputs=[dat_input, com_input], outputs=out)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model


if __name__ == "__main__":
    config = dict()

    config['tdatvocabsize'] = 260
    config['comvocabsize'] = 260

    try:
        config['comlen'] = 200
        config['tdatlen'] = 200
    except KeyError:
        pass

    config['batch_size'] = 200
    config['memorynetwork_input'] = 'positional-encoding'
    # config['memorynetwork_input'] = 'eos-embedding'
    config['max_sentence_len'] = 30
    config['max_sentence_cnt'] = 70

    memorynetwork_input = config['memorynetwork_input']
    if memorynetwork_input == 'eos-embedding':
        config['batch_config'] = [['tdat', 'tdat_sent', 'com'], ['comout']]
    else:
        config['batch_config'] = [['tdat_sent', 'com'], ['comout']]


    smn = StatementMemoryNetwork(config)
    config, model = dmn.create_model()
    model.summary()

