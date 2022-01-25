from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.ops.gen_math_ops import arg_max
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental import preprocessing

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import pandas
import numpy as np

import typing
from typing import Any, Tuple
import keras_encoder as enc

train = pandas.read_csv("./cleanData/cleanData/20D16.tsv", sep= '\t')
test = pandas.read_csv("./cleanData/cleanData/20B12.tsv", sep= '\t')

class Encoder2(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, embedding_dim, enc_units):
    super(Encoder2, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size

    # The embedding layer converts tokens to vectors
    self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                               embedding_dim)

    # The GRU RNN layer processes those vectors sequentially.
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   # Return the sequence and state
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, tokens, state=None):
    # 2. The embedding layer looks up the embedding for each token.
    vectors = self.embedding(tokens)

    # 3. The GRU processes the embedding sequence.
    output, state = self.gru(vectors, initial_state=state)

    # 4. Returns the new sequence and its state.
    return output, state

"""for my dataset"""
# dataset_creator = enc.NMTDataset('seq-seq')
# BUFFER_SIZE = 1273 # 72? len(train)
# BATCH_SIZE = 32
# num_examples = 3
# # train_dataset, val_dataset, inp_seq, targ_seq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE, test)
# # print(train_dataset, next(iter(train_dataset)), 'AAAAAAAAAAAAAIIIIIII')
# train_dataset, test_input, test_output, targ_seq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE, test)
# max_vocab_size = 20
# embedding_dim = 32
# units = 20
# example_input_batch, example_target_batch = next(iter(train_dataset))
# # for e in train_dataset:
# #     print(e, 'AAAAAAAA')
# print(example_input_batch, 'AAAAAAAA')
# input_size = 20
# # Encode the input sequence.
# encoder = Encoder2(input_size,
#                   embedding_dim, units)
# example_enc_output, example_enc_state = encoder(example_input_batch)

# example_enc_output_out, example_enc_state_out = encoder(example_target_batch)

# print(f'Input batch, shape (batch): {example_input_batch.shape}')
# print(f'Input batch tokens, shape (batch, s): {example_enc_output_out.shape}')
# print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
# print(f'Encoder state, shape (batch, units): {example_enc_state_out.shape}')



class DecoderInput(typing.NamedTuple):
  new_tokens: Any
  enc_output: Any
  mask: Any

class DecoderOutput(typing.NamedTuple):
  logits: Any
#   attention_weights: Any


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # For Step 1. The embedding layer convets token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                                embedding_dim)

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

        # For step 4. Eqn. (3): converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                        use_bias=False)

        # For step 5. This fully connected layer produces the logits for each
        # output token.
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    def call(self, new_tokens, encoder_output,
            state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        
        # print(new_tokens.shape, 'AAAAAAAAAAA')
        # print(encoder_output.shape, 'BBBBBBBBBBB')

        # Step 1. Lookup the embeddings
        vectors = self.embedding(new_tokens)

        # Step 2. Process one step with the RNN
        rnn_output, state = self.gru(vectors, initial_state=state)

        return DecoderOutput(rnn_output), state
            
# output_size = input_size
# decoder = Decoder(output_size,
#                   embedding_dim, units)

# dec_result, dec_state = decoder( example_target_batch,example_enc_output, state = example_enc_state)

# # dec_result, dec_state = decoder( example_target_batch,example_enc_output, state = dec_state)


# print(f'logits shape: (batch_size, t, output_vocab_size) {dec_result.logits.shape}')
# print(tf.math.argmax(dec_result, axis = -1))
# print()
# print(f'state shape: (batch_size, dec_units) {dec_state.shape}')
# """Compute results losses"""
# print(tf.math.argmax(dec_result, axis = -1))


# loss = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True, reduction='none')
# loss(y_true, y_pred)
# tf.reduce_mean(loss)