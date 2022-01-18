import tensorflow as tf
import decoder as dec
import keras_encoder as enc
import pandas
import numpy as np

import typing
from typing import Any, Tuple

import tensorflow as tf

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


embedding_dim = 32
input_size = 20
output_size = 20
units = 20

BUFFER_SIZE = 72 # 72? len(train)
BATCH_SIZE = 67
num_examples = 2

loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

train = pandas.read_csv("./cleanData/cleanData/20D16.tsv", sep= '\t')
dataset_creator = enc.NMTDataset('seq-seq')
train_dataset, test_input, test_output, targ_seq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE, train)

example_input_batch, example_target_batch = next(iter(train_dataset))
inputs = (example_input_batch, example_target_batch)

# encoder decoder initialization
decoder = dec.Decoder(output_size,
                embedding_dim, units)
encoder = dec.Encoder2(input_size,
                  embedding_dim, units)

# optimizer 
optimizer=tf.optimizers.Adam()

trainable_variables = tf.keras.Model.trainable_variables
print(trainable_variables)

#training class
class TrainSequence(tf.keras.Model):
    def __init__(self, embedding_dim, units):
        super().__init__()
    
    # encoder decoder initialization
    decoder = dec.Decoder(output_size,
                    embedding_dim, units)
    encoder = dec.Encoder2(input_size,
                    embedding_dim, units)
    optimizer = tf.optimizers.Adam()

    def _loop_step(self, new_tokens, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        decoder = dec.Decoder(output_size,
                        embedding_dim, units)


        dec_result, dec_state = decoder(input_token ,enc_output, state = dec_state)

        # `self.loss` returns the total for non-padded tokens
        y = target_token
        y_pred = dec_result.logits
        print(tf.math.argmax(dec_result, axis = -1))
        step_loss = loss_fun(y, y_pred)
        # print(y, y_pred, 'KKKKKKKKKKKKKKKKKK')

        return step_loss, dec_state

    def _train_step(self, inputs):
        input_text, target_text = inputs
        
        #   (input_tokens, input_mask,
        #    target_tokens, target_mask) = self._preprocess(input_text, target_text)

        max_target_length = tf.shape(target_text)[1]
        # print(max_target_length, 'BBBBBBBBBBBBBBBBBBBBBBBB')
        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = encoder(input_text)

            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)
            # length of one sequence
            # print(max_target_length-1,'BBBBBBBBBBBBBBBBB')
            for t in tf.range(1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target for the decoder's next prediction.
                new_tokens = target_text[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens,
                                                        enc_output, dec_state)
                loss = loss + step_loss
                # print(loss )

            # # Average the loss over all non padding tokens.
            target_mask = input_text !=0

            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        # Return a dict mapping metric names to current value
        return {'batch_loss': loss}

translator = TrainSequence(
    embedding_dim, units)
for n in range(1):
    print(translator._train_step(inputs))