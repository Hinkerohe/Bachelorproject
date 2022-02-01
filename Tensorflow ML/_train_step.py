import tensorflow as tf
import decoder as dec
import keras_encoder as enc
import pandas
import numpy as np
import Dataset
import time
import typing
from typing import Any, Tuple

import tensorflow as tf
import csv
import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


train = pandas.read_csv("./cleanData/cleanData/19A20A.tsv", sep= '\t')
train = pandas.read_csv("./cleanData/cleanData/20D16.tsv", sep= '\t')
embedding_dim = 22
input_size = 20
output_size = 20
units = 4

if Dataset.tec == 2:
    embedding_dim = 20*20*20
BUFFER_SIZE = len(train)
BATCH_SIZE = 4
num_examples = 4

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

dataset_creator = Dataset.NMTDataset('seq-seq')
train_dataset, test_input, test_output, targ_seq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE, train)
input_size = targ_seq # vocab size
output_size = targ_seq # vocab size
units = targ_seq
print(train_dataset)
train_dataset.take
example_input_batch, example_target_batch = next(iter(train_dataset))

inputs = ([example_input_batch, example_target_batch])

# optimizer increase to 0.1
optimizer=tf.optimizers.Adam(0.1)

trainable_variables = tf.keras.Model.trainable_variables

def translate(dec_result):
    result = []
    pred_seqences = tf.math.argmax(dec_result, axis = -1)
    def tranl(batch):
        sl = []
        if Dataset.tec == 2:
            vocab = np.array(Dataset.input_text_processor.get_vocabulary())
            for seq in batch:
                for i in range(0,1270,3):
                    sl.append(vocab[seq[i].numpy()])
        else:
            for seq in batch:
                translated = Dataset.ordinal_decode(seq)
                sl.append(translated)
            return sl

    
    result = list(map(lambda x: tranl(x), pred_seqences))
    
    with open('./Output1.csv', "w") as I:
        writer = csv.writer(I)
        writer.writerows(result)
    

#training class
class TrainSequence(tf.keras.Model):
    def __init__(self, embedding_dim, units, use_tf_function=True):
        super().__init__()
    
    # encoder decoder initialization
        decoder = dec.Decoder(output_size,
                        embedding_dim, units)
        encoder = dec.Encoder2(input_size,
                        embedding_dim, units)
        optimizer = tf.optimizers.Adam()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer


    def _loop_step(self, new_tokens, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]


        dec_result, dec_state = self.decoder(input_token ,enc_output, state = dec_state)
        # `self.loss` returns the total for non-padded tokens
        y = target_token
        y_pred = dec_result.logits
        step_loss = loss_fun(y, y_pred)
        step_loss = tf.reduce_mean(step_loss) #use this if the batch is greater than 1

        return step_loss, dec_state

    def _train_step(self, inputs, get_output = False):
        # print(inputs,'BBBBBBBBBBBBBBBBB')
        
        input_text, target_text = inputs

        max_target_length = tf.shape(target_text)[1]
        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = self.encoder(input_text)
            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)
            # length of one sequence
            for t in tf.range(max_target_length-1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target for the decoder's next prediction.
                new_tokens = target_text[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens,
                                                        enc_output, dec_state)
                loss = loss + step_loss

            # # Average the loss over all non padding tokens. 
            # fürs auffüllen der sequencen auf die gleiche länge gedacht
            target_mask = input_text !=0 

            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
        
        # Apply an optimization step
        # variables = self.trainable_variables
        variables = self.decoder.trainable_variables + self.encoder.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        # print output to file
        if get_output == True:
            dec_result, dec_state_end = self.decoder(target_text ,enc_output, state = dec_state)
            translate(dec_result)


        # Return a dict mapping metric names to current value
        return {'batch_loss': loss}

    def train_step(self, inputs):
        return self._train_step(inputs)

class BatchLogs(tf.keras.callbacks.Callback):
  def __init__(self, key):
    self.key = key
    self.logs = []

  def on_train_batch_end(self, n, logs):
    self.logs.append(logs[self.key])

batch_loss = BatchLogs('batch_loss')

f = open("./Tensorflow ML/DecoderOutput/Output5.csv", "a")

translator = TrainSequence(
    embedding_dim, units)

# for i in range(1):
#     inputs = next(iter(train_dataset))
#     print(i)
#     a = translator._train_step(inputs, True)
#     print(a, file=f)

# inputs = next(iter(train_dataset))

for i in range(1):
    
    print(time.process_time())
    print(translator._train_step(inputs, True))
    print(time.process_time())

# for x in train_dataset:
#     for n in range(32):
#         inputs = next(iter(train_dataset))
#         # print(x[0][n], x[1][n])
#         # print(translator._train_step(x))
#         print(translator._train_step(inputs))
# 32 64 8

# batch size reducing


