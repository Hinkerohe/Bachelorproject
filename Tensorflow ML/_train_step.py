from email.policy import default
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

levenshtein = []
train = pandas.read_csv("./cleanData/cleanData/19A20A.tsv", sep= '\t')
# train = pandas.read_csv("./cleanData/cleanData/20D16.tsv", sep= '\t')

# per default
embedding_dim = 20
units = 20

# for kmer encoding:
if Dataset.tec == 2:
    embedding_dim = 20*20*20


BUFFER_SIZE = len(train)
BATCH_SIZE = 4
num_examples = 500

# create the dataset 
dataset_creator = Dataset.NMTDataset('seq-seq')
train_dataset, test_input, test_output, targ_seq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE, train)
input_size = targ_seq # vocab size
output_size = targ_seq # vocab size
units = targ_seq
example_input_batch, example_target_batch = next(iter(train_dataset))

inputs = ([example_input_batch, example_target_batch])

def levenshtein_substitution(sequence1, sequence2):
    """
    Implement the function levenshtein_substitution() which takes two sequences
    of the same length and computes the minimum number of substitutions to
    transform one into another.
    """
    number_substitutions = 0
    for i in range (0, len(sequence1)):
        if sequence1[i] != sequence2[i]:
            number_substitutions +=1
    return number_substitutions

# to translate the decoder output back to sequence
def translate(dec_result, full = True):
    result = []
    def tranl(seq):
        # for kmer encoding
        if Dataset.tec == 2:
            translated = Dataset.k_mer_decoding(seq)
        else:
            translated = Dataset.ordinal_decode(seq)
        return translated
    # for translating the 
    if full:
        pred_seqences = tf.math.argmax(dec_result, axis = -1)
        for i in pred_seqences:
            for j in i:
                result.append(tranl(j))
        with open('./Output1.csv', "a") as I:
            writer = csv.writer(I)
            writer.writerows(result)
    else:
        for i in dec_result:
            result.append(tranl(i))
    
    return result

# definitions for training the NN
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
# optimizer increased to 0.1
optimizer=tf.optimizers.Adam(0.1)
trainable_variables = tf.keras.Model.trainable_variables


#training class
class TrainSequence(tf.keras.Model):
    def __init__(self, embedding_dim, units, use_tf_function=False):
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
    
    def _preprocess(Data):
        return Data
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
       
        input_text, target_text = inputs
        max_target_length = tf.shape(target_text)[1] #time steps
        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = self.encoder(input_text)
            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length-1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target for the decoder's next prediction.
                new_tokens = target_text[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens,
                                                        enc_output, dec_state)
                loss = loss + step_loss

            # Average the loss over all non padding tokens. 
            target_mask = input_text

            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
        
        # Apply an optimization step
        # variables = self.trainable_variables
        variables = self.decoder.trainable_variables + self.encoder.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        # save the translated output and compare
        if get_output:
            dec_result, dec_state_end = self.decoder(target_text ,enc_output, state = dec_state)
            translated = translate(dec_result)
            seq = target_text[0:1,:]
            result = translate(seq, full = False)
            dis = levenshtein_substitution(translated[0], result[0])
            levenshtein.append(dis)



        # Return a dict mapping metric names to current value
        return {'batch_loss': loss}

    def train_step(self, inputs):
        return self._train_step(inputs)


f = open("./Tensorflow ML/DecoderOutput/Output12.csv", "a")

translator = TrainSequence(
    embedding_dim, units)

"""To test the Network:"""

inputs = next(iter(train_dataset))
losses = []
for i in range(1000):
    print(i)
    # if i%10 == 0:
    #     a = translator._train_step(inputs, True)
    # else:
    a = translator._train_step(inputs, True)
    losses.append(a['batch_loss'].numpy())
    print(a, file=f)

"""for plotting:"""

x_axis = levenshtein
y_axis = [i for i in range(0,len(levenshtein))]
print(levenshtein)
plt.plot(y_axis,x_axis)
plt.axis([0, len(levenshtein), 0, 1273])
plt.xlabel('Training Iterations')
plt.ylabel('Levenshtein Distance')
plt.savefig('Levenshtein_Diagram8.png', dpi = 1000)

# inputs = next(iter(train_dataset))

# for i in range(1):
    
#     print(time.process_time())
#     print(translator._train_step(inputs, True))
#     print(time.process_time())

# for x in train_dataset:
#     for n in range(32):
#         inputs = next(iter(train_dataset))
#         # print(x[0][n], x[1][n])
#         # print(translator._train_step(x))
#         print(translator._train_step(inputs))
# 32 64 8

# batch size reducing

# translator.save('./savedModel')
