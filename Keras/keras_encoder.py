from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
import tensorflow_addons as tfa

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

"""Load Dataset"""

train = pandas.read_csv("./cleanData/cleanData/20D16.tsv", sep= '\t')
test = pandas.read_csv("./cleanData/cleanData/20B12.tsv", sep= '\t')

def ordinal(seq):
    """ each unique category value is assigned an integer value, here the index of the letter in the list"""
    amino_acids = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V','*']
    encoded_seq = list(map(lambda x: amino_acids.index(x), seq))
    """delete all * """

    return encoded_seq

def one_hot(seq):
    ordinal_encoded_seq = ordinal(seq)
    def to_binary(number):
        letter = [0 for _ in range(0,21)]
        letter[number] = 1
        return letter
    one_hot_encoded_seq = list(map(lambda x: to_binary(x), ordinal_encoded_seq))
    return one_hot_encoded_seq

class NMTDataset:
    
    def __init__(self, problem_type='seq-seq'):
        self.problem_type = 'seq-seq'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None

    def preprocessing_sequence(self, seq):
        seq = re.sub(r"[*]", "", seq)
        new_seq =""
        # so that each letter will be seen
        for i in seq:
            new_seq = new_seq + i + ' '
        return new_seq

    """Dataset for sequences"""
    def create_dataset(self, file:DataFrame, num_examples):
        if num_examples != None:
            file = file.head(num_examples)
        input_seq = file['In']
        output_seq = file['Out']
        input_seq = [self.preprocessing_sequence(l)   for l in input_seq]
        output_seq = [self.preprocessing_sequence(l)   for l in output_seq]
        return input_seq, output_seq



    # Step 3 and Step 4
    def tokenize(self, dataset):
        
        seq_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower = False, oov_token='<OOV>')
        seq_tokenizer.fit_on_texts(dataset)
        tensor = seq_tokenizer.texts_to_sequences(dataset)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
        ## and pads the sequences to match the longest sequences in the given input
        # tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        # print('tokenizer', tensor, len(tensor), len(tensor[7]))
        return tensor, seq_tokenizer

    def load_dataset(self, file, num_examples=None):
        # creating cleaned input, output pairs
        in_seq, out_seq = self.create_dataset(file, num_examples)
        # encoding
        input_tensor, inp_seq_tokenizer = self.tokenize(in_seq)
        output_tensor, out_seq_tokenizer = self.tokenize(out_seq)
        return input_tensor, inp_seq_tokenizer, output_tensor, out_seq_tokenizer

    """for sequences"""
    def call_seq(self, num_examples, BUFFER_SIZE, BATCH_SIZE, file2):
        test_input_tensor, self.test_inp_seq_tokenizer, test_output_tensor, self.test_out_seq_tokenizer = self.load_dataset(file2, num_examples)
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(test_input_tensor, test_output_tensor, test_size=0.2)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
        
        return train_dataset, val_dataset, self.test_inp_seq_tokenizer, self.test_out_seq_tokenizer

class Encoder(tf.keras.Model):
    """ Embedding layer: bsp ordinal: converts the numbers in fix size vector = hyperparameter
    spezifiy the 
    """
    # vocab_size = 72
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        """size of the fixed vector it depens on how many unique letters are there (bei uns 20 -> mit 32 starten) 
        for each number bekommt man jetzt ein 32 sized vektor 
        und jedes mal wenn er auf die zahl trifft nimmt er den 32 sized vektor.
        -> einfacher für Matrix multiplication"""
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        ##-------- LSTM layer in Encoder ------- ##
        """lstm = LONG-SHORT-MEMORY-LAYER für sequencial data 
        ist ein layer convolutional ist besser für bilder /images
        recurrent layers ==> lesen darüber!!
        """
        self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')



    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state = hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]



## Test Encoder Stack
dataset_creator = NMTDataset('seq-seq')
BUFFER_SIZE = 1273
BATCH_SIZE = 32
num_examples = 100
train_dataset, val_dataset, inp_seq, targ_seq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE, test)
example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape

vocab_inp_size = len(train)
vocab_tar_size = len(train)
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 32
units = 32
steps_per_epoch = num_examples//BATCH_SIZE

## Test Encoder Stack
# print(vocab_inp_size)
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))
