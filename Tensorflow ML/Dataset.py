from logging import raiseExceptions
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numpy import int64
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import itertools
import pandas

amino_acids = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
train = pandas.read_csv("./cleanData/cleanData/20D16.tsv", sep= '\t')

def ordinal_encode(seq):
    """ each unique category value is assigned an integer value, here the index of the letter in the list"""
    seq = re.sub(r"[*]", "", seq)
    encoded_seq = list(map(lambda x: amino_acids.index(x), seq))
    return encoded_seq

def ordinal_decode(seq):
    """ each unique category value is assigned an integer value, here the index of the letter in the list"""
    decoded_seq = list(map(lambda x: amino_acids[x], seq))
    decoded_seq = ''.join(map(str, decoded_seq))
    return decoded_seq

def one_hot_encoding(seq):
    def ordinal(seq):
        """ each unique category value is assigned an integer value, here the index of the letter in the list"""
        seq = re.sub(r"[*]", "", seq)
        encoded_seq = list(map(lambda x: amino_acids.index(x), seq))
        return encoded_seq
    
    def to_binary(number):
        letter = [0 for _ in range(0,20)]
        letter[number] = 1
        return letter
    
    ordinal_encoded_seq = ordinal(seq)
    one_hot_encoded_seq = list(map(lambda x: to_binary(x), ordinal_encoded_seq))
    return one_hot_encoded_seq

def one_hot_decoding(seq):
    int_seq = list(map(lambda x: x.index(1), seq))
    decoded_seq = ordinal_decode(int_seq)
    return decoded_seq

def k_mer(seq):
    n = 3
    result = []
    for i in range(1273-n):
        result.append(seq[i : i + n])
    return result



def encoding(n):
    if n == 0:
        input_ordinal_text_processor = tf.keras.layers.TextVectorization(
            split = None,
            max_tokens=22)
        input_ordinal_text_processor.adapt(amino_acids)
        r = input_ordinal_text_processor
    if n == 1:
        input_one_hot_text_processor = tf.keras.layers.IntegerLookup(
            output_mode = "one_hot")
        vocab = [i for i in range(20)]
        input_one_hot_text_processor.adapt(vocab)
        r = input_one_hot_text_processor
        # print(input_one_hot_text_processor.get_vocabulary())
    if n == 2:
        k = 3
        input_ordinal_text_processor = tf.keras.layers.TextVectorization(
            split = None)
        cross_join_list = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        for i in range(1,k):
            cross_join_list = [x + y for x, y in itertools.product(cross_join_list, amino_acids)]
        vocab = cross_join_list
        input_ordinal_text_processor.adapt(vocab)
        r = input_ordinal_text_processor
    return r

tec = 0
input_text_processor = encoding(tec)


class NMTDataset:
    
    def __init__(self, problem_type='seq-seq'):
        self.problem_type = 'seq-seq'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None

    def preprocessing_sequence(self, seq):
        seq = re.sub(r"[*]", "", seq)
        # new_seq =""
        # new_seq = ordinal_encode(seq)
        if tec == 0:
            new_seq = ordinal_encode(seq)
        elif tec == 1:
            new_seq = ordinal_encode(seq)
            new_seq = one_hot_encoding(seq)
            # new_seq = input_text_processor(new_seq)
        elif tec == 2:
            new_seq = k_mer(seq)
            new_seq = input_text_processor(new_seq)
        # print(new_seq)
        return new_seq

    """Dataset for sequences"""
    def create_dataset(self, file:DataFrame, num_examples):
        if num_examples != None:
            file = file.head(num_examples)
        input_seq = file['In']
        output_seq = file['Out']
        input_seq = [self.preprocessing_sequence(l)   for l in input_seq]
        output_seq = [self.preprocessing_sequence(l)   for l in output_seq]
        input_seq = tf.convert_to_tensor(input_seq, dtype = int64)
        output_seq = tf.convert_to_tensor(output_seq, dtype = int64)
        print(input_seq)
        return input_seq, output_seq

    # Step 3 and Step 4
    def tokenize(self, dataset):
        
        seq_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower = False, oov_token='<OOV>')
        seq_tokenizer.fit_on_texts(dataset)
        tensor = seq_tokenizer.texts_to_sequences(dataset)
        return tensor, seq_tokenizer

    def load_dataset(self, file, num_examples=None):
        # creating cleaned input, output pairs
        in_seq, out_seq = self.create_dataset(file, num_examples)
        vocab_size = input_text_processor.vocabulary_size()
        return in_seq, vocab_size, out_seq, vocab_size

    """for sequences"""
    def call_seq(self, num_examples, BUFFER_SIZE, BATCH_SIZE, file2):
        test_input_tensor, self.test_inp_seq_tokenizer, test_output_tensor, self.test_out_seq_tokenizer = self.load_dataset(file2, num_examples)
        # train_dataset = tf.data.Dataset.from_tensor_slices(([test_input_tensor], [test_output_tensor]))
        train_dataset = tf.data.Dataset.from_tensor_slices((test_input_tensor, test_output_tensor))
        print(train_dataset)
        # print(list(train_dataset.as_numpy_iterator()), 'OOOOOOOOOOOOOOO')
        train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
        # print(list(train_dataset.as_numpy_iterator()), 'MMMMMMMMMMMMMM')
        return train_dataset, test_input_tensor, test_output_tensor, self.test_out_seq_tokenizer
#67:19

"""Dataset test"""
# args = [22, 20, 9, 4, 9, 2, 4, 2, 2, 14, 2, 4, 3, 3, 11, 18, 4, 6, 2, 5, 5, 17, 5, 11, 2, 14, 14, 8, 15, 5, 6, 3, 9, 5, 17, 7, 4, 15, 15, 14, 13, 12, 4, 9, 17, 3, 3, 4, 2, 19, 3, 5, 11, 13, 2, 9, 2, 14, 9, 9, 3, 6, 4, 5, 21, 9, 19, 8, 10, 19, 4, 3, 7, 5, 6, 7, 5, 12, 17, 9, 13, 6, 14, 4, 2, 14, 9, 6, 13, 7, 4, 15, 9, 8, 3, 5, 16, 12, 3, 6, 10, 10, 17, 7, 21, 10, 9, 7, 5, 5, 2, 13, 3, 12, 5, 11, 3, 2, 2, 10, 4, 6, 6, 8, 5, 6, 4, 4, 10, 12, 4, 18, 16, 9, 11, 9, 18, 6, 13, 14, 9, 2, 7, 4, 15, 15, 19, 12, 6, 6, 12, 3, 21, 20, 16, 3, 16, 9, 17, 4, 15, 3, 3, 8, 6, 6, 18, 5, 9, 16, 15, 4, 3, 11, 14, 9, 2, 20, 13, 2, 16, 7, 12, 11, 7, 6, 9, 12, 6, 2, 17, 16, 9, 4, 9, 12, 6, 10, 13, 7, 15, 9, 12, 10, 15, 3, 12, 19, 5, 14, 10, 6, 2, 4, 17, 13, 2, 14, 11, 7, 9, 3, 8, 2, 16, 14, 2, 4, 13, 2, 14, 10, 7, 10, 6, 10, 5, 17, 9, 11, 5, 2, 2, 8, 2, 19, 17, 3, 15, 2, 5, 14, 7, 13, 3, 3, 3, 7, 21, 5, 8, 7, 8, 8, 8, 15, 15, 4, 7, 15, 2, 11, 14, 17, 5, 9, 2, 2, 12, 15, 6, 16, 6, 7, 5, 10, 5, 13, 8, 4, 13, 18, 8, 2, 13, 14, 2, 3, 16, 5, 12, 18, 5, 2, 12, 3, 9, 5, 4, 16, 12, 7, 10, 15, 11, 5, 3, 6, 9, 17, 4, 11, 14, 5, 16, 3, 10, 4, 17, 9, 14, 6, 10, 5, 6, 2, 18, 14, 9, 7, 16, 4, 9, 6, 8, 5, 17, 9, 8, 3, 4, 15, 8, 21, 6, 17, 12, 17, 10, 3, 6, 18, 4, 8, 13, 15, 3, 4, 2, 15, 6, 3, 8, 3, 9, 3, 5, 9, 12, 18, 15, 7, 4, 3, 14, 5, 12, 2, 6, 13, 2, 18, 9, 5, 6, 4, 15, 8, 13, 3, 9, 4, 10, 17, 7, 13, 16, 4, 17, 11, 10, 8, 14, 7, 11, 5, 7, 12, 10, 8, 13, 15, 6, 15, 12, 2, 14, 13, 13, 9, 5, 7, 18, 4, 10, 8, 21, 6, 3, 6, 6, 2, 13, 3, 12, 4, 7, 7, 6, 15, 6, 15, 2, 15, 17, 2, 9, 17, 12, 3, 6, 2, 12, 14, 9, 16, 17, 13, 10, 3, 5, 16, 10, 15, 11, 8, 7, 3, 12, 14, 18, 6, 7, 4, 16, 7, 9, 6, 18, 15, 9, 14, 2, 11, 3, 15, 7, 9, 11, 14, 5, 6, 7, 4, 7, 15, 11, 14, 15, 17, 4, 4, 4, 2, 3, 9, 16, 2, 2, 19, 8, 14, 8, 5, 4, 18, 7, 14, 12, 12, 3, 5, 6, 2, 4, 12, 6, 12, 18, 4, 6, 9, 6, 9, 6, 7, 2, 5, 7, 5, 7, 4, 2, 5, 16, 3, 6, 12, 12, 9, 2, 14, 9, 11, 11, 9, 7, 17, 13, 10, 8, 13, 5, 5, 13, 8, 4, 17, 13, 14, 11, 5, 2, 16, 10, 2, 13, 10, 5, 14, 18, 3, 9, 7, 7, 4, 3, 4, 10, 5, 14, 7, 5, 6, 5, 3, 6, 11, 4, 8, 4, 2, 15, 11, 7, 4, 6, 18, 5, 16, 4, 14, 4, 8, 10, 19, 8, 13, 11, 2, 5, 14, 5, 21, 17, 4, 15, 3, 5, 7, 3, 6, 4, 9, 11, 5, 17, 8, 7, 18, 2, 10, 7, 8, 16, 19, 4, 6, 6, 3, 15, 16, 18, 13, 10, 14, 10, 7, 8, 7, 10, 18, 8, 3, 15, 11, 5, 11, 5, 6, 3, 19, 17, 17, 8, 17, 3, 4, 8, 3, 11, 3, 10, 10, 8, 15, 5, 20, 3, 2, 7, 8, 16, 6, 3, 4, 8, 15, 3, 6, 6, 3, 10, 8, 10, 14, 5, 6, 9, 5, 10, 3, 4, 5, 5, 16, 10, 2, 14, 4, 3, 20, 8, 12, 5, 3, 4, 13, 18, 5, 20, 15, 10, 18, 7, 13, 3, 5, 16, 18, 3, 6, 2, 2, 2, 11, 15, 7, 3, 9, 18, 5, 11, 2, 6, 17, 8, 2, 5, 7, 10, 8, 4, 16, 11, 13, 12, 6, 5, 11, 16, 4, 9, 8, 11, 4, 12, 11, 10, 15, 12, 5, 14, 14, 10, 12, 13, 9, 7, 7, 9, 6, 9, 3, 11, 10, 2, 14, 13, 14, 3, 12, 14, 3, 12, 17, 3, 9, 10, 16, 13, 2, 2, 9, 6, 12, 4, 5, 2, 8, 13, 8, 7, 9, 10, 12, 11, 15, 7, 13, 18, 2, 7, 13, 10, 8, 8, 17, 13, 2, 10, 18, 8, 11, 12, 9, 6, 7, 2, 5, 4, 2, 14, 14, 2, 2, 5, 13, 16, 20, 10, 8, 11, 15, 5, 3, 8, 2, 2, 8, 7, 5, 10, 5, 3, 7, 21, 5, 9, 7, 8, 7, 8, 8, 2, 11, 10, 14, 9, 8, 20, 11, 20, 8, 15, 17, 9, 6, 7, 10, 7, 4, 5, 11, 6, 4, 2, 15, 16, 6, 11, 12, 2, 10, 8, 6, 11, 9, 6, 3, 8, 10, 7, 12, 10, 11, 13, 3, 2, 3, 3, 5, 8, 3, 8, 2, 7, 12, 2, 11, 13, 4, 4, 6, 11, 6, 8, 11, 8, 2, 6, 5, 2, 4, 12, 11, 2, 3, 3, 6, 9, 7, 8, 10, 3, 3, 4, 2, 6, 13, 10, 2, 3, 17, 2, 13, 12, 4, 16, 8, 16, 4, 11, 10, 13, 17, 2, 10, 5, 7, 17, 2, 11, 3, 2, 11, 5, 15, 4, 5, 11, 11, 2, 10, 17, 8, 8, 16, 10, 17, 8, 3, 8, 6, 2, 8, 8, 5, 12, 20, 3, 16, 18, 4, 2, 7, 11, 3, 12, 17, 4, 13, 9, 18, 7, 12, 7, 15, 19, 2, 20, 3, 9, 14, 11, 3, 8, 14, 19, 7, 4, 4, 9, 2, 19, 4, 5, 15, 4, 14, 8, 11, 16, 12, 6, 9, 5, 5, 8, 14, 8, 10, 18, 19, 13, 7, 12, 8, 19, 9, 14, 17, 16, 7, 4, 9, 4, 3, 6, 7, 5, 19, 21, 9, 4, 5, 11, 17, 6, 9, 15, 16, 14, 11, 10, 10, 5, 5, 13, 6, 5, 9, 4, 3, 7, 6, 18, 13, 4, 4, 10, 7, 10, 4, 6, 6, 5, 4, 15, 13, 14, 2, 11, 14, 16, 2, 13, 3, 9, 12, 16, 16, 2, 13, 12, 15, 9, 12, 6, 19, 5, 3, 14, 13, 4, 13, 2, 7, 13, 10, 3, 7, 10, 6, 8, 3, 4, 4, 6, 10, 11, 12, 16, 10, 13, 17, 2, 6, 16, 4, 8, 12, 6, 2, 6, 16, 3, 2, 10, 13, 2, 11, 16, 2, 7, 12, 15, 16, 11, 15, 10, 12, 21, 14, 21, 15, 10, 21, 2, 7, 9, 10, 8, 7, 2, 10, 8, 10, 4, 20, 4, 5, 10, 20, 2, 18, 18, 20, 5, 3, 18, 18, 3, 18, 2, 12, 7, 18, 18, 3, 18, 7, 3, 18, 18, 12, 9, 13, 16, 13, 13, 3, 16, 14, 4, 2, 12, 7, 4, 12, 2, 19, 15, 5]
# # print(len(args))
# brgs = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
# # print(len(brgs))
# ## Test Encoder Stack
# dataset_creator = NMTDataset('seq-seq')
# BUFFER_SIZE = len(train)
# num_examples = len(train)
# BATCH_SIZE = 4
# train_dataset, val_dataset, inp_seq, targ_seq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE, train)
# # print(train_dataset,"AAAAAAAAAAAAA")
# # print(next(iter(train_dataset)),"AAAAAAAAAAAAA")
# # # print(train.head(3))
# example_input_batch, example_target_batch = next(iter(train_dataset))
# # print(next(iter(train_dataset)),"AAAAAAAAAAAAA")
# print(example_input_batch.shape, example_target_batch.shape) 

# vocab_inp_size = 22
# vocab_tar_size = 22
# max_length_input = example_input_batch.shape[1]
# max_length_output = example_target_batch.shape[1]

# embedding_dim = 1
# units = 32
# steps_per_epoch = num_examples//BATCH_SIZE

