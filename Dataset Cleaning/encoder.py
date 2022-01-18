from os import sep
# from bio import SeqIO
import csv
import pandas
import numpy
import itertools

from pandas.core.frame import DataFrame

amino_acids = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

test_data = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRHLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
test_data2 = 'XXXXXXXXXXXSSQCVNLTTRTQLPXXXXXXXXXXVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEXXVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGXXKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXSIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
test_data_set = pandas.read_csv('./cleanData/19A20A.tsv', sep= '\t')

amino_acids_dict = dict(map(lambda it: (it[1], it[0]), enumerate([ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])))

def k_mer_ID_list (k):
    amino_acids = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    cross_join_list = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    full_name = './test/k_mer_ID_list.txt' 
    for i in range(1,k):
        cross_join_list = [x + y for x, y in itertools.product(cross_join_list, amino_acids)]
    with open( full_name, "w") as I:
        print(cross_join_list, file=I )
    return cross_join_list


def k_mer(k,seq):
    if type(seq)  == int:
        return None
    length = len(seq)
    encode = amino_acids
    k_mers_encoded = [0 for _ in range(length - k)]

    def unique_encode(x):
        k = len(x) - 1
        encoded_number = 0
        for i in x:
            if i == 'X':
                encoded_number = -1
                return encoded_number
            elif i == '*':
                return encoded_number
            number = amino_acids_dict[i]
            encoded_number += number * 20**k
            k -= 1
        return encoded_number

    for i in range(0, length - k):
        k_mer_seq = seq[i: i + k]
        k_mers_encoded[i] = unique_encode(k_mer_seq)
    # print('encoded')
    return k_mers_encoded

def kmer_decode_to_sequence(k, seq):

    def find_sequence(k, enc_seq):
        amino_acids = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        new_decoded_seq = []
        while k > 0:
            number = 20**(k-1)
            i = 0
            while i < 21:
                if enc_seq - number*i < 0:
                    new_decoded_seq.append(amino_acids[i-1])
                    enc_seq = enc_seq - number * (i-1)
                    i = 21
                else:
                    i+=1
            k -= 1
            print(k,new_decoded_seq)
        return new_decoded_seq
            
    result = []
    j = 0
    for i in seq:
        if j % k ==0:
            decoded = find_sequence(k, i)
            result += decoded
            j+=1
        else:
            j+=1
    return result



def ordinal(seq):
    """ each unique category value is assigned an integer value, here the index of the letter in the list"""
    amino_acids = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    encoded_seq = list(map(lambda x: amino_acids.index(x), seq))
    return encoded_seq

def one_hot(seq):
    ordinal_encoded_seq = ordinal(seq)
    def to_binary(number):
        letter = [0 for _ in range(0,20)]
        letter[number] = 1
        return letter
    one_hot_encoded_seq = list(map(lambda x: to_binary(x), ordinal_encoded_seq))
    return one_hot_encoded_seq

def encoding(kmer : bool, ordnial : bool, onehot : bool, seq):
    if kmer == True:
        result= k_mer(25,seq)
    if ordnial == True:
        result = ordinal(seq)
    if onehot == True:
        result = one_hot(seq)
    return result

# print(ordinal(test_data))
# print(one_hot(test_data))
k = 10
seq = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# to_decode = k_mer(k,seq)
# print(to_decode)
# print(kmer_decode_to_sequence(k, to_decode))

 
def encode_full_table(name, dataset:DataFrame):


    print('finished list')
    encoded_result = dataset.apply(lambda row: k_mer(5,row.In), axis= 1)
    with open( name, "w") as I:
            encoded_result.to_csv(I,sep = '\t')
    return encoded_result


encode_full_table('./test/encoded4.tsv', test_data_set)