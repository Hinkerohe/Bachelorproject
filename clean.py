from os import sep, write

from pandas.core.frame import DataFrame
# from Bio import SeqIO
import csv
import pandas
import numpy
import matplotlib.pyplot as plt


# com_json = pandas.read_json("./combine.json")
# test_20A1 = pandas.read_csv('./InOutPairs/20A1.tsv', sep= '\t')
# full_data = pandas.read_csv('./InOutPairs/FullTable.tsv', sep= '\t')

# a = pandas.read_csv('./InOutPairs/19A20AFull.tsv', sep= '\t')
# b = pandas.read_csv('./InOutPairs/19A19BFull.tsv', sep= '\t')
# c = pandas.read_csv('./InOutPairs/20A1Full.tsv', sep= '\t')
# d = pandas.read_csv('./InOutPairs/20A2Full.tsv', sep= '\t')

# e = pandas.read_csv('./InOutPairs/20A3Full.tsv', sep= '\t')
# f = pandas.read_csv('./InOutPairs/20A4Full.tsv', sep= '\t')
# g = pandas.read_csv('./InOutPairs/20A5Full.tsv', sep= '\t')
# h = pandas.read_csv('./InOutPairs/20A6Full.tsv', sep= '\t')

# i = pandas.read_csv('./InOutPairs/20A7Full.tsv', sep= '\t')
# j = pandas.read_csv('./InOutPairs/20C8Full.tsv', sep= '\t')
# k = pandas.read_csv('./InOutPairs/20C9Full.tsv', sep= '\t')
# l = pandas.read_csv('./InOutPairs/20C10Full.tsv', sep= '\t')

# m = pandas.read_csv('./InOutPairs/20C11Full.tsv', sep= '\t')
# n = pandas.read_csv('./InOutPairs/20B12Full.tsv', sep= '\t')
# o = pandas.read_csv('./InOutPairs/20B13Full.tsv', sep= '\t')
# p = pandas.read_csv('./InOutPairs/20B14Full.tsv', sep= '\t')

# q = pandas.read_csv('./InOutPairs/20B15Full.tsv', sep= '\t')
# r = pandas.read_csv('./InOutPairs/20D16Full.tsv', sep= '\t')


# , "./cleanData/19A20A.tsv"
# , "./cleanData/19A20AFull.tsv"
# , "./cleanData/19A19B.tsv"
# , "./cleanData/19A19BFull.tsv"
# , "./cleanData/20A1.tsv"
# , "./cleanData/20A1Full.tsv"
# , "./cleanData/20A2.tsv"
# , "./cleanData/20A2Full.tsv"
# , "./cleanData/20A3.tsv"
# , "./cleanData/20A3Full.tsv"
# , "./cleanData/20A4.tsv"
# , "./cleanData/20A4Full.tsv"
# , "./cleanData/20A5.tsv"
# , "./cleanData/20A5Full.tsv"
# , "./cleanData/20A6.tsv"
# , "./cleanData/20A6Full.tsv"
# , "./cleanData/20A7.tsv"
# , "./cleanData/20A7Full.tsv"
# , "./cleanData/20C8.tsv"
# , "./cleanData/20C8Full.tsv"
# , "./cleanData/20C9.tsv"
# , "./cleanData/20C9Full.tsv"
# , "./cleanData/20C11.tsv"
# , "./cleanData/20C11Full.tsv"
# , "./cleanData/20C10.tsv"
# , "./cleanData/20C10Full.tsv"
# , "./cleanData/20B12.tsv"
# , "./cleanData/20B12Full.tsv"
# , "./cleanData/20B13.tsv"
# , "./cleanData/20B13Full.tsv"
# , "./cleanData/20B14.tsv"
# , "./cleanData/20B14Full.tsv"
# , "./cleanData/20B15.tsv"
# , "./cleanData/20B15Full.tsv"
# , "./cleanData/20D16.tsv"
# , "./cleanData/20D16Full.tsv"


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

def row_numbers (file : DataFrame):
    """Gives a list of all rows where at least one sequence isn't the length of 12073"""
    # checks length and if they are the same
    lengths = file.applymap(lambda x: len(str(x)))
    bad_seq = numpy.where((lengths['seq_x'] != 1274) | (lengths['seq_y'] != 1274) | (file['seq_x'] == file['seq_y']), lengths.index, -1)
    return bad_seq


def delete_rows(file, index_list):
    """deletes all bad rows"""
    for row in index_list:
        if row != -1:
            file = file.drop(row)
    return file


def clean_data(file):
    print('Starting . . .')
    index_list = row_numbers(file)
    print('first clean')
    result = delete_rows(file, index_list)
    print('first delete')
    sub = result.apply(lambda row: levenshtein_substitution(row.seq_x, row.seq_y), axis= 1)
    lev_sub = numpy.where(sub>= 100, sub.index, -1)
    print('finished levenshtein')
    result = delete_rows(result, lev_sub)
    print('finished cleaning')
    return result

# file_list = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]
name_list = ["./cleanData/19A20A.tsv"
            , "./cleanData/19A20AFull.tsv"
            , "./cleanData/19A19B.tsv"
            , "./cleanData/19A19BFull.tsv"
            , "./cleanData/20A1.tsv"
            , "./cleanData/20A1Full.tsv"
            , "./cleanData/20A2.tsv"
            , "./cleanData/20A2Full.tsv"
            , "./cleanData/20A3.tsv"
            , "./cleanData/20A3Full.tsv"
            , "./cleanData/20A4.tsv"
            , "./cleanData/20A4Full.tsv"
            , "./cleanData/20A5.tsv"
            , "./cleanData/20A5Full.tsv"
            , "./cleanData/20A6.tsv"
            , "./cleanData/20A6Full.tsv"
            , "./cleanData/20A7.tsv"
            , "./cleanData/20A7Full.tsv"
            , "./cleanData/20C8.tsv"
            , "./cleanData/20C8Full.tsv"
            , "./cleanData/20C9.tsv"
            , "./cleanData/20C9Full.tsv"
            , "./cleanData/20C10.tsv"
            , "./cleanData/20C10Full.tsv"
            , "./cleanData/20C11.tsv"
            , "./cleanData/20C11Full.tsv"
            , "./cleanData/20B12.tsv"
            , "./cleanData/20B12Full.tsv"
            , "./cleanData/20B13.tsv"
            , "./cleanData/20B13Full.tsv"
            , "./cleanData/20B14.tsv"
            , "./cleanData/20B14Full.tsv"
            , "./cleanData/20B15.tsv"
            , "./cleanData/20B15Full.tsv"
            , "./cleanData/20D16.tsv"
            , "./cleanData/20D16Full.tsv"]


def clean (files, names):
    index = 0
    for i in files:
        print(index/2)
        result = clean_data(i)
        with open( names[index], "w") as I:
            result.to_csv(I,sep = '\t', columns=['seq_x', 'seq_y'], header= ['In', 'Out'])
        index +=1
        with open( names[index], "w") as I:
            result.to_csv(I,sep = '\t')
        index +=1


# clean(file_list, name_list)


# s = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT*'
# print (len(s))

"""For plotting the levenshtein distance:"""



# a = pandas.read_csv('./cleanData/19A20A.tsv', sep= '\t')
# b = pandas.read_csv('./cleanData/19A19B.tsv', sep= '\t')
# c = pandas.read_csv('./cleanData/20A1.tsv', sep= '\t')
# d = pandas.read_csv('./cleanData/20A2.tsv', sep= '\t')

# e = pandas.read_csv('./cleanData/20A3.tsv', sep= '\t')
# f = pandas.read_csv('./cleanData/20A4.tsv', sep= '\t')
# g = pandas.read_csv('./cleanData/20A5.tsv', sep= '\t')
# h = pandas.read_csv('./cleanData/20A6.tsv', sep= '\t')

# i = pandas.read_csv('./cleanData/20A7.tsv', sep= '\t')
# j = pandas.read_csv('./cleanData/20C8.tsv', sep= '\t')
# k = pandas.read_csv('./cleanData/20C9.tsv', sep= '\t')
# l = pandas.read_csv('./cleanData/20C10.tsv', sep= '\t')

# m = pandas.read_csv('./cleanData/20C11.tsv', sep= '\t')
# n = pandas.read_csv('./cleanData/20B12.tsv', sep= '\t')
# o = pandas.read_csv('./cleanData/20B13.tsv', sep= '\t')
# p = pandas.read_csv('./cleanData/20B14.tsv', sep= '\t')

# q = pandas.read_csv('./cleanData/20B15.tsv', sep= '\t')
# r = pandas.read_csv('./cleanData/20D16.tsv', sep= '\t')




def raw_data(file):
    sub = file.apply(lambda row: levenshtein_substitution(row.In, row.Out), axis= 1)
    result = sub.value_counts() 
    result = result.sort_index()
    result = result.to_dict()
    return result
# sd = {i : 0 for i in range(0,100)}
# file_list = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]
# raw_data_per_file = list(map(raw_data, file_list))

# to_plot = [(index, sum(map(lambda dict: dict.get(index, 0), raw_data_per_file))) for index in range(100)]
# print(to_plot)


to_plot_data = [(0, 0), (1, 33944), (2, 26735), (3, 19647), (4, 12905), (5, 10059), (6, 7358), (7, 7338), (8, 7831), 
(9, 7514), (10, 6785), (11, 16166), (12, 16352), (13, 10411), (14, 8532), (15, 5193), (16, 3787), (17, 2377), 
(18, 1322), (19, 1010), (20, 998), (21, 1029), (22, 933), (23, 775), (24, 681), (25, 553), (26, 957), (27, 765), 
(28, 961), (29, 964), (30, 553), (31, 707), (32, 711), (33, 389), (34, 268), (35, 366), (36, 323), (37, 305), (38, 385), 
(39, 751), (40, 1037), (41, 917), (42, 987), (43, 706), (44, 944), (45, 807), (46, 413), (47, 381), (48, 618), (49, 600), 
(50, 668), (51, 547), (52, 325), (53, 241), (54, 526), (55, 628), (56, 765), (57, 558), (58, 388), (59, 415), (60, 815), (61, 609), 
(62, 902), (63, 2381), (64, 2648), (65, 2217), (66, 1892), (67, 1357), (68, 1162), (69, 852), (70, 1017), (71, 981), (72, 1053), 
(73, 2597), (74, 2596), (75, 2113), (76, 1260), (77, 992), (78, 1102), (79, 1024), (80, 1033), (81, 1058), (82, 1460), (83, 1225), 
(84, 1019), (85, 857), (86, 915), (87, 565), (88, 368), (89, 258), (90, 1115), (91, 1017), (92, 856), (93, 718), (94, 691), (95, 660), 
(96, 1028), (97, 576), (98, 357), (99, 750)]
y_axis = []
x_axis = [i for i in range(100)]
for (key, value) in to_plot_data:
    y_axis.append(value)
print(y_axis)
plt.bar(x_axis,y_axis)
plt.xlabel('Levenshtein Number')
plt.ylabel('Number of Sequences')
plt.xticks([i for i in range(0,100,10)])
plt.savefig('Levenshtein_Diagram3.png', dpi = 1000)
# li = raw_data(c, sd)
# print(li)