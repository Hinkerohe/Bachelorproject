from os import sep
from Bio import SeqIO
import csv
import pandas
import numpy

"""Important fields are: Sequence ID, clade and Sequence"""

tsv_file = pandas.read_csv("./sars_cov_spike_protein_data/hcov_global.tsv", sep = '\t')
fasta_file = SeqIO.parse("./sars_cov_spike_protein_data/spikeprot.fasta", "fasta")

"""# Important information from fasta"""
# with open('./spikeprot.tsv', 'w') as spikeprot_file:
#     writer = csv.DictWriter(spikeprot_file, fieldnames = ['gisaid_epi_isl','seq'], delimiter = "\t")
#     writer.writeheader()
#     for record in fasta_file:
#         decrp_list = record.description.split("|")
#         seq_ID = decrp_list[3]
#         Seq = record.seq
#         writer.writerow({'gisaid_epi_isl':seq_ID, 'seq': str(Seq)})


"""# Important information from tsv"""
# with open('./small_hcov.tsv', 'w') as small_hcov_global:
#     seq_id = tsv_file.columns[2]
#     clade = tsv_file.columns[17]
#     headers = [seq_id,clade]
#     tsv_file.to_csv(small_hcov_global, sep = '\t', columns=[seq_id,clade], header = headers)



""" to combine the tables"""
# tsv_hcov = pandas.read_csv("./small_hcov.tsv", sep = '\t')
# tsv_spikeprot = pandas.read_csv("./spikeprot.tsv", sep = '\t')

# combined = tsv_hcov.merge(tsv_spikeprot)
# with open('./combined.tsv', 'w') as combi:
#     combined.to_csv(combi, sep = '\t')



""" Second Table """

com_json = pandas.read_json("./combine.json")
com_tsv = pandas.read_csv("./combined1.tsv", sep = '\t')
records = com_json.to_records()
clade_list = set(map(lambda x: x[2], records))
In_tsv =pandas.read_csv('./InOutPairs/In.tsv', sep = '\t')
Out_tsv =pandas.read_csv('./InOutPairs/Out.tsv', sep = '\t')
"""{nan, '19A', '19B', '20B', '20E (EU1)',
 '20F', '21F (Iota)', '21B (Kappa)', '20A', '21D (Eta)', 
 '20H (Beta, V2)', '20D', '21C (Epsilon)', '20I (Alpha, V1)', '21A (Delta)', 
 '21H', '20G', '20C', '21G (Lambda)', '20J (Gamma, V3)'}"""



"""merging"""


clade_20A = com_json[com_json['Nextstrain_clade'] == "20A"]
clade_19A = com_json[com_json['Nextstrain_clade'] == "19A"]
clade_19B = com_json[com_json['Nextstrain_clade'] == "19B"]
clade_20B = com_json[com_json['Nextstrain_clade'] == "20B"]
clade_20C = com_json[com_json['Nextstrain_clade'] == "20C"]
clade_20E = com_json[com_json['Nextstrain_clade'] == "20E (EU1)"]
clade_20F = com_json[com_json['Nextstrain_clade'] == "20F"]
clade_21F = com_json[com_json['Nextstrain_clade'] == "21F (Iota)"]
clade_21B = com_json[com_json['Nextstrain_clade'] == "21B (Kappa)"]
clade_21D = com_json[com_json['Nextstrain_clade'] == "21D (Eta)"]
clade_20H = com_json[com_json['Nextstrain_clade'] == "20H (Beta, V2)"]
clade_20D = com_json[com_json['Nextstrain_clade'] == "20D"]
clade_21C = com_json[com_json['Nextstrain_clade'] == "21C (Epsilon)"]
clade_20I = com_json[com_json['Nextstrain_clade'] == "20I (Alpha, V1)"]
clade_21A = com_json[com_json['Nextstrain_clade'] == "21A (Delta)"]
clade_21H = com_json[com_json['Nextstrain_clade'] == "21H"]
clade_20G = com_json[com_json['Nextstrain_clade'] == "20G"]
clade_21G = com_json[com_json['Nextstrain_clade'] == "21G (Lambda)"]
clade_20J = com_json[com_json['Nextstrain_clade'] == "20J (Gamma, V3)"]

c =clade_19A

d =clade_20A
"""merged with 6 columns"""
InOut = clade_19A.merge(clade_20A, how='cross')
InOut2 = clade_19A.merge(clade_19B, how='cross')

InOut3 = clade_20A.merge(clade_20C, how='cross')
InOut4 = clade_20A.merge(clade_20E, how='cross')
InOut5 = clade_20A.merge(clade_20B, how='cross')
InOut6 = clade_20A.merge(clade_21D, how='cross')
InOut7 = clade_20A.merge(clade_21A, how='cross')
InOut8 = clade_20A.merge(clade_21B, how='cross')
InOut9 = clade_20A.merge(clade_21H, how='cross')

InOut12 = clade_20C.merge(clade_21C, how='cross')
InOut13 = clade_20C.merge(clade_20H, how='cross')
InOut14 = clade_20C.merge(clade_20G, how='cross')
InOut15 = clade_20C.merge(clade_21F, how='cross')

InOut23 = clade_20B.merge(clade_20F, how='cross')
InOut24 = clade_20B.merge(clade_20D, how='cross')
InOut25 = clade_20B.merge(clade_20J, how='cross')
InOut26 = clade_20B.merge(clade_20I, how='cross')
InOut27 = clade_20D.merge(clade_21G, how='cross')


# """gisaid_epi_isl	Nextstrain_clade	seq"""
# In  = InOut[['gisaid_epi_isl_x','Nextstrain_clade_x','seq_x']]
# Out = InOut[['gisaid_epi_isl_y','Nextstrain_clade_y','seq_y']]

InOut.append(InOut2)
InOut.append(InOut3)
InOut.append(InOut4)
InOut.append(InOut5)
InOut.append(InOut6)
InOut.append(InOut7)
InOut.append(InOut8)
InOut.append(InOut9)
InOut.append(InOut12)
InOut.append(InOut13)
InOut.append(InOut14)
InOut.append(InOut15)
InOut.append(InOut23)
InOut.append(InOut24)
InOut.append(InOut25)
InOut.append(InOut26)
InOut.append(InOut27)


with open( "./InOutPairs/InOutPairs.tsv", "w") as I:
    InOut.to_csv(I,sep = '\t', columns=['seq_x', 'seq_y'], header= ['In', 'Out'])

with open( "./InOutPairs/FullTable.tsv", "w") as I:
    InOut.to_csv(I,sep = '\t')


# print(In)
# a = In.stack()
# b = Out.stack()


# with open( "./InOutPairs/In.tsv", "w") as I:
#     a.to_csv(I, sep = '\t', header = ['in'])

# with open( "./InOutPairs/Out.tsv", "w") as O:
#     b.to_csv(O, sep = '\t', header = ['out'])


# with open( "./InOutPairs/InOutPairs.json", "w") as I:
#     InOut.to_json(I)
