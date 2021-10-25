from Bio import SeqIO
import csv
import pandas
import numpy

""" Experimenting with fasta and tsv files """

"""fasta files"""
fasta_sequence = SeqIO.parse("sars_cov_spike_protein_data\spikeprot.fasta", "fasta") 
# for record in fasta_sequence:
#     print(record.id)


# only for files with only one record
# fasta_read = SeqIO.read("sars_cov_spike_protein_data\spikeprot.fasta", "fasta") only for files with only one record


# only the first record 
first_record = next(fasta_sequence)
# print (first_record)


id_list = first_record.id.split("|")
"""['Spike', 'hCoV-19/Wuhan/WIV04/2019', '2019-12-30', 'EPI_ISL_402124', 'Original', 'hCoV-19^^Hubei', 'Human', 'Wuhan']"""

decrp_list = first_record.description.split("|")

# print(decrp_list)
"""['Spike', 'hCoV-19/Wuhan/WIV04/2019', '2019-12-30', 'EPI_ISL_402124', 
    'Original', 'hCoV-19^^Hubei', 'Human', 'Wuhan Jinyintan Hospital', 
    'Wuhan Institute of Virology', 'Shi', 'China']"""

count = 0
a=0
for record in fasta_sequence:
    # record_id_list = record.description.split("|")
    count+= 1
    # if '2019' in record_id_list[2]:
    #     print (record_id_list)
    #     count+= 1
print (count)


"""tsv files"""

tsv_file = open("sars_cov_spike_protein_data\hcov_global.tsv")
tsv_table = csv.reader(tsv_file, delimiter = "\t")
first_tsv_line = next(tsv_file)
# print (first_tsv_line)
i = 1
for line in tsv_file:
    if i>0 :
        # print (line)
        i-=1
    else:
        break


tsv_data = pandas.read_csv("sars_cov_spike_protein_data\hcov_global.tsv", sep = "\t")

# print(tsv_data)



"""tsv files

strain	decrp_list[1]
virus	
gisaid_epi_isl	decrp_list[3]
genbank_accession	
date	decrp_list[2]
region	
country	
division	
location	
region_exposure	
country_exposure	decrp_list[10]
division_exposure	
segment	
length	
host	decrp_list[6]
age	
sex	
Nextstrain_clade	
pango_lineage	
GISAID_clade	
originating_lab	decrp_list[7] 
submitting_lab	decrp_list[8]
authors	
url	
title	
paper_url	
date_submitted	
purpose_of_sequencing	
variant
"""